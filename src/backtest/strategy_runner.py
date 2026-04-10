"""
Strategy Replay Runner — Phase 6 Step 6.3 of the Trading Decision Support System.

Connects DataReplayEngine, TechnicalAggregator, RegimeDetector,
SignalGenerator, and TradeSimulator into a single backtest loop.

At each bar the runner:
1. Updates open positions (SL / target / trailing)
2. Runs technical analysis on look-ahead-safe price history
3. Optionally runs anomaly detection (FULL mode)
4. Detects the current market regime
5. Generates a trading signal
6. Executes entry if the signal passes all filters

Usage
-----
::

    runner = StrategyRunner(db)
    config = BacktestConfig(
        index_id="NIFTY50",
        start_date=date(2024, 1, 1),
        end_date=date(2024, 6, 30),
        timeframe="1d",
        mode="TECHNICAL_ONLY",
    )
    result = runner.run(config)
    print(f"Trades: {result.total_trades}, Return: {result.total_return_pct:.2f}%")
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Optional

from src.analysis.anomaly.anomaly_aggregator import AnomalyAggregator, AnomalyVote
from src.analysis.news.news_engine import NewsVote
from src.analysis.technical_aggregator import TechnicalAggregator
from src.backtest.data_replay import DataReplayEngine, ReplayIterator, ReplaySession
from src.backtest.trade_simulator import (
    ClosedTrade,
    EquityPoint,
    PortfolioState,
    SimulatorConfig,
    TradeExecution,
    TradeSimulator,
)
from src.database.db_manager import DatabaseManager
from src.engine.regime_detector import RegimeDetector
from src.engine.signal_generator import SignalGenerator, TradingSignal

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Confidence ranking — higher is better
# ---------------------------------------------------------------------------

_CONFIDENCE_RANK = {"LOW": 1, "MEDIUM": 2, "HIGH": 3}

# Valid strategy modes
_VALID_MODES = {"FULL", "TECHNICAL_ONLY", "TECHNICAL_OPTIONS", "CUSTOM"}


def _confidence_rank(level: str) -> int:
    """Return numeric rank for a confidence level string."""
    return _CONFIDENCE_RANK.get(level.upper(), 0)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class BacktestConfig:
    """All parameters needed to execute a single backtest run."""

    # What to test
    index_id: str
    start_date: date
    end_date: date
    timeframe: str = "1d"
    benchmark_id: str = "NIFTY50"
    warmup_bars: int = 250

    # Simulator settings
    simulator_config: Optional[SimulatorConfig] = None

    # Strategy mode
    #   FULL:              technical + options + news (simulated) + anomaly
    #   TECHNICAL_ONLY:    technical indicators only (fastest)
    #   TECHNICAL_OPTIONS: technical + options data
    #   CUSTOM:            user provides a custom signal function
    mode: str = "TECHNICAL_ONLY"

    # Signal filters
    min_confidence: str = "LOW"
    signal_types: list[str] = field(
        default_factory=lambda: ["BUY_CALL", "BUY_PUT"]
    )

    # Execution rules
    signal_cooldown_bars: int = 3
    max_signals_per_day: int = 5

    # Anomaly settings (FULL mode)
    anomaly_timeframe: Optional[str] = None

    # Progress
    show_progress: bool = True
    progress_interval: int = 50

    def __post_init__(self) -> None:
        if self.mode not in _VALID_MODES:
            raise ValueError(
                f"Invalid mode {self.mode!r}. "
                f"Must be one of {sorted(_VALID_MODES)}"
            )
        if self.simulator_config is None:
            self.simulator_config = SimulatorConfig()


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------


@dataclass
class BacktestResult:
    """Complete output of a single backtest run."""

    # Configuration
    config: BacktestConfig

    # Session info
    index_id: str
    start_date: date
    end_date: date
    total_bars: int
    trading_days: int

    # Trade results
    trade_history: list[ClosedTrade]
    total_trades: int

    # Signals
    total_signals_generated: int
    actionable_signals: int
    executed_trades: int

    # Core metrics — populated by Step 6.4 metrics calculator
    metrics: Optional[dict] = None

    # Equity curve
    equity_curve: list[EquityPoint] = field(default_factory=list)

    # Capital
    initial_capital: float = 0.0
    final_capital: float = 0.0
    total_return_pct: float = 0.0

    # Timing
    backtest_duration_seconds: float = 0.0
    bars_per_second: float = 0.0

    # Data quality
    data_quality_score: float = 1.0
    warnings: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _simulate_news_vote(
    mode: str, index_id: str, timestamp: datetime
) -> Optional[NewsVote]:
    """Simulate a news vote for backtesting.

    Historical news archives are not available, so:
    - TECHNICAL_ONLY / TECHNICAL_OPTIONS → None
    - FULL → NEUTRAL with low confidence (the default most-of-the-time
      condition).
    """
    if mode in ("TECHNICAL_ONLY", "TECHNICAL_OPTIONS"):
        return None

    # FULL mode: return neutral news (no strong news)
    return NewsVote(
        index_id=index_id,
        timestamp=timestamp,
        vote="NEUTRAL",
        confidence=0.3,
        active_article_count=0,
        weighted_sentiment=0.0,
        top_headline=None,
        event_regime="NORMAL",
        reasoning="Simulated neutral news for backtest (no historical archive)",
    )


def _run_anomaly_detection(
    aggregator: AnomalyAggregator,
    index_id: str,
    current_bar: dict,
    price_history,
    timeframe: Optional[str] = None,
) -> Optional[AnomalyVote]:
    """Run anomaly detection on historical price/volume data.

    Anomaly detection IS possible historically because it only needs
    price and volume data which we have.  Returns None on any error
    (graceful degradation).
    """
    try:
        recent = price_history.tail(20)
        recent_bars = recent.to_dict("records")
        result = aggregator.run_detection_cycle(
            index_id=index_id,
            current_price_bar=current_bar,
            recent_price_bars=recent_bars,
            timeframe=timeframe,
        )
        if result is None:
            return None
        return AnomalyVote(
            index_id=result.index_id,
            vote=result.anomaly_vote,
            confidence=result.anomaly_confidence,
            risk_level=result.risk_level,
            position_size_modifier=result.position_size_modifier,
            active_alerts=result.active_alert_count,
            primary_alert_message=(
                result.primary_alert.message if result.primary_alert else None
            ),
            reasoning=result.summary,
        )
    except Exception:
        logger.debug(
            "Anomaly detection failed for %s — skipping", index_id, exc_info=True
        )
        return None


def _parse_options(options_snapshot: Optional[dict]):
    """Convert a raw options snapshot dict to OptionsChainData for the aggregator.

    Returns *None* when the snapshot is missing or empty.
    """
    if not options_snapshot:
        return None

    try:
        from datetime import date as _date

        from src.data.options_chain import OptionsChainData, OptionStrike

        strikes_raw = options_snapshot.get("strikes", [])
        if not strikes_raw:
            return None

        strikes = tuple(
            OptionStrike(
                strike_price=s.get("strike_price", 0),
                ce_oi=s.get("ce_oi", 0),
                ce_oi_change=s.get("ce_oi_change", 0),
                ce_volume=s.get("ce_volume", 0),
                ce_ltp=s.get("ce_ltp", 0.0),
                ce_iv=s.get("ce_iv", 0.0),
                pe_oi=s.get("pe_oi", 0),
                pe_oi_change=s.get("pe_oi_change", 0),
                pe_volume=s.get("pe_volume", 0),
                pe_ltp=s.get("pe_ltp", 0.0),
                pe_iv=s.get("pe_iv", 0.0),
            )
            for s in strikes_raw
        )

        snap_ts = options_snapshot.get("snapshot_timestamp", "")
        spot = strikes_raw[0].get("spot_price", 0.0) if strikes_raw else 0.0

        return OptionsChainData(
            index_id=options_snapshot.get("index_id", ""),
            spot_price=spot,
            timestamp=datetime.fromisoformat(snap_ts) if snap_ts else datetime.now(),
            expiry_date=_date.today(),
            strikes=strikes,
            available_expiries=(
                _date.today(),
            ),
        )
    except Exception:
        logger.debug("Failed to parse options snapshot", exc_info=True)
        return None


def _build_result(
    config: BacktestConfig,
    session: ReplaySession,
    simulator: TradeSimulator,
    signals_generated: list[dict],
    wall_seconds: float,
    warnings: list[str],
) -> BacktestResult:
    """Assemble the final BacktestResult from a completed run."""
    state = simulator.get_portfolio_state()
    initial = config.simulator_config.initial_capital
    final = state.current_capital + state.unrealized_pnl
    total_bars = session.total_bars

    return BacktestResult(
        config=config,
        index_id=config.index_id,
        start_date=session.actual_start,
        end_date=session.actual_end,
        total_bars=total_bars,
        trading_days=session.trading_days,
        trade_history=simulator.trade_history,
        total_trades=len(simulator.trade_history),
        total_signals_generated=total_bars,
        actionable_signals=len(signals_generated),
        executed_trades=len(signals_generated),
        equity_curve=list(simulator.equity_curve),
        initial_capital=initial,
        final_capital=round(final, 2),
        total_return_pct=round(((final - initial) / initial) * 100, 4)
        if initial
        else 0.0,
        backtest_duration_seconds=round(wall_seconds, 3),
        bars_per_second=round(total_bars / wall_seconds, 1)
        if wall_seconds > 0
        else 0.0,
        data_quality_score=session.data_quality_score,
        warnings=warnings,
    )


# ---------------------------------------------------------------------------
# Strategy Runner
# ---------------------------------------------------------------------------


class StrategyRunner:
    """Orchestrates a full backtest by wiring together all DSS components.

    Parameters
    ----------
    db:
        DatabaseManager instance used by every sub-component.
    """

    def __init__(self, db: DatabaseManager) -> None:
        self.db = db
        self.replay_engine = DataReplayEngine(db)
        self.technical_aggregator = TechnicalAggregator()
        self.regime_detector = RegimeDetector(db)
        self.signal_generator = SignalGenerator(db)

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(self, config: BacktestConfig) -> BacktestResult:
        """Execute a complete backtest.

        Parameters
        ----------
        config:
            Full backtest configuration (index, dates, mode, filters, etc.).

        Returns
        -------
        BacktestResult
            Trade history, equity curve, summary statistics.
        """
        t0 = time.monotonic()
        warnings: list[str] = []

        if config.mode == "CUSTOM":
            raise NotImplementedError(
                "CUSTOM mode requires a user-provided signal function — "
                "not yet supported."
            )

        # ---- Step 1: Prepare replay session ----
        logger.info(
            "Preparing replay: %s %s→%s (%s, mode=%s)",
            config.index_id,
            config.start_date,
            config.end_date,
            config.timeframe,
            config.mode,
        )
        session = self.replay_engine.prepare_replay(
            index_id=config.index_id,
            start_date=config.start_date,
            end_date=config.end_date,
            timeframe=config.timeframe,
            benchmark_id=config.benchmark_id,
            warmup_bars=config.warmup_bars,
        )

        if session.total_bars == 0:
            warnings.append("No tradeable bars after warmup — empty result")
            return _build_result(
                config, session, TradeSimulator(config.simulator_config),
                [], time.monotonic() - t0, warnings,
            )

        if session.total_bars == 1:
            warnings.append(
                "Only 1 tradeable bar — insufficient for meaningful backtest"
            )

        # Check options data availability for modes that need it
        if config.mode in ("FULL", "TECHNICAL_OPTIONS") and not session.has_options_data:
            warnings.append(
                f"Options data unavailable for {config.index_id} in this period. "
                "Degrading to TECHNICAL_ONLY internally."
            )
            effective_mode = "TECHNICAL_ONLY"
        else:
            effective_mode = config.mode

        # ---- Step 2: Initialize simulator ----
        simulator = TradeSimulator(config.simulator_config)

        # ---- Step 3: Initialize anomaly detector (FULL mode) ----
        anomaly_aggregator: Optional[AnomalyAggregator] = None
        if effective_mode == "FULL":
            anomaly_aggregator = AnomalyAggregator(
                self.db,
                timeframe=config.anomaly_timeframe or config.timeframe,
            )

        # ---- Step 4: Main bar-by-bar loop ----
        signals_generated: list[dict] = []
        last_signal_bar: int = -999
        daily_signal_count: dict[date, int] = {}
        current_date: Optional[date] = None
        min_conf_rank = _confidence_rank(config.min_confidence)
        bars_processed = 0

        if config.show_progress:
            print(
                f"\n{'='*60}\n"
                f"  Backtest: {config.index_id} | {session.actual_start} → {session.actual_end}\n"
                f"  Mode: {effective_mode} | Bars: {session.total_bars} | "
                f"Warmup: {config.warmup_bars}\n"
                f"{'='*60}"
            )

        iterator = ReplayIterator(self.replay_engine, session)

        for time_slice in iterator:
            bars_processed += 1

            # --- Track day changes ---
            bar_date = (
                time_slice.timestamp.date()
                if hasattr(time_slice.timestamp, "date")
                and callable(time_slice.timestamp.date)
                else time_slice.timestamp
            )
            if bar_date != current_date:
                current_date = bar_date
                daily_signal_count.setdefault(current_date, 0)

            # --- Step 4a: Update open positions (SL / target / trailing) ---
            try:
                simulator.update_positions(
                    time_slice.current_bar, time_slice.timestamp
                )
            except Exception:
                logger.debug(
                    "update_positions failed at bar %d", time_slice.bar_index,
                    exc_info=True,
                )

            # --- Step 4b: Technical analysis ---
            try:
                options_chain = (
                    _parse_options(time_slice.options_snapshot)
                    if effective_mode in ("FULL", "TECHNICAL_OPTIONS")
                    else None
                )
                tech_result = self.technical_aggregator.analyze(
                    index_id=config.index_id,
                    price_df=time_slice.price_history,
                    options_chain=options_chain,
                    vix_value=time_slice.vix_value,
                    benchmark_df=time_slice.benchmark_history,
                    timeframe=config.timeframe,
                )
            except Exception:
                logger.debug(
                    "Technical analysis failed at bar %d — skipping",
                    time_slice.bar_index,
                    exc_info=True,
                )
                continue

            if tech_result is None:
                continue

            # --- Step 4c: News vote (simulated) ---
            news_vote = _simulate_news_vote(
                effective_mode, config.index_id, time_slice.timestamp
            )

            # --- Step 4d: Anomaly detection (FULL mode) ---
            anomaly_vote: Optional[AnomalyVote] = None
            if effective_mode == "FULL" and anomaly_aggregator is not None:
                anomaly_vote = _run_anomaly_detection(
                    anomaly_aggregator,
                    config.index_id,
                    time_slice.current_bar,
                    time_slice.price_history,
                    config.anomaly_timeframe or config.timeframe,
                )

            # --- Step 4e: Regime detection ---
            try:
                regime = self.regime_detector.detect_regime(
                    index_id=config.index_id,
                    price_df=time_slice.price_history,
                    technical_result=tech_result,
                    news_event_modifier=None,
                    anomaly_result=None,
                    vix_value=time_slice.vix_value,
                )
            except Exception:
                logger.debug(
                    "Regime detection failed at bar %d — skipping",
                    time_slice.bar_index,
                    exc_info=True,
                )
                continue

            # --- Step 4f: Signal generation ---
            try:
                signal = self.signal_generator.generate_signal(
                    index_id=config.index_id,
                    technical=tech_result,
                    news=news_vote,
                    anomaly=anomaly_vote,
                    regime=regime,
                    current_spot_price=time_slice.current_bar["close"],
                )
            except Exception:
                logger.debug(
                    "Signal generation failed at bar %d — skipping",
                    time_slice.bar_index,
                    exc_info=True,
                )
                continue

            # --- Step 4g: Execution filter ---
            signal_conf_rank = _confidence_rank(signal.confidence_level)
            should_execute = (
                signal.signal_type in config.signal_types
                and signal_conf_rank >= min_conf_rank
                and (time_slice.bar_index - last_signal_bar)
                >= config.signal_cooldown_bars
                and daily_signal_count.get(current_date, 0)
                < config.max_signals_per_day
            )

            if should_execute:
                try:
                    execution = simulator.execute_entry(
                        signal, time_slice.current_bar
                    )
                except Exception:
                    logger.debug(
                        "execute_entry failed at bar %d",
                        time_slice.bar_index,
                        exc_info=True,
                    )
                    execution = None

                if execution is not None:
                    last_signal_bar = time_slice.bar_index
                    daily_signal_count[current_date] = (
                        daily_signal_count.get(current_date, 0) + 1
                    )
                    signals_generated.append(
                        {
                            "bar_index": time_slice.bar_index,
                            "timestamp": time_slice.timestamp,
                            "signal": signal,
                            "execution": execution,
                        }
                    )

            # --- Step 4i: Progress reporting ---
            if (
                config.show_progress
                and bars_processed % config.progress_interval == 0
            ):
                state = simulator.get_portfolio_state()
                print(
                    f"  Bar {bars_processed}/{session.total_bars} "
                    f"({time_slice.progress_pct:.0f}%) | "
                    f"Capital: \u20b9{state.current_capital:,.0f} | "
                    f"Trades: {state.total_trades} | "
                    f"Open: {state.open_position_count}"
                )

        # ---- Step 5: Force-close remaining open positions ----
        state = simulator.get_portfolio_state()
        if state.open_positions:
            logger.info(
                "Force-closing %d open positions at backtest end",
                len(state.open_positions),
            )
            # Use the last bar's close as exit price
            last_bar = iterator._session.full_price_data.iloc[-1]
            last_close = float(last_bar["close"])
            last_ts = (
                last_bar.name
                if hasattr(last_bar.name, "date")
                else datetime.now()
            )
            for pos in list(state.open_positions):
                try:
                    simulator.close_position(
                        pos, last_close, "BACKTEST_END", last_ts
                    )
                except Exception:
                    logger.debug(
                        "Failed to close position %s at backtest end",
                        pos.trade_id,
                        exc_info=True,
                    )

        wall_seconds = time.monotonic() - t0

        if config.show_progress:
            final_state = simulator.get_portfolio_state()
            initial = config.simulator_config.initial_capital
            final_cap = final_state.current_capital + final_state.unrealized_pnl
            ret = ((final_cap - initial) / initial) * 100 if initial else 0
            print(
                f"\n{'='*60}\n"
                f"  Backtest complete in {wall_seconds:.1f}s "
                f"({session.total_bars / wall_seconds:.0f} bars/sec)\n"
                f"  Trades: {final_state.total_trades} | "
                f"Signals executed: {len(signals_generated)}\n"
                f"  Capital: \u20b9{initial:,.0f} → \u20b9{final_cap:,.0f} "
                f"({ret:+.2f}%)\n"
                f"{'='*60}\n"
            )

        return _build_result(
            config, session, simulator, signals_generated, wall_seconds, warnings
        )

    # ------------------------------------------------------------------
    # Multi-index runner
    # ------------------------------------------------------------------

    def run_multi_index(
        self, configs: list[BacktestConfig]
    ) -> list[BacktestResult]:
        """Run backtests for multiple indices sequentially.

        Parameters
        ----------
        configs:
            One BacktestConfig per index to test.

        Returns
        -------
        list[BacktestResult]
            Results in the same order as *configs*.
        """
        results: list[BacktestResult] = []

        for i, cfg in enumerate(configs, 1):
            logger.info(
                "Multi-index run %d/%d: %s", i, len(configs), cfg.index_id
            )
            result = self.run(cfg)
            results.append(result)

        # Summary comparison
        if len(results) > 1:
            print(f"\n{'='*70}")
            print("  Multi-Index Backtest Summary")
            print(f"{'='*70}")
            print(
                f"  {'Index':<15} {'Trades':>7} {'Return':>10} "
                f"{'Win Rate':>10} {'Duration':>10}"
            )
            print(f"  {'-'*55}")
            for r in results:
                wins = sum(
                    1 for t in r.trade_history if t.outcome == "WIN"
                )
                win_rate = (
                    f"{wins / r.total_trades * 100:.1f}%"
                    if r.total_trades
                    else "N/A"
                )
                print(
                    f"  {r.index_id:<15} {r.total_trades:>7} "
                    f"{r.total_return_pct:>+9.2f}% "
                    f"{win_rate:>10} "
                    f"{r.backtest_duration_seconds:>8.1f}s"
                )
            print(f"{'='*70}\n")

        return results
