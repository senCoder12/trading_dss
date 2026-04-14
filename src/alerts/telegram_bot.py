"""
Interactive Telegram Bot for the Trading Decision Support System.

Provides a full command-driven interface for querying signals, portfolio,
performance, alerts, and system health during market hours.  Also sends
proactive alerts based on priority (CRITICAL/HIGH/NORMAL/LOW).

Usage
-----
::

    bot = TelegramBot(db, decision_engine)
    bot.start_bot()          # non-blocking — polls in background thread
    bot.send_alert(msg, priority="CRITICAL")
    bot.stop_bot()
"""

from __future__ import annotations

import asyncio
import logging
import queue
import threading
import time
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any, Callable, Optional

from config.constants import IST_TIMEZONE
from config.settings import settings
from src.database.db_manager import DatabaseManager

logger = logging.getLogger(__name__)

# Telegram message hard limit
_MAX_MESSAGE_LENGTH = 4096

# ---------------------------------------------------------------------------
# Priority constants
# ---------------------------------------------------------------------------

PRIORITY_CRITICAL = "CRITICAL"
PRIORITY_HIGH = "HIGH"
PRIORITY_NORMAL = "NORMAL"
PRIORITY_LOW = "LOW"


# ---------------------------------------------------------------------------
# TelegramBot
# ---------------------------------------------------------------------------


class TelegramBot:
    """Interactive Telegram bot and alert dispatcher for Trading DSS.

    Registers command handlers for live querying via phone during market hours
    and sends proactive alerts based on priority levels.

    Parameters
    ----------
    db:
        Shared :class:`DatabaseManager` instance.
    decision_engine:
        Optional reference to a live :class:`DecisionEngine`.  When ``None``,
        commands that need it will return a "system starting up" message.
    """

    def __init__(
        self,
        db: DatabaseManager,
        decision_engine: Any = None,
    ) -> None:
        self._db = db
        self._engine = decision_engine

        self._token: str = settings.telegram.bot_token or ""
        self._chat_id: str = settings.telegram.chat_id or ""
        self._configured = bool(self._token and self._chat_id)

        if not self._configured:
            logger.warning(
                "Telegram bot not configured (TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID "
                "missing). All bot methods will be no-ops."
            )

        # python-telegram-bot Application (lazy)
        self._app: Any = None
        self._running = False
        self._thread: Optional[threading.Thread] = None

        # Digest queues
        self._normal_queue: queue.Queue[str] = queue.Queue()
        self._low_queue: queue.Queue[str] = queue.Queue()

        # Rate limiting: track last send time per priority
        self._last_high_send: float = 0.0
        self._high_cooldown_seconds: float = 60.0  # min 60s between HIGH alerts

        # Start time for uptime tracking
        self._start_time: Optional[datetime] = None

    # ------------------------------------------------------------------
    # Formatting helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _format_currency(amount: float) -> str:
        """Format amount as Indian Rupee string with sign."""
        if amount >= 0:
            return f"\u20b9{amount:,.0f}"
        return f"\u20b9{amount:,.0f}"

    @staticmethod
    def _format_pct(value: float) -> str:
        """Format percentage with sign."""
        sign = "+" if value >= 0 else ""
        return f"{sign}{value:.1f}%"

    @staticmethod
    def _format_signal_emoji(signal_type: str) -> str:
        if "CALL" in signal_type.upper():
            return "\U0001f7e2"  # green circle
        if "PUT" in signal_type.upper():
            return "\U0001f534"  # red circle
        return "\u26aa"  # white circle

    @staticmethod
    def _format_confidence_emoji(level: str) -> str:
        level = (level or "").upper()
        if level == "HIGH":
            return "\U0001f7e2"
        if level == "MEDIUM":
            return "\U0001f7e1"
        return "\U0001f535"

    @staticmethod
    def _format_severity_emoji(severity: str) -> str:
        severity = (severity or "").upper()
        if severity in ("CRITICAL", "HIGH"):
            return "\U0001f534"
        if severity == "MEDIUM":
            return "\U0001f7e1"
        return "\u26aa"

    @staticmethod
    def _truncate(text: str, max_len: int = 200) -> str:
        if len(text) <= max_len:
            return text
        return text[: max_len - 3] + "..."

    @staticmethod
    def _human_timedelta(td: timedelta) -> str:
        """Convert a timedelta to a compact human string like '2h 15m'."""
        total_seconds = int(td.total_seconds())
        if total_seconds < 0:
            return "0m"
        hours, remainder = divmod(total_seconds, 3600)
        minutes, _ = divmod(remainder, 60)
        if hours > 0:
            return f"{hours}h {minutes}m"
        return f"{minutes}m"

    @staticmethod
    def _human_age(seconds: Optional[float]) -> str:
        """Convert age in seconds to a human string like '12s ago'."""
        if seconds is None:
            return "never"
        s = int(seconds)
        if s < 60:
            return f"{s}s ago"
        if s < 3600:
            return f"{s // 60}m ago"
        return f"{s // 3600}h ago"

    # ------------------------------------------------------------------
    # Message splitting
    # ------------------------------------------------------------------

    @staticmethod
    def _split_message(text: str) -> list[str]:
        """Split text into chunks that fit within Telegram's 4096 char limit."""
        if len(text) <= _MAX_MESSAGE_LENGTH:
            return [text]
        chunks: list[str] = []
        while text:
            if len(text) <= _MAX_MESSAGE_LENGTH:
                chunks.append(text)
                break
            # Try to split at last newline before limit
            cut = text.rfind("\n", 0, _MAX_MESSAGE_LENGTH)
            if cut <= 0:
                cut = _MAX_MESSAGE_LENGTH
            chunks.append(text[:cut])
            text = text[cut:].lstrip("\n")
        return chunks

    # ------------------------------------------------------------------
    # Low-level send
    # ------------------------------------------------------------------

    async def _send_message_async(self, text: str, parse_mode: Optional[str] = None) -> bool:
        """Send a message to the configured chat. Splits if too long."""
        if not self._configured:
            return False

        try:
            from telegram import Bot

            bot = Bot(token=self._token)
            for chunk in self._split_message(text):
                await bot.send_message(
                    chat_id=self._chat_id,
                    text=chunk,
                    parse_mode=parse_mode,
                )
            return True
        except ImportError:
            logger.error("python-telegram-bot not installed")
            return False
        except Exception as exc:
            logger.error("Failed to send Telegram message: %s", exc)
            return False

    def send_message(self, text: str, parse_mode: Optional[str] = None) -> bool:
        """Synchronous wrapper — safe to call from any thread."""
        if not self._configured:
            return False
        try:
            loop = asyncio.new_event_loop()
            try:
                return loop.run_until_complete(self._send_message_async(text, parse_mode))
            finally:
                loop.close()
        except Exception as exc:
            logger.error("send_message failed: %s", exc)
            return False

    # ------------------------------------------------------------------
    # Proactive alert API
    # ------------------------------------------------------------------

    def send_alert(self, message: str, priority: str = PRIORITY_NORMAL) -> bool:
        """Send an alert routed by priority level.

        CRITICAL / HIGH → immediate send (HIGH is rate-limited).
        NORMAL → queued for hourly digest.
        LOW → queued for EOD summary.
        """
        if not self._configured:
            return False

        priority = priority.upper()

        if priority == PRIORITY_CRITICAL:
            return self.send_message(message)

        if priority == PRIORITY_HIGH:
            now = time.monotonic()
            if now - self._last_high_send < self._high_cooldown_seconds:
                logger.debug("HIGH alert rate-limited, queuing as NORMAL")
                self._normal_queue.put(message)
                return True
            self._last_high_send = now
            return self.send_message(message)

        if priority == PRIORITY_NORMAL:
            self._normal_queue.put(message)
            return True

        # LOW
        self._low_queue.put(message)
        return True

    def flush_normal_digest(self) -> bool:
        """Flush the NORMAL priority queue as a single digest message."""
        messages: list[str] = []
        while not self._normal_queue.empty():
            try:
                messages.append(self._normal_queue.get_nowait())
            except queue.Empty:
                break
        if not messages:
            return False
        digest = "\U0001f4cb Hourly Digest\n\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\n" + "\n\n".join(messages)
        return self.send_message(digest)

    def flush_eod_summary(self) -> bool:
        """Flush the LOW priority queue as an end-of-day summary."""
        messages: list[str] = []
        # Drain both normal (leftovers) and low queues
        for q in (self._normal_queue, self._low_queue):
            while not q.empty():
                try:
                    messages.append(q.get_nowait())
                except queue.Empty:
                    break
        if not messages:
            return False
        summary = "\U0001f4ca EOD Summary\n\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\n" + "\n\n".join(messages)
        return self.send_message(summary)

    # ------------------------------------------------------------------
    # Command handler implementations
    # ------------------------------------------------------------------

    def _cmd_start(self) -> str:
        return (
            "\U0001f916 Trading DSS Bot Active\n"
            "\n"
            "Commands:\n"
            "/status \u2014 System health & market status\n"
            "/signal \u2014 Current signals for all indices\n"
            "/signal NIFTY50 \u2014 Signal for specific index\n"
            "/portfolio \u2014 Open positions & today's P&L\n"
            "/performance \u2014 7-day performance summary\n"
            "/news \u2014 Top 5 impactful news articles\n"
            "/alerts \u2014 Active anomaly alerts\n"
            "/vix \u2014 Current VIX & regime\n"
            "/fii \u2014 Latest FII/DII activity\n"
            "/levels NIFTY50 \u2014 Key support/resistance levels\n"
            "/params \u2014 Currently approved strategy parameters\n"
            "/health \u2014 System component health check\n"
            "/kill \u2014 EMERGENCY: Halt all signal generation\n"
            "/resume \u2014 Resume after kill switch\n"
            "/help \u2014 Show this message"
        )

    def _cmd_status(self) -> str:
        try:
            from src.utils.market_hours import MarketHoursManager
            from src.data.rate_limiter import freshness_tracker

            mh = MarketHoursManager()
            now = datetime.now(tz=__import__("zoneinfo").ZoneInfo(IST_TIMEZONE))

            # Market status
            status_info = mh.get_market_status(now)
            market_open = status_info.get("is_trading_day", False) and mh.is_market_open(now)
            if market_open:
                remaining = mh.time_to_market_close(now)
                remaining_str = self._human_timedelta(remaining) if remaining else "?"
                market_line = f"Market: \U0001f7e2 OPEN (closes in {remaining_str})"
            else:
                session = mh.get_session(now).name
                market_line = f"Market: \U0001f534 {session}"

            # VIX
            vix_row = self._db.fetch_one(
                "SELECT vix_value FROM vix_data ORDER BY timestamp DESC LIMIT 1", ()
            )
            vix_val = vix_row["vix_value"] if vix_row else 0.0
            vix_regime = "NORMAL"
            if vix_val > settings.thresholds.vix_panic_threshold:
                vix_regime = "HIGH"
            elif vix_val > settings.thresholds.vix_elevated_threshold:
                vix_regime = "ELEVATED"
            elif vix_val < settings.thresholds.vix_normal_threshold:
                vix_regime = "LOW"

            # Kill switch / mode
            kill_active = self._engine and self._engine._is_kill_switch_active() if self._engine else False
            mode = "\U0001f6a8 KILLED" if kill_active else "FULL \u2705"

            # Data freshness
            fresh = freshness_tracker.get_all_status()
            comp_lines: list[str] = []
            _labels = {
                "index_prices": ("Price Data", 300),
                "options_chain": ("Options Chain", 600),
                "vix": ("VIX Data", 480),
                "fii_dii": ("FII Data", 86400),
            }
            for dtype, (label, max_age) in _labels.items():
                info = fresh.get(dtype, {})
                age = info.get("age_seconds")
                stale = info.get("is_stale", True)
                age_str = self._human_age(age)
                if stale and age is not None and age > max_age:
                    emoji = "\u26a0\ufe0f"
                else:
                    emoji = "\u2705"
                extra = ""
                if dtype == "fii_dii" and age and age > 3600:
                    extra = " \u2014 normal, updates daily"
                comp_lines.append(f"{emoji} {label} ({age_str}{extra})")

            # Today's signals and P&L
            today_str = now.strftime("%Y-%m-%d")
            sig_row = self._db.fetch_one(
                "SELECT COUNT(*) AS cnt FROM trading_signals "
                "WHERE date(generated_at) = ? AND signal_type != 'NO_TRADE'",
                (today_str,),
            )
            sig_count = sig_row["cnt"] if sig_row else 0

            open_row = self._db.fetch_one(
                "SELECT COUNT(*) AS cnt FROM trading_signals WHERE outcome = 'OPEN'", ()
            )
            open_count = open_row["cnt"] if open_row else 0

            pnl_row = self._db.fetch_one(
                "SELECT COALESCE(SUM(actual_pnl), 0) AS total FROM trading_signals "
                "WHERE outcome IN ('WIN','LOSS') AND date(closed_at) = ?",
                (today_str,),
            )
            daily_pnl = pnl_row["total"] if pnl_row else 0.0
            cap = 100_000.0
            if self._engine:
                cap = self._engine.risk_manager.config.total_capital
            pnl_pct = (daily_pnl / cap * 100) if cap > 0 else 0.0

            components = "\n".join(comp_lines)
            return (
                f"\U0001f4ca System Status\n"
                f"\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\n"
                f"{market_line}\n"
                f"Mode: {mode}\n"
                f"VIX: {vix_val:.1f} ({vix_regime})\n"
                f"\n"
                f"Components:\n"
                f"{components}\n"
                f"\n"
                f"Today: {sig_count} signals | {open_count} open position(s)\n"
                f"Daily P&L: {self._format_currency(daily_pnl)} ({self._format_pct(pnl_pct)})"
            )
        except Exception as exc:
            logger.exception("Error in /status")
            return f"Unable to fetch status: {exc}"

    def _cmd_signal(self, index_id: Optional[str] = None) -> str:
        if not self._engine:
            return "\u23f3 System starting up, please wait..."

        try:
            if index_id:
                return self._signal_detailed(index_id.upper())
            return self._signal_compact()
        except Exception as exc:
            logger.exception("Error in /signal")
            return f"Unable to fetch signals: {exc}"

    def _signal_compact(self) -> str:
        """Compact view: latest signal per F&O index."""
        cache = getattr(self._engine, "_result_cache", {})
        if not cache:
            return "\U0001f4c8 No signals generated yet. Run a decision cycle first."

        lines: list[str] = ["\U0001f4c8 Current Signals", "\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501"]
        for idx_id, result in sorted(cache.items()):
            sig = result.signal
            sig_type = getattr(sig, "signal_type", "NO_TRADE")
            conf_level = getattr(sig, "confidence_level", "-")
            conf_score = getattr(sig, "confidence_score", 0.0) or 0.0
            emoji = self._format_signal_emoji(sig_type)

            label = sig_type.replace("_", " ") if sig_type != "NO_TRADE" else "NO TRADE"
            conf_str = f"{conf_level} ({conf_score:.2f})" if sig_type != "NO_TRADE" else "\u2014"
            lines.append(f"{idx_id:<12} {label} {emoji} {conf_str}")

        return "\n".join(lines)

    def _signal_detailed(self, index_id: str) -> str:
        """Detailed signal for a single index."""
        cache = getattr(self._engine, "_result_cache", {})
        result = cache.get(index_id)
        if not result:
            return f"No recent signal for {index_id}. Try /signal to see all."

        sig = result.signal
        sig_type = getattr(sig, "signal_type", "NO_TRADE")
        if sig_type == "NO_TRADE":
            return f"\u26aa NO TRADE \u2014 {index_id}\nNo actionable signal at this time."

        emoji = self._format_signal_emoji(sig_type)
        label = sig_type.replace("_", " ")
        conf_level = getattr(sig, "confidence_level", "-")
        conf_score = getattr(sig, "confidence_score", 0.0) or 0.0
        regime = getattr(sig, "regime", "Unknown")
        entry = getattr(sig, "refined_entry", 0.0) or getattr(sig, "entry_price", 0.0) or 0.0
        target = getattr(sig, "refined_target", 0.0) or getattr(sig, "target_price", 0.0) or 0.0
        sl = getattr(sig, "refined_stop_loss", 0.0) or getattr(sig, "stop_loss", 0.0) or 0.0
        rr = getattr(sig, "risk_reward_ratio", 0.0) or 0.0

        target_pts = abs(target - entry)
        sl_pts = abs(entry - sl)

        # Option details (if RefinedSignal)
        strike = getattr(sig, "recommended_strike", None)
        expiry = getattr(sig, "recommended_expiry", None)
        premium = getattr(sig, "option_premium", None)
        lots = getattr(sig, "lots", 0) or getattr(sig, "suggested_lot_count", 0)
        max_loss = getattr(sig, "max_loss_amount", 0.0) or getattr(sig, "estimated_max_loss", 0.0)
        risk_pct = getattr(sig, "risk_pct_of_capital", 0.0) or 0.0
        reasoning = getattr(sig, "reasoning", "") or ""
        warnings = getattr(sig, "warnings", []) or []

        option_type = "CE" if "CALL" in sig_type else "PE"
        strike_str = ""
        if strike:
            strike_str = (
                f"\n\U0001f4cb Strike: {index_id} {strike:.0f} {option_type}"
                f"{' (' + expiry + ')' if expiry else ''}"
            )
            if premium:
                strike_str += f"\nPremium: ~{self._format_currency(premium)}"
            if lots:
                strike_str += f" | Lots: {lots}"
            if max_loss:
                strike_str += f"\nRisk: {self._format_currency(max_loss)} ({risk_pct:.1f}%)"

        # Build reasoning bullets (first 4 lines)
        reason_lines = [l.strip() for l in reasoning.split("\n") if l.strip()][:4]
        reason_str = "\n".join(f"\u2022 {self._truncate(l, 80)}" for l in reason_lines) if reason_lines else "\u2022 See full analysis"

        warn_str = ""
        if warnings:
            warn_str = "\n\n\u26a0\ufe0f " + " | ".join(warnings[:3])

        return (
            f"{emoji} {label} \u2014 {index_id}\n"
            f"\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\n"
            f"Confidence: {conf_level} ({conf_score:.2f})\n"
            f"Regime: {regime}\n"
            f"\n"
            f"\U0001f4ca Levels:\n"
            f"Entry: {entry:,.0f}\n"
            f"Target: {target:,.0f} (+{target_pts:.0f} pts)\n"
            f"Stop Loss: {sl:,.0f} (-{sl_pts:.0f} pts)\n"
            f"RR: 1:{rr:.2f}"
            f"{strike_str}\n"
            f"\n"
            f"\U0001f4c8 Why:\n"
            f"{reason_str}"
            f"{warn_str}"
        )

    def _cmd_portfolio(self) -> str:
        if not self._engine:
            return "\u23f3 System starting up, please wait..."
        try:
            summary = self._engine.risk_manager.get_portfolio_summary()
            cap = summary.total_capital
            pnl = summary.today_pnl
            pnl_pct = (pnl / cap * 100) if cap > 0 else 0.0

            lines: list[str] = [
                "\U0001f4bc Portfolio",
                "\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501",
                f"Capital: {self._format_currency(cap)}",
                f"Today's P&L: {self._format_currency(pnl)} ({self._format_pct(pnl_pct)})",
            ]

            if summary.open_positions:
                lines.append("\nOpen Positions:")
                for pos in summary.open_positions:
                    sig_type = pos.get("signal_type", "?")
                    idx = pos.get("index", "?")
                    entry = pos.get("entry", 0.0)
                    sl = pos.get("current_sl", 0.0)
                    lots = pos.get("lots", 0)
                    emoji = "\U0001f4d7" if "CALL" in sig_type else "\U0001f4d5"
                    lines.append(
                        f"{emoji} {idx} {sig_type.replace('BUY_', '')} @ {entry:,.0f}\n"
                        f"   SL: {sl:,.0f} | Lots: {lots}"
                    )
            else:
                lines.append("\nNo open positions")

            lines.append(
                f"\nToday: {summary.total_trades_today} trades\n"
                f"Risk Used: {100 - summary.daily_limit_remaining_pct:.0f}% of daily limit"
            )
            return "\n".join(lines)
        except Exception as exc:
            logger.exception("Error in /portfolio")
            return f"Unable to fetch portfolio: {exc}"

    def _cmd_performance(self) -> str:
        if not self._engine:
            return "\u23f3 System starting up, please wait..."
        try:
            stats = self._engine.tracker.get_performance_stats(days=7)

            # Streak calculation
            streak_line = ""
            if stats.wins > 0 or stats.losses > 0:
                recent = self._db.fetch_all(
                    "SELECT outcome FROM trading_signals "
                    "WHERE outcome IN ('WIN','LOSS') "
                    "ORDER BY closed_at DESC LIMIT 10",
                    (),
                )
                if recent:
                    streak_type = recent[0]["outcome"]
                    streak = 0
                    for r in recent:
                        if r["outcome"] == streak_type:
                            streak += 1
                        else:
                            break
                    label = "wins" if streak_type == "WIN" else "losses"
                    emoji = "\U0001f525" if streak_type == "WIN" else "\u26a0\ufe0f"
                    streak_line = f"\nStreak: {streak} consecutive {label} {emoji}"

            high_wr_emoji = "\u2705" if stats.high_confidence_win_rate >= 60 else "\u26a0\ufe0f"
            med_wr_emoji = "\u2705" if stats.medium_confidence_win_rate >= 50 else "\u26a0\ufe0f"

            return (
                f"\U0001f4ca 7-Day Performance\n"
                f"\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\n"
                f"Trades: {stats.total_trades} | Win Rate: {stats.win_rate:.1f}%\n"
                f"P&L: {self._format_currency(stats.total_pnl)} ({self._format_pct(stats.total_pnl / 100_000 * 100 if stats.total_pnl else 0)})\n"
                f"\n"
                f"Profit Factor: {stats.profit_factor:.2f}\n"
                f"Avg Win: {self._format_currency(stats.avg_win)} | "
                f"Avg Loss: {self._format_currency(stats.avg_loss)}\n"
                f"Max Drawdown: {self._format_pct(stats.max_drawdown_pct)}\n"
                f"\n"
                f"By Confidence:\n"
                f"HIGH: {stats.high_confidence_win_rate:.0f}% WR "
                f"{high_wr_emoji}\n"
                f"MED:  {stats.medium_confidence_win_rate:.0f}% WR "
                f"{med_wr_emoji}"
                f"{streak_line}"
            )
        except Exception as exc:
            logger.exception("Error in /performance")
            return f"Unable to fetch performance: {exc}"

    def _cmd_news(self) -> str:
        if not self._engine:
            return "\u23f3 System starting up, please wait..."
        try:
            # Fetch recent news from DB
            articles = self._db.fetch_all(
                "SELECT title, adjusted_sentiment, impact_category, source "
                "FROM news_articles "
                "WHERE is_processed = 1 "
                "ORDER BY published_at DESC LIMIT 5",
                (),
            )
            if not articles:
                return "\U0001f4f0 No recent news articles."

            lines: list[str] = ["\U0001f4f0 Top Market News", "\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501"]
            for art in articles:
                title = self._truncate(art.get("title", ""), 80)
                sentiment = art.get("adjusted_sentiment", 0.0) or 0.0
                impact = (art.get("impact_category", "MEDIUM") or "MEDIUM").upper()
                emoji = self._format_severity_emoji(impact)
                bias = "Bullish" if sentiment > 0.1 else ("Bearish" if sentiment < -0.1 else "Neutral")
                lines.append(f"{emoji} [{impact}] {title}\n   \u2192 {bias}")

            # Overall sentiment
            avg_sent = sum((a.get("adjusted_sentiment", 0) or 0) for a in articles) / max(len(articles), 1)
            if avg_sent > 0.15:
                overall = "BULLISH"
            elif avg_sent > 0.05:
                overall = "SLIGHTLY BULLISH"
            elif avg_sent < -0.15:
                overall = "BEARISH"
            elif avg_sent < -0.05:
                overall = "SLIGHTLY BEARISH"
            else:
                overall = "NEUTRAL"

            lines.append(f"\nOverall Sentiment: {overall}")
            return "\n".join(lines)
        except Exception as exc:
            logger.exception("Error in /news")
            return f"Unable to fetch news: {exc}"

    def _cmd_alerts(self) -> str:
        try:
            alerts = self._db.fetch_all(
                "SELECT index_id, anomaly_type, severity, message, timestamp "
                "FROM anomaly_events "
                "WHERE is_active = 1 "
                "ORDER BY CASE severity WHEN 'HIGH' THEN 0 WHEN 'MEDIUM' THEN 1 ELSE 2 END, "
                "timestamp DESC LIMIT 10",
                (),
            )
            if not alerts:
                return "\u2705 No active alerts."

            lines: list[str] = [
                f"\u26a0\ufe0f Active Alerts ({len(alerts)})",
                "\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501",
            ]
            now = datetime.now(tz=__import__("zoneinfo").ZoneInfo(IST_TIMEZONE))
            for a in alerts:
                sev = (a.get("severity", "MEDIUM") or "MEDIUM").upper()
                emoji = self._format_severity_emoji(sev)
                idx = a.get("index_id", "MARKET")
                desc = self._truncate(a.get("message", "") or a.get("anomaly_type", ""), 100)
                ts = a.get("timestamp", "")
                try:
                    alert_time = datetime.fromisoformat(str(ts))
                    if alert_time.tzinfo is None:
                        from zoneinfo import ZoneInfo
                        alert_time = alert_time.replace(tzinfo=ZoneInfo(IST_TIMEZONE))
                    age = now - alert_time
                    age_str = f"Detected {self._human_timedelta(age)} ago"
                except (ValueError, TypeError):
                    age_str = ""

                lines.append(f"{emoji} {sev} | {idx}\n   {desc}\n   {age_str}")
            return "\n".join(lines)
        except Exception as exc:
            logger.exception("Error in /alerts")
            return f"Unable to fetch alerts: {exc}"

    def _cmd_vix(self) -> str:
        try:
            row = self._db.fetch_one(
                "SELECT vix_value, vix_change, vix_change_pct, timestamp "
                "FROM vix_data ORDER BY timestamp DESC LIMIT 1",
                (),
            )
            if not row:
                return "VIX data not available."

            val = row["vix_value"]
            chg = row.get("vix_change", 0.0) or 0.0
            chg_pct = row.get("vix_change_pct", 0.0) or 0.0

            # Determine regime
            if val > settings.thresholds.vix_panic_threshold:
                regime, impl, options, sizing = "HIGH", "Extreme caution", "Expensive", "Reduce by 50%"
            elif val > settings.thresholds.vix_elevated_threshold:
                regime, impl, options, sizing = "ELEVATED", "Increased volatility", "Above average", "Reduce by 25%"
            elif val < settings.thresholds.vix_normal_threshold:
                regime, impl, options, sizing = "LOW", "Complacency risk", "Cheap", "Standard sizing"
            else:
                regime, impl, options, sizing = "NORMAL", "Standard conditions", "Normally priced", "No adjustment needed"

            return (
                f"\U0001f4c9 India VIX: {val:.2f} ({chg:+.2f}, {chg_pct:+.1f}%)\n"
                f"Regime: {regime}\n"
                f"\n"
                f"Implication: {impl}\n"
                f"Options: {options}\n"
                f"Position sizing: {sizing}"
            )
        except Exception as exc:
            logger.exception("Error in /vix")
            return f"Unable to fetch VIX: {exc}"

    def _cmd_fii(self) -> str:
        try:
            row = self._db.fetch_one(
                "SELECT * FROM fii_dii_activity ORDER BY date DESC LIMIT 1", ()
            )
            if not row:
                return "FII/DII data not available."

            trade_date = row.get("date", "")

            # Cash segment
            fii_net = 0.0
            dii_net = 0.0
            fii_fo_net = 0.0

            # The table might store FII and DII as separate rows or combined
            # Fetch all rows for the latest date
            all_rows = self._db.fetch_all(
                "SELECT * FROM fii_dii_activity WHERE date = ? ORDER BY category",
                (trade_date,),
            )
            for r in all_rows:
                cat = (r.get("category", "") or "").upper()
                net = r.get("net_value", 0.0) or 0.0
                segment = (r.get("segment", "cash") or "cash").lower()
                if "FII" in cat or "FPI" in cat:
                    if "f&o" in segment or "fo" in segment:
                        fii_fo_net = net
                    else:
                        fii_net = net
                elif "DII" in cat:
                    dii_net = net

            # FII trend from recent data
            recent_rows = self._db.fetch_all(
                "SELECT net_value FROM fii_dii_activity "
                "WHERE (category LIKE '%FII%' OR category LIKE '%FPI%') "
                "AND (segment IS NULL OR segment = 'cash' OR segment = '') "
                "ORDER BY date DESC LIMIT 5",
                (),
            )
            fii_5d_total = sum(r.get("net_value", 0) or 0 for r in recent_rows)
            bias = "BUYING" if fii_5d_total > 0 else ("SELLING" if fii_5d_total < 0 else "NEUTRAL")
            today_bias = "BULLISH" if fii_net > 0 else ("BEARISH" if fii_net < 0 else "NEUTRAL")

            def _cr(val: float) -> str:
                """Format as Crore string."""
                sign = "Bought" if val >= 0 else "Sold"
                return f"{sign} {self._format_currency(abs(val))} Cr"

            return (
                f"\U0001f3e6 FII/DII Activity ({trade_date})\n"
                f"\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\n"
                f"FII: {_cr(fii_net)} (Cash)"
                f"{f' | {_cr(fii_fo_net)} (F&O)' if fii_fo_net else ''}\n"
                f"DII: {_cr(dii_net)} (Cash)\n"
                f"\n"
                f"FII 5-day trend: {bias} ({self._format_currency(fii_5d_total)} Cr)\n"
                f"Bias: {today_bias} for today"
            )
        except Exception as exc:
            logger.exception("Error in /fii")
            return f"Unable to fetch FII/DII data: {exc}"

    def _cmd_levels(self, index_id: Optional[str] = None) -> str:
        if not index_id:
            return "Usage: /levels NIFTY50"
        index_id = index_id.upper()

        try:
            # Get latest technical indicators
            row = self._db.fetch_one(
                "SELECT * FROM technical_indicators "
                "WHERE index_id = ? ORDER BY timestamp DESC LIMIT 1",
                (index_id,),
            )
            if not row:
                return f"No technical data for {index_id}."

            # Current price
            price_row = self._db.fetch_one(
                "SELECT close FROM price_data WHERE index_id = ? ORDER BY timestamp DESC LIMIT 1",
                (index_id,),
            )
            cmp = price_row["close"] if price_row else 0.0

            support = row.get("support_1", 0.0) or 0.0
            resistance = row.get("resistance_1", 0.0) or 0.0
            ema20 = row.get("ema_20", 0.0) or 0.0
            ema50 = row.get("ema_50", 0.0) or 0.0
            signal_str = row.get("technical_signal", "NEUTRAL") or "NEUTRAL"

            # OI-based levels from options chain
            oi_row = self._db.fetch_one(
                "SELECT max_pain_strike, pcr, highest_ce_oi_strike, highest_pe_oi_strike "
                "FROM oi_aggregated WHERE index_id = ? ORDER BY timestamp DESC LIMIT 1",
                (index_id,),
            )
            max_pain = oi_row.get("max_pain_strike", 0.0) if oi_row else 0.0
            pcr = oi_row.get("pcr", 0.0) if oi_row else 0.0
            ce_wall = oi_row.get("highest_ce_oi_strike", 0.0) if oi_row else 0.0
            pe_support = oi_row.get("highest_pe_oi_strike", 0.0) if oi_row else 0.0

            # Build resistance/support arrays
            r_levels: list[tuple[float, str]] = []
            s_levels: list[tuple[float, str]] = []

            if resistance:
                r_levels.append((resistance, "Technical"))
            if ce_wall and ce_wall > cmp:
                r_levels.append((ce_wall, "OI CE wall"))
            if ema20 and ema20 > cmp:
                r_levels.append((ema20, "EMA20"))

            if support:
                s_levels.append((support, "Technical"))
            if pe_support and pe_support < cmp:
                s_levels.append((pe_support, "OI PE support"))
            if ema50 and ema50 < cmp:
                s_levels.append((ema50, "EMA50"))

            r_levels.sort(key=lambda x: x[0])
            s_levels.sort(key=lambda x: x[0], reverse=True)

            r_lines = ""
            for i, (val, label) in enumerate(r_levels[:3], 1):
                r_lines += f"R{i}: {val:,.0f} ({label})\n"
            if not r_lines:
                r_lines = "No resistance levels computed\n"

            s_lines = ""
            for i, (val, label) in enumerate(s_levels[:3], 1):
                s_lines += f"S{i}: {val:,.0f} ({label})\n"
            if not s_lines:
                s_lines = "No support levels computed\n"

            pcr_bias = "Bullish" if pcr > 1.0 else ("Bearish" if pcr < 0.7 else "Neutral")

            return (
                f"\U0001f4d0 Key Levels \u2014 {index_id}\n"
                f"\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\n"
                f"CMP: {cmp:,.0f}\n"
                f"\n"
                f"Resistance:\n{r_lines}"
                f"\n"
                f"Support:\n{s_lines}"
                f"\n"
                f"Max Pain: {max_pain:,.0f}\n"
                f"PCR: {pcr:.2f} ({pcr_bias})"
            )
        except Exception as exc:
            logger.exception("Error in /levels")
            return f"Unable to fetch levels: {exc}"

    def _cmd_params(self) -> str:
        try:
            from src.backtest.optimizer.param_applier import ApprovedParameterManager
            from src.utils.date_utils import get_ist_now

            mgr = ApprovedParameterManager()
            all_params = mgr.list_all_approved()
            if not all_params:
                return "\u2699\ufe0f No approved parameters found."

            now = get_ist_now()
            lines: list[str] = ["\u2699\ufe0f Strategy Parameters", "\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501"]

            for idx_id, entry in sorted(all_params.items()):
                status = entry.get("status", "UNKNOWN")
                params = entry.get("params", {})
                approved_at = entry.get("approved_at", "")

                age_str = ""
                try:
                    approved_dt = datetime.fromisoformat(approved_at)
                    age_days = (now - approved_dt).days
                    age_str = f"{age_days} days ago"
                except (ValueError, TypeError):
                    age_str = "unknown"

                status_emoji = "\u2705" if status == "ACTIVE" else "\u274c"

                lines.append(f"{idx_id}:")

                # Show key params
                sl_str = params.get("sl_multiplier", "default")
                tgt_str = params.get("target_multiplier", "default")
                min_conf = params.get("min_confidence", "default")
                lines.append(f"  SL: {sl_str} ATR | Target: {tgt_str} ATR")
                if min_conf != "default":
                    lines.append(f"  Min Confidence: {min_conf}")
                lines.append(f"  Approved: {approved_at[:10]} ({age_str}) {status_emoji}")

            return "\n".join(lines)
        except Exception as exc:
            logger.exception("Error in /params")
            return f"Unable to fetch parameters: {exc}"

    def _cmd_health(self) -> str:
        try:
            from src.utils.date_utils import get_ist_now

            now = get_ist_now()
            kill_active = False
            if self._engine:
                kill_active = self._engine._is_kill_switch_active()

            mode = "\U0001f6a8 KILLED" if kill_active else "FULL \u2705"

            uptime_str = "N/A"
            if self._start_time:
                uptime = now - self._start_time
                uptime_str = self._human_timedelta(uptime)

            db_size = self._db.get_db_size()

            # Component health from DB
            components = self._db.fetch_all(
                "SELECT component, status, message, response_time_ms, timestamp "
                "FROM system_health "
                "WHERE component != 'kill_switch' "
                "GROUP BY component "
                "HAVING timestamp = MAX(timestamp) "
                "ORDER BY component",
                (),
            )
            comp_lines: list[str] = []
            for c in components:
                comp_name = c.get("component", "unknown")
                status = (c.get("status", "UNKNOWN") or "UNKNOWN").upper()
                msg = c.get("message", "")
                rt = c.get("response_time_ms")
                if status == "OK":
                    emoji = "\u2705"
                elif status == "WARNING":
                    emoji = "\u26a0\ufe0f"
                else:  # ERROR
                    emoji = "\u274c"
                rt_str = f" (avg {rt}ms)" if rt else ""
                extra = f" \u2014 {msg}" if msg else ""
                comp_lines.append(f"{emoji} {comp_name}: {status}{rt_str}{extra}")

            if not comp_lines:
                comp_lines.append("No health data recorded yet")

            # Today's stats
            today_str = now.strftime("%Y-%m-%d")
            sig_row = self._db.fetch_one(
                "SELECT COUNT(*) AS total, "
                "SUM(CASE WHEN signal_type != 'NO_TRADE' THEN 1 ELSE 0 END) AS actionable "
                "FROM trading_signals WHERE date(generated_at) = ?",
                (today_str,),
            )
            total_sigs = sig_row["total"] if sig_row else 0
            actionable = sig_row["actionable"] if sig_row else 0

            alert_row = self._db.fetch_one(
                "SELECT COUNT(*) AS cnt FROM anomaly_events WHERE date(timestamp) = ?",
                (today_str,),
            )
            alert_count = alert_row["cnt"] if alert_row else 0

            return (
                f"\U0001f527 System Health\n"
                f"\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\n"
                f"Mode: {mode}\n"
                f"Uptime: {uptime_str}\n"
                f"DB Size: {db_size}\n"
                f"\n"
                + "\n".join(comp_lines)
                + f"\n\nToday's Stats:\n"
                f"Signals generated: {total_sigs} | Actionable: {actionable}\n"
                f"Alerts fired: {alert_count}"
            )
        except Exception as exc:
            logger.exception("Error in /health")
            return f"Unable to fetch health: {exc}"

    def _cmd_kill(self) -> str:
        if not self._engine:
            return "\u23f3 System starting up, please wait..."
        try:
            self._engine.activate_kill_switch(reason="Manual kill via Telegram bot")
            return (
                "\U0001f6a8 KILL SWITCH ACTIVATED\n"
                "All signal generation halted immediately.\n"
                "Open positions are NOT auto-closed.\n"
                "\n"
                "To resume: /resume\n"
                "To close all positions: (manual action required)"
            )
        except Exception as exc:
            logger.exception("Error activating kill switch")
            return f"Failed to activate kill switch: {exc}"

    def _cmd_resume(self) -> str:
        if not self._engine:
            return "\u23f3 System starting up, please wait..."
        try:
            self._engine.deactivate_kill_switch()
            return (
                "\u2705 System resumed. Signal generation active.\n"
                "Next cycle in ~60 seconds."
            )
        except Exception as exc:
            logger.exception("Error deactivating kill switch")
            return f"Failed to resume: {exc}"

    # ------------------------------------------------------------------
    # Command routing (used by telegram handler)
    # ------------------------------------------------------------------

    _COMMANDS = {
        "start", "help", "status", "signal", "portfolio", "performance",
        "news", "alerts", "vix", "fii", "levels", "params", "health",
        "kill", "resume",
    }

    def handle_command(self, command: str, args: str = "") -> str:
        """Route a command string to the appropriate handler.

        Returns the response text. Does NOT send it — the caller is
        responsible for delivery.
        """
        cmd = command.lower().strip().lstrip("/")
        args = args.strip()

        if cmd in ("start", "help"):
            return self._cmd_start()
        if cmd == "status":
            return self._cmd_status()
        if cmd == "signal":
            return self._cmd_signal(args if args else None)
        if cmd == "portfolio":
            return self._cmd_portfolio()
        if cmd == "performance":
            return self._cmd_performance()
        if cmd == "news":
            return self._cmd_news()
        if cmd == "alerts":
            return self._cmd_alerts()
        if cmd == "vix":
            return self._cmd_vix()
        if cmd == "fii":
            return self._cmd_fii()
        if cmd == "levels":
            return self._cmd_levels(args if args else None)
        if cmd == "params":
            return self._cmd_params()
        if cmd == "health":
            return self._cmd_health()
        if cmd == "kill":
            return self._cmd_kill()
        if cmd == "resume":
            return self._cmd_resume()

        # Unknown command — suggest closest match
        import difflib
        close = difflib.get_close_matches(cmd, self._COMMANDS, n=1, cutoff=0.5)
        suggestion = f" Did you mean /{close[0]}?" if close else ""
        return f"Unknown command: /{cmd}.{suggestion}\nType /help for available commands."

    # ------------------------------------------------------------------
    # python-telegram-bot integration
    # ------------------------------------------------------------------

    def _build_app(self) -> Any:
        """Build the python-telegram-bot Application with command handlers."""
        try:
            from telegram.ext import Application, CommandHandler, MessageHandler, filters
        except ImportError:
            logger.error("python-telegram-bot not installed — cannot build app")
            return None

        app = Application.builder().token(self._token).build()

        async def _handler(update, context, cmd_name: str) -> None:
            """Generic async handler that delegates to handle_command."""
            args_text = " ".join(context.args) if context.args else ""
            try:
                response = self.handle_command(cmd_name, args_text)
            except Exception as exc:
                logger.exception("Unhandled error in /%s", cmd_name)
                response = f"Internal error processing /{cmd_name}. Please try again."
            for chunk in self._split_message(response):
                await update.message.reply_text(chunk)

        # Register every known command
        for cmd_name in self._COMMANDS:
            # Create a closure that captures cmd_name correctly
            def _make_handler(name: str) -> Callable:
                async def _h(update, context) -> None:
                    await _handler(update, context, name)
                return _h
            app.add_handler(CommandHandler(cmd_name, _make_handler(cmd_name)))

        # Fallback for unknown commands
        async def _unknown(update, context) -> None:
            text = update.message.text or ""
            if text.startswith("/"):
                parts = text.split(None, 1)
                cmd = parts[0].lstrip("/").split("@")[0]
                args = parts[1] if len(parts) > 1 else ""
                response = self.handle_command(cmd, args)
                await update.message.reply_text(response)

        app.add_handler(MessageHandler(filters.COMMAND, _unknown))

        return app

    # ------------------------------------------------------------------
    # Lifecycle management
    # ------------------------------------------------------------------

    def start_bot(self) -> None:
        """Start polling for commands in a background thread.

        Non-blocking — safe to call from the main DataCollector process.
        """
        if not self._configured:
            logger.info("Telegram not configured — bot will not start")
            return

        if self._running:
            logger.warning("Telegram bot is already running")
            return

        app = self._build_app()
        if app is None:
            return

        self._app = app
        self._running = True
        self._start_time = datetime.now(
            tz=__import__("zoneinfo").ZoneInfo(IST_TIMEZONE)
        )

        def _run_polling() -> None:
            try:
                # Create a new event loop for this thread
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(self._app.initialize())
                loop.run_until_complete(self._app.start())
                loop.run_until_complete(
                    self._app.updater.start_polling(drop_pending_updates=True)
                )
                logger.info("Telegram bot polling started")
                loop.run_forever()
            except Exception:
                logger.exception("Telegram bot polling failed")
            finally:
                self._running = False

        self._thread = threading.Thread(
            target=_run_polling, name="telegram-bot", daemon=True,
        )
        self._thread.start()
        logger.info("Telegram bot thread started")

    def stop_bot(self) -> None:
        """Gracefully shut down the bot."""
        if not self._running or not self._app:
            return

        self._running = False
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self._app.updater.stop())
            loop.run_until_complete(self._app.stop())
            loop.run_until_complete(self._app.shutdown())
            loop.close()
        except Exception:
            logger.exception("Error during Telegram bot shutdown")

        logger.info("Telegram bot stopped")

    def is_running(self) -> bool:
        """Return whether the bot is currently polling for commands."""
        return self._running

    # ------------------------------------------------------------------
    # Signal formatting for proactive alerts
    # ------------------------------------------------------------------

    def format_signal_alert(self, result: Any) -> str:
        """Format a DecisionResult as a concise proactive alert message."""
        sig = result.signal
        sig_type = getattr(sig, "signal_type", "NO_TRADE")
        index_id = result.index_id
        conf_level = getattr(sig, "confidence_level", "-")
        conf_score = getattr(sig, "confidence_score", 0.0) or 0.0
        emoji = self._format_signal_emoji(sig_type)

        entry = getattr(sig, "refined_entry", 0.0) or getattr(sig, "entry_price", 0.0) or 0.0
        target = getattr(sig, "refined_target", 0.0) or getattr(sig, "target_price", 0.0) or 0.0
        sl = getattr(sig, "refined_stop_loss", 0.0) or getattr(sig, "stop_loss", 0.0) or 0.0
        rr = getattr(sig, "risk_reward_ratio", 0.0) or 0.0

        label = sig_type.replace("_", " ")
        return (
            f"{emoji} NEW SIGNAL: {label} \u2014 {index_id}\n"
            f"Confidence: {conf_level} ({conf_score:.2f})\n"
            f"Entry: {entry:,.0f} | Target: {target:,.0f} | SL: {sl:,.0f}\n"
            f"RR: 1:{rr:.2f}\n"
            f"Use /signal {index_id} for full details"
        )

    def format_exit_alert(self, index_id: str, reason: str, pnl: float) -> str:
        """Format a position exit alert."""
        emoji = "\U0001f4d7" if pnl >= 0 else "\U0001f4d5"
        result = "WIN" if pnl >= 0 else "LOSS"
        return (
            f"{emoji} POSITION EXIT \u2014 {index_id}\n"
            f"Reason: {reason}\n"
            f"Result: {result} | P&L: {self._format_currency(pnl)}"
        )
