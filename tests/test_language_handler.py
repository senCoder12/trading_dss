"""
Tests for language detection, translation, and Hinglish sentiment processing.

Network-dependent operations (Google Translate) are mocked so tests run
offline and deterministically.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.analysis.news.hinglish_processor import HinglishProcessor
from src.analysis.news.language_handler import LanguageHandler, ProcessedText


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def handler_no_translators():
    """LanguageHandler with both translators disabled (no network, no argos)."""
    with (
        patch(
            "src.analysis.news.language_handler.LanguageHandler._init_primary_translator"
        ),
        patch(
            "src.analysis.news.language_handler.LanguageHandler._init_fallback_translator"
        ),
    ):
        h = LanguageHandler()
        h._primary_available = False
        h._fallback_available = False
        return h


@pytest.fixture()
def handler_with_mock_google():
    """LanguageHandler with Google Translate mocked to return predictable text."""
    with (
        patch(
            "src.analysis.news.language_handler.LanguageHandler._init_primary_translator"
        ),
        patch(
            "src.analysis.news.language_handler.LanguageHandler._init_fallback_translator"
        ),
    ):
        h = LanguageHandler()
        h._primary_available = True
        h._fallback_available = False
        return h


# ---------------------------------------------------------------------------
# 1. Pure English headline — zero overhead path
# ---------------------------------------------------------------------------


def test_english_article_no_translation(handler_no_translators):
    text = "Nifty 50 surges 300 points on strong global cues, IT stocks lead rally"
    result = handler_no_translators.process(text)

    assert result.original_lang == "en"
    assert result.was_translated is False
    assert result.translation_confidence == 1.0
    assert result.is_hinglish is False
    assert result.english_text == text  # returned as-is


# ---------------------------------------------------------------------------
# 2. Pure Hindi (Devanagari) — script detection + translation
# ---------------------------------------------------------------------------


def test_hindi_devanagari_detected(handler_with_mock_google):
    hindi_text = "निफ्टी ने आज 500 अंकों की छलांग लगाई"

    translated = "Nifty made a leap of 500 points today"
    mock_translator = MagicMock()
    mock_translator.return_value.translate.return_value = translated

    with patch("src.analysis.news.language_handler.GoogleTranslator", mock_translator):
        result = handler_with_mock_google.process(hindi_text)

    assert result.original_lang == "hi"
    assert result.was_translated is True
    assert result.english_text == translated
    assert result.is_hinglish is False


# ---------------------------------------------------------------------------
# 3. Hinglish (Latin script with Hindi words) — detected and translated
# ---------------------------------------------------------------------------


def test_hinglish_detected_and_translated(handler_with_mock_google):
    hinglish_text = "Nifty ne aaj 500 points ki chhalaang lagayi, bulls ka raj"

    translated = "Nifty made a 500-point jump today, bulls dominate"
    mock_translator = MagicMock()
    mock_translator.return_value.translate.return_value = translated

    with patch("src.analysis.news.language_handler.GoogleTranslator", mock_translator):
        result = handler_with_mock_google.process(hinglish_text)

    assert result.is_hinglish is True
    assert result.original_lang == "hinglish"
    assert result.was_translated is True
    assert result.english_text == translated


# ---------------------------------------------------------------------------
# 4. Bearish Hindi — detected, translated, and sentiment should be BEARISH
# ---------------------------------------------------------------------------


def test_bearish_hindi_detected(handler_with_mock_google):
    bearish_hindi = "बाजार में भारी गिरावट, निवेशकों में घबराहट"
    translated = "Heavy fall in the market, panic among investors"

    mock_translator = MagicMock()
    mock_translator.return_value.translate.return_value = translated

    with patch("src.analysis.news.language_handler.GoogleTranslator", mock_translator):
        result = handler_with_mock_google.process(bearish_hindi)

    assert result.original_lang == "hi"
    assert result.was_translated is True
    assert "fall" in result.english_text.lower() or "panic" in result.english_text.lower()


# ---------------------------------------------------------------------------
# 5. Each article in a batch gets independent language detection
# ---------------------------------------------------------------------------


def test_batch_independent_detection(handler_with_mock_google):
    articles = [
        "Nifty rallies 200 points on FII buying",                         # English
        "निफ्टी आज 100 अंक गिरा",                                          # Hindi
        "Sensex mein aaj bhari giravat, investors pareshan",              # Hinglish
    ]

    english_translations = [
        None,
        "Nifty fell 100 points today",
        "Heavy fall in Sensex today, investors worried",
    ]

    def make_translator(idx):
        m = MagicMock()
        m.return_value.translate.return_value = english_translations[idx]
        return m

    results = []
    for i, text in enumerate(articles):
        if english_translations[i] is not None:
            with patch(
                "src.analysis.news.language_handler.GoogleTranslator",
                make_translator(i),
            ):
                results.append(handler_with_mock_google.process(text))
        else:
            results.append(handler_with_mock_google.process(text))

    assert results[0].original_lang == "en"
    assert results[0].was_translated is False
    assert results[1].original_lang == "hi"
    assert results[2].is_hinglish is True


# ---------------------------------------------------------------------------
# 6. Translation failure → return original, no crash, is_translated=False
# ---------------------------------------------------------------------------


def test_translation_failure_graceful(handler_no_translators):
    hindi_text = "निफ्टी आज बहुत गिरा, निवेशकों में चिंता"

    result = handler_no_translators.process(hindi_text)

    # Both translators unavailable — should return original, not crash
    assert result.was_translated is False
    assert result.english_text == hindi_text
    assert result.translation_confidence == 0.0


# ---------------------------------------------------------------------------
# 7. Cache — second call does not make a new API call
# ---------------------------------------------------------------------------


def test_translation_cache_hit(handler_with_mock_google):
    text = "निफ्टी ने आज रिकॉर्ड बनाया"
    translated = "Nifty set a record today"

    call_count = 0

    def fake_translate(self_inner):
        nonlocal call_count
        call_count += 1
        return translated

    mock_cls = MagicMock()
    mock_cls.return_value.translate.side_effect = lambda t: (
        setattr(mock_cls, "_call_count", getattr(mock_cls, "_call_count", 0) + 1)
        or translated
    )

    with patch("src.analysis.news.language_handler.GoogleTranslator", mock_cls):
        r1 = handler_with_mock_google.translate_to_english(text, "hi")
        r2 = handler_with_mock_google.translate_to_english(text, "hi")

    assert r1.is_translated is True
    assert r2.method == "cache"
    assert r2.is_translated is True
    # Translator constructor should have been called only once
    assert mock_cls.call_count == 1
    assert handler_with_mock_google.stats["cache_hits"] == 1


# ---------------------------------------------------------------------------
# 8. Hinglish "mandi" keyword → bearish boost applied
# ---------------------------------------------------------------------------


def test_hinglish_mandi_bearish_boost():
    processor = HinglishProcessor()

    base_score = 0.0  # NEUTRAL from VADER
    text_with_mandi = "Market mein bhari mandi, investors pareshan"
    adjusted = processor.boost_hinglish_sentiment(text_with_mandi, base_score)

    assert adjusted < base_score, "mandi should produce a bearish (negative) boost"
    assert adjusted < -0.05


def test_hinglish_tezi_bullish_boost():
    processor = HinglishProcessor()

    base_score = 0.0
    text_with_tezi = "Nifty mein zabardast tezi, bulls ka raj"
    adjusted = processor.boost_hinglish_sentiment(text_with_tezi, base_score)

    assert adjusted > base_score, "tezi + zabardast should produce bullish boost"
    assert adjusted > 0.05


def test_hinglish_no_keywords_no_change():
    processor = HinglishProcessor()
    score = processor.boost_hinglish_sentiment("completely neutral text here", 0.2)
    assert score == 0.2


# ---------------------------------------------------------------------------
# 9. Stats tracking — hindi_pct increases correctly
# ---------------------------------------------------------------------------


def test_stats_tracking(handler_with_mock_google):
    handler = handler_with_mock_google
    # Reset stats to clean state
    handler.stats = {k: 0 for k in handler.stats}

    english_texts = [
        "Nifty rises 100 points",
        "Sensex gains on global cues",
        "RBI keeps repo rate unchanged",
    ]
    hindi_texts = [
        "निफ्टी आज बड़ी छलांग लगाई",
        "बाजार में तेजी का माहौल",
    ]

    translated = "Market translation"
    mock_cls = MagicMock()
    mock_cls.return_value.translate.return_value = translated

    for text in english_texts:
        handler.process(text)

    with patch("src.analysis.news.language_handler.GoogleTranslator", mock_cls):
        for text in hindi_texts:
            handler.process(text)

    stats = handler.get_stats()
    assert handler.stats["total_articles"] == 5
    assert handler.stats["english_articles"] == 3
    assert handler.stats["hindi_articles"] == 2
    assert stats["hindi_pct"] == "40.0%"


# ---------------------------------------------------------------------------
# 10. Script detection — Devanagari vs Latin
# ---------------------------------------------------------------------------


def test_devanagari_script_detection(handler_no_translators):
    devanagari = "निफ्टी में आज बड़ा उछाल"
    result = handler_no_translators._detect_script(devanagari)
    assert result == "devanagari"


def test_latin_script_detection(handler_no_translators):
    latin = "Nifty 50 surges on global cues"
    result = handler_no_translators._detect_script(latin)
    assert result == "latin"


# ---------------------------------------------------------------------------
# 11. Timeout — translate_with_timeout returns untranslated on slow API
# ---------------------------------------------------------------------------


def test_translate_with_timeout(handler_with_mock_google):
    import concurrent.futures

    def slow_translate(text, lang):
        import time
        time.sleep(10)
        return MagicMock(is_translated=True)

    handler = handler_with_mock_google
    with patch.object(handler, "translate_to_english", side_effect=slow_translate):
        result = handler.translate_with_timeout("some hindi text", "hi", timeout=0.1)

    assert result.is_translated is False
    assert result.method == "timeout"
    assert result.english_text == "some hindi text"
