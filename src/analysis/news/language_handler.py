"""
Language detection and translation layer for Indian financial news.

Detects non-English content (Hindi, Marathi, Hinglish, etc.) and translates
to English before sentiment analysis.  English articles pass through unchanged.
"""

from __future__ import annotations

import concurrent.futures
import hashlib
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class LanguageDetectionResult:
    detected_lang: str   # ISO 639-1 code: "en", "hi", "hinglish", "mr", etc.
    confidence: float    # 0–1
    is_hinglish: bool
    script: str          # "latin" / "devanagari"


@dataclass
class TranslationResult:
    original_text: str
    translated_text: str
    source_lang: str
    is_translated: bool
    method: str          # "google_translate" / "argostranslate_offline" / "cache" / "failed" / "timeout"
    confidence: float


@dataclass
class ProcessedText:
    english_text: str          # Always English (translated or original)
    original_lang: str         # What it was originally
    was_translated: bool
    translation_confidence: float  # 0–1
    is_hinglish: bool


class LanguageHandler:
    """Detects language of text and translates non-English content to English.

    Strategy:
    1. Detect language using langdetect
    2. If English → return as-is (no cost, no latency)
    3. If Hindi/Hinglish → translate using deep-translator (Google backend, free)
    4. If translation fails → fall back to argostranslate (offline)
    5. If all fails → return original text with is_translated=False flag

    The existing sentiment pipeline runs on the translated text unchanged.
    """

    SUPPORTED_LANGUAGES = {
        "en": "English",
        "hi": "Hindi",
        "mr": "Marathi",
        "gu": "Gujarati",
        "ta": "Tamil",
        "te": "Telugu",
        "bn": "Bengali",
    }

    def __init__(self) -> None:
        self._primary_available = False
        self._primary_translator = None
        self._fallback_available = False
        self._init_primary_translator()
        self._init_fallback_translator()

        # Cache translations to avoid re-translating same content
        # Key: hash of original text, Value: translated text
        self._translation_cache: dict[str, str] = {}
        self._cache_max_size = 500

        self.stats = {
            "total_articles": 0,
            "english_articles": 0,
            "hindi_articles": 0,
            "hinglish_articles": 0,
            "other_language_articles": 0,
            "translation_successes": 0,
            "translation_failures": 0,
            "cache_hits": 0,
        }

    def _init_primary_translator(self) -> None:
        """Initialize deep-translator (uses Google Translate, no API key)."""
        try:
            from deep_translator import GoogleTranslator

            test = GoogleTranslator(source="hi", target="en").translate("नमस्ते")
            if test:
                self._primary_translator = GoogleTranslator
                self._primary_available = True
                logger.info("Primary translator (deep-translator/Google) initialized")
        except Exception as exc:
            logger.warning("Primary translator unavailable: %s", exc)

    def _init_fallback_translator(self) -> None:
        """Initialize argostranslate (offline fallback)."""
        try:
            import argostranslate.translate

            installed = argostranslate.translate.get_installed_languages()
            lang_codes = [lang.code for lang in installed]
            if "hi" in lang_codes and "en" in lang_codes:
                self._fallback_available = True
                logger.info("Fallback translator (argostranslate) initialized")
            else:
                self._install_argos_language_pack("hi", "en")
        except ImportError:
            logger.info("argostranslate not installed — offline translation unavailable")
        except Exception as exc:
            logger.warning("Fallback translator unavailable: %s", exc)

    def _install_argos_language_pack(self, source: str, target: str) -> None:
        """Download and install argostranslate language pack."""
        try:
            import argostranslate.package

            argostranslate.package.update_package_index()
            available = argostranslate.package.get_available_packages()
            package = next(
                (p for p in available if p.from_code == source and p.to_code == target),
                None,
            )
            if package:
                argostranslate.package.install_from_path(package.download())
                self._fallback_available = True
                logger.info(
                    "Installed argostranslate %s→%s language pack", source, target
                )
            else:
                self._fallback_available = False
        except Exception as exc:
            self._fallback_available = False
            logger.warning("Failed to install argostranslate pack: %s", exc)

    # ── Language detection ────────────────────────────────────────────────────

    def detect_language(self, text: str) -> LanguageDetectionResult:
        """Detect the language of the given text.

        Returns structured result with confidence score and whether the text
        is Hinglish (mixed Hindi-English in Latin script).
        """
        if not text or len(text.strip()) < 10:
            return LanguageDetectionResult(
                detected_lang="en", confidence=0.5, is_hinglish=False, script="latin"
            )

        script = self._detect_script(text)
        is_hinglish = self._is_hinglish(text, script)

        if is_hinglish:
            return LanguageDetectionResult(
                detected_lang="hinglish",
                confidence=0.85,
                is_hinglish=True,
                script="latin",
            )

        # Devanagari script → Hindi or Marathi (both need translation)
        if script == "devanagari":
            try:
                from langdetect import detect_langs

                langs = detect_langs(text)
                top = langs[0]
                return LanguageDetectionResult(
                    detected_lang=top.lang,
                    confidence=top.prob,
                    is_hinglish=False,
                    script="devanagari",
                )
            except Exception:
                # Devanagari present but detection failed → assume Hindi
                return LanguageDetectionResult(
                    detected_lang="hi",
                    confidence=0.80,
                    is_hinglish=False,
                    script="devanagari",
                )

        # Latin script — use langdetect
        try:
            from langdetect import detect_langs

            langs = detect_langs(text)
            top = langs[0]
            return LanguageDetectionResult(
                detected_lang=top.lang,
                confidence=top.prob,
                is_hinglish=False,
                script="latin",
            )
        except Exception:
            return LanguageDetectionResult(
                detected_lang="en", confidence=0.5, is_hinglish=False, script="latin"
            )

    def _detect_script(self, text: str) -> str:
        """Return 'devanagari' if >15% of non-whitespace chars are Devanagari."""
        devanagari_chars = sum(1 for c in text if "\u0900" <= c <= "\u097F")
        total_chars = len([c for c in text if not c.isspace()])
        if total_chars > 0 and devanagari_chars / total_chars > 0.15:
            return "devanagari"
        return "latin"

    def _is_hinglish(self, text: str, script: str) -> bool:
        """Detect Hinglish: Hindi words written in Latin script mixed with English."""
        if script == "devanagari":
            return False

        # Common Hindi words transliterated to Latin script
        # Excludes words that are too ambiguous in English (e.g. 'par', 'ho')
        hindi_roman_indicators = [
            "aaj", "kal", "mein", "hai", "hain", "hua", "ki", "ka", "ke",
            "se", "ne", "ko", "par", "ho", "tha", "bhi", "kya",
            "yeh", "woh", "lekin", "aur", "nahi", "bahut", "accha",
            "bazaar", "bazar", "tezi", "mandi", "lagayi", "chali", "gaya",
            "rupaye", "paisa", "lakh", "crore", "karobar",
        ]

        text_lower = text.lower()
        words = text_lower.split()
        if not words:
            return False

        hindi_word_count = sum(1 for w in words if w in hindi_roman_indicators)
        return (hindi_word_count / len(words)) > 0.08

    # ── Translation ───────────────────────────────────────────────────────────

    def translate_to_english(self, text: str, source_lang: str) -> TranslationResult:
        """Translate text to English using primary then fallback translator.

        Uses an in-process LRU cache to avoid re-translating identical content.
        """
        if not text or len(text.strip()) < 5:
            return TranslationResult(
                original_text=text,
                translated_text=text,
                source_lang=source_lang,
                is_translated=False,
                method="none",
                confidence=1.0,
            )

        cache_key = f"{source_lang}:{hashlib.md5(text.encode()).hexdigest()}"
        if cache_key in self._translation_cache:
            self.stats["cache_hits"] += 1
            return TranslationResult(
                original_text=text,
                translated_text=self._translation_cache[cache_key],
                source_lang=source_lang,
                is_translated=True,
                method="cache",
                confidence=0.85,
            )

        text_to_translate = text[:1000]
        translated: str | None = None
        method: str | None = None

        # Primary: Google via deep-translator (free, no API key)
        if self._primary_available:
            try:
                from deep_translator import GoogleTranslator

                translated = GoogleTranslator(
                    source=source_lang if source_lang != "hinglish" else "hi",
                    target="en",
                ).translate(text_to_translate)
                method = "google_translate"
                self.stats["translation_successes"] += 1
            except Exception as exc:
                logger.debug("Primary translation failed: %s", exc)

        # Fallback: offline argostranslate
        if not translated and self._fallback_available:
            try:
                import argostranslate.translate

                installed = argostranslate.translate.get_installed_languages()
                src = next((l for l in installed if l.code == "hi"), None)
                tgt = next((l for l in installed if l.code == "en"), None)
                if src and tgt:
                    translation = src.get_translation(tgt)
                    translated = translation.translate(text_to_translate)
                    method = "argostranslate_offline"
                    self.stats["translation_successes"] += 1
            except Exception as exc:
                logger.debug("Fallback translation failed: %s", exc)

        if not translated:
            self.stats["translation_failures"] += 1
            logger.warning(
                "Translation failed for %s text: %.50s...", source_lang, text
            )
            return TranslationResult(
                original_text=text,
                translated_text=text,
                source_lang=source_lang,
                is_translated=False,
                method="failed",
                confidence=0.0,
            )

        if len(self._translation_cache) < self._cache_max_size:
            self._translation_cache[cache_key] = translated

        return TranslationResult(
            original_text=text,
            translated_text=translated,
            source_lang=source_lang,
            is_translated=True,
            method=method,
            confidence=0.80 if method == "google_translate" else 0.70,
        )

    def translate_with_timeout(
        self, text: str, lang: str, timeout: float = 3.0
    ) -> TranslationResult:
        """Translate with a hard timeout so slow API calls don't stall the pipeline."""
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(self.translate_to_english, text, lang)
            try:
                return future.result(timeout=timeout)
            except concurrent.futures.TimeoutError:
                logger.warning("Translation timed out after %.1fs for %s text", timeout, lang)
                return TranslationResult(
                    original_text=text,
                    translated_text=text,
                    source_lang=lang,
                    is_translated=False,
                    method="timeout",
                    confidence=0.0,
                )

    # ── Main entry point ──────────────────────────────────────────────────────

    def process(self, text: str) -> ProcessedText:
        """Detect language, translate if needed, and return English text.

        This is the method called by the article parser.  English articles are
        returned immediately with zero translation overhead.
        """
        self.stats["total_articles"] += 1
        detection = self.detect_language(text)

        # Fast path: already English
        if detection.detected_lang == "en" and detection.confidence > 0.70:
            self.stats["english_articles"] += 1
            return ProcessedText(
                english_text=text,
                original_lang="en",
                was_translated=False,
                translation_confidence=1.0,
                is_hinglish=False,
            )

        if detection.is_hinglish:
            self.stats["hinglish_articles"] += 1
            result = self.translate_with_timeout(text, "hi")
            return ProcessedText(
                english_text=result.translated_text if result.is_translated else text,
                original_lang="hinglish",
                was_translated=result.is_translated,
                translation_confidence=result.confidence,
                is_hinglish=True,
            )

        if detection.detected_lang in self.SUPPORTED_LANGUAGES:
            if detection.detected_lang == "hi":
                self.stats["hindi_articles"] += 1
            else:
                self.stats["other_language_articles"] += 1
            result = self.translate_with_timeout(text, detection.detected_lang)
            return ProcessedText(
                english_text=result.translated_text if result.is_translated else text,
                original_lang=detection.detected_lang,
                was_translated=result.is_translated,
                translation_confidence=result.confidence,
                is_hinglish=False,
            )

        # Unknown language — pass through with low confidence flag
        self.stats["other_language_articles"] += 1
        return ProcessedText(
            english_text=text,
            original_lang=detection.detected_lang,
            was_translated=False,
            translation_confidence=0.3,
            is_hinglish=False,
        )

    # ── Statistics ────────────────────────────────────────────────────────────

    def get_stats(self) -> dict:
        total = max(self.stats["total_articles"], 1)
        non_english = max(
            self.stats["hindi_articles"]
            + self.stats["hinglish_articles"]
            + self.stats["other_language_articles"],
            1,
        )
        return {
            **self.stats,
            "hindi_pct": f"{self.stats['hindi_articles'] / total * 100:.1f}%",
            "hinglish_pct": f"{self.stats['hinglish_articles'] / total * 100:.1f}%",
            "translation_success_rate": (
                f"{self.stats['translation_successes'] / non_english * 100:.1f}%"
            ),
        }
