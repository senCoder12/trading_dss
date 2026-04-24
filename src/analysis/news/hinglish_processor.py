"""
Hinglish-specific sentiment processing for Indian financial news.

Hinglish (Hindi-English mix) is common in financial social media, some news
tickers, and informal market commentary.  Translation engines often miss
domain-specific Hinglish terms.  This module provides a keyword-based
sentiment boost layer that runs on top of VADER scores.
"""

from __future__ import annotations


class HinglishProcessor:
    """Apply domain-specific Hinglish keyword boosts to a base VADER score.

    Terms that translation engines frequently mistranslate or ignore are mapped
    to sentiment intensities.  The boost uses diminishing returns to prevent
    keyword stuffing from dominating the final score.
    """

    # Hinglish financial terms → sentiment intensity (0–1)
    HINGLISH_BULLISH_TERMS: dict[str, float] = {
        "tezi": 0.7,          # bullish run / strength
        "tezee": 0.7,
        "chhalaang": 0.6,     # leap / price jump
        "udaan": 0.6,         # surge / flight upward
        "chadha": 0.5,        # went up
        "badha": 0.5,         # increased
        "tej": 0.6,           # strong / fast market
        "zabardast": 0.6,     # tremendous / great result
        "achha": 0.3,         # good
        "mast": 0.4,          # great (informal)
        "all time high": 0.8,
        "naya uchhai": 0.7,   # new high
        "record todha": 0.7,  # broke record
        "khareedi": 0.4,      # buying pressure
        "bulls ka raj": 0.8,  # bulls' dominance
    }

    HINGLISH_BEARISH_TERMS: dict[str, float] = {
        "mandi": 0.7,         # bearish market / recession
        "giravat": 0.7,       # decline / fall
        "crash": 0.8,
        "bikwali": 0.6,       # selling pressure
        "dawab": 0.5,         # downward pressure
        "kamzor": 0.5,        # weak
        "bhari giravat": 0.8, # heavy fall
        "bears ka raj": 0.8,  # bears' dominance
        "neeche": 0.4,        # downward
        "gira": 0.5,          # fell
        "toot gaya": 0.7,     # broke down / support broken
        "dhakai": 0.6,        # hammered / beaten down
    }

    def boost_hinglish_sentiment(self, text: str, base_vader_score: float) -> float:
        """Apply Hinglish keyword boosts to a VADER base score.

        Positive boost for bullish terms, negative for bearish.
        Diminishing returns prevent heavy keyword concentration from
        overwhelming the base score.

        Args:
            text: The original (pre-translation) text, ideally the title.
            base_vader_score: VADER compound score on the translated text [-1, +1].

        Returns:
            Adjusted score clamped to [-1, +1].
        """
        text_lower = text.lower()
        bullish_boost = 0.0
        bearish_boost = 0.0
        matches = 0

        for term, intensity in self.HINGLISH_BULLISH_TERMS.items():
            if term in text_lower:
                bullish_boost += intensity * 0.12
                matches += 1

        for term, intensity in self.HINGLISH_BEARISH_TERMS.items():
            if term in text_lower:
                bearish_boost += intensity * 0.12
                matches += 1

        if matches == 0:
            return base_vader_score

        # Diminishing returns for multiple keyword matches
        total_boost = (bullish_boost - bearish_boost) * (1 / (1 + 0.1 * matches))
        return max(-1.0, min(1.0, base_vader_score + total_boost))
