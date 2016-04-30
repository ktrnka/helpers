from __future__ import unicode_literals
from __future__ import print_function

import unicodedata
import unittest

"""
Very simple assorted helpers for natural language processing that I've used a few times.
"""

_CHAR_TRANSLATIONS = {
    # chars to remove
    "\u00ae": None,
    "\u2122": None,

    # chars to normalize that aren't handled by combining char stripping
    "\u2018": "'",
    "\u2019": "'",
    "\u201c": '"',
    "\u201d": '"',
    "\u2013": "-",
    "\u2014": "-",
    "\u00bd": "1/2"
}

_CODEPOINT_TRANSLATIONS = {ord(k): v for k, v in _CHAR_TRANSLATIONS.items()}


def strip_diacritics(s):
    """Remove accents and other diacritics"""
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")


def normalize_unicode(s):
    """Remove trademark sign, normalize smart quotes, etc"""
    return s.translate(_CODEPOINT_TRANSLATIONS)


def ngramify(sequence, n=3, start="^", end="$"):
    if start:
        sequence = [start] + sequence
    if end:
        sequence = sequence + [end]

    for i in range(n, len(sequence) + 1):
        yield tuple(sequence[i - n:i])


def get_hash_indicators(items, num_features):
    features = [0 for _ in range(num_features)]

    for item in items:
        features[hash(item) % num_features] += 1

    return features


class NlpTests(unittest.TestCase):
    def test_ngramify(self):
        self.assertSequenceEqual([("^", "This", "is"), ("This", "is", "a"), ("is", "a", "test"), ("a", "test", "$")],
                                 list(ngramify("This is a test".split())))
