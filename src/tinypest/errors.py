"""Exception types raised by tinypest.

Every error the package raises on purpose derives from :class:`TinyPestError`, so callers
can catch the whole family with one ``except`` clause and still tell the cases apart.
"""

from __future__ import annotations


class TinyPestError(Exception):
    """Base class for all errors raised deliberately by tinypest."""


class AnnotationError(TinyPestError):
    """The annotation file is missing, malformed or internally inconsistent."""


class DatasetError(TinyPestError):
    """An image cannot be read or does not match its annotation record."""
