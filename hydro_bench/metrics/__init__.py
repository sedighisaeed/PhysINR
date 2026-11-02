"""Image quality assessment: full-reference, no-reference, and statistics.

The metrics layer is deliberately side-effect free. Given arrays it returns
numbers; it never touches the filesystem, never imports a method's code, and
never sees the training split.
"""

from . import full_reference, no_reference, statistics

__all__ = ["full_reference", "no_reference", "statistics"]
