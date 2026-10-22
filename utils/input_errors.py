"""Platform-neutral text insertion outcomes.

An accepted operating-system input event is not an acknowledgement from the
target editor (or remote desktop). Never automatically replay a transcript
after an error that may have inserted some of it.
"""
from __future__ import annotations

from typing import Callable


CancelCheck = Callable[[], bool] | None


class InputInsertionError(RuntimeError):
    """Insertion failed, with an explicit indication of replay safety."""

    def __init__(
        self, message: str, *, events_inserted: int = 0,
        may_have_inserted: bool = False,
    ) -> None:
        super().__init__(message)
        self.events_inserted = events_inserted
        self.may_have_inserted = may_have_inserted or events_inserted > 0


class InputInsertionCancelled(InputInsertionError):
    """Cancellation stopped insertion; already accepted input cannot be undone."""


def check_insertion_cancelled(
    cancel_check: CancelCheck, *, events_inserted: int = 0,
) -> None:
    if cancel_check is not None and cancel_check():
        raise InputInsertionCancelled(
            "Text insertion cancelled. Already inserted text was not replayed or removed."
            if events_inserted else "Text insertion cancelled before typing.",
            events_inserted=events_inserted,
        )
