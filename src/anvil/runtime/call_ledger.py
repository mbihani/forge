"""Per-row ledger of every model call an agent makes during an eval.

A domain engine arms one eval row at a time (:meth:`CallLedger.begin` /
:meth:`CallLedger.end`); every model call for that row then passes through
:meth:`CallLedger.call`, however the agent reached the transport. It gives
forge three things no agent can opt out of:

* **A call policy.** With ``max_sends_per_document`` set
  (``harness/config.yaml > eval.call_policy``), a document — the exact
  bytes sent, e.g. a whole PDF or a page split out of it — may go to a
  model at most that many times per row, whatever the model or input mode.
  Racing the same call and re-running a row as a fallback are refused with
  :class:`DuplicateCallError`; complementary calls on *different*
  documents stay allowed.
* **Honest cost.** :meth:`end` waits for calls still in flight (e.g. one the
  agent stopped waiting for) and returns all of them, so the row's cost
  counts what was actually spent. :func:`price_calls` prices them.
* **Infra vs agent failures.** Each record keeps the call's error, and
  :func:`is_infra_error` tells transport failures (timeouts, 5xx, 429,
  connection resets) from the agent's own mistakes (a rejected parameter,
  a duplicate send), so an engine can re-run a row that failed for
  reasons the agent did not cause.

A row that was never armed passes through unrecorded.
"""

from __future__ import annotations

import hashlib
import re
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from anvil.catalog import cost_usd

# Transport failures the agent did not cause. Matched against
# "<ExceptionType>: <message>" strings, as engines record them.
_INFRA_ERROR = re.compile(
    r"TimeoutError|timed out|HTTP 5\d\d|HTTP 429|URLError|ConnectionResetError|"
    r"ConnectionRefusedError|RemoteDisconnected|IncompleteRead|"
    r"Temporary failure in name resolution",
    re.IGNORECASE,
)


def is_infra_error(error: str | None) -> bool:
    """True when ``error`` is a transport failure (not the agent's doing)."""
    return bool(error) and bool(_INFRA_ERROR.search(error))


def describe_error(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"


class DuplicateCallError(RuntimeError):
    """A row's document was already sent the maximum number of times."""


class CallLedger:
    """Every model call made for an eval row, while the eval has it armed."""

    def __init__(self) -> None:
        self._cond = threading.Condition()
        self._rows: dict[str, dict[str, Any]] = {}

    def begin(self, row_id: str, *, max_sends_per_document: int | None = None) -> None:
        """Arm ``row_id``; ``max_sends_per_document=None`` records without limiting."""
        with self._cond:
            self._rows[row_id] = {
                "sends": {},
                "calls": [],
                "inflight": 0,
                "max_sends": max_sends_per_document,
            }

    def end(self, row_id: str, *, timeout_s: float = 900.0) -> list[dict[str, Any]]:
        """Disarm ``row_id``; wait for in-flight calls, return every call record."""
        deadline = time.monotonic() + timeout_s
        with self._cond:
            row = self._rows.get(row_id)
            if row is None:
                return []
            while row["inflight"] and (left := deadline - time.monotonic()) > 0:
                self._cond.wait(left)
            del self._rows[row_id]
            return list(row["calls"])

    @contextmanager
    def call(self, row_id: str, document: bytes, info: dict[str, Any]) -> Iterator[dict[str, Any]]:
        """Admit one call; yield its record (fill in token usage).

        Raises :class:`DuplicateCallError` (before any request is made) when
        the policy's send limit for ``document`` is used up. A call that
        raises is still recorded, with its ``error``.
        """
        record = dict(info)
        with self._cond:
            row = self._rows.get(row_id)
            if row is not None:
                key = hashlib.sha256(document).hexdigest()
                sent = row["sends"].get(key, 0)
                if row["max_sends"] is not None and sent >= row["max_sends"]:
                    raise DuplicateCallError(
                        f"row {row_id}: this document was already sent {sent} time(s), "
                        f"the call policy allows {row['max_sends']} (no racing / fallback "
                        "re-runs; send a different document instead)"
                    )
                row["sends"][key] = sent + 1
                row["inflight"] += 1
        try:
            yield record
        except BaseException as exc:
            record["error"] = describe_error(exc)[:500]
            raise
        finally:
            if row is not None:
                with self._cond:
                    row["calls"].append(record)
                    row["inflight"] -= 1
                    self._cond.notify_all()


def price_calls(
    calls: list[dict[str, Any]], prices: dict[str, dict[str, Any]]
) -> list[float | None]:
    """Dollar cost of each call record (``model``, ``input_tokens``, ``output_tokens``)."""
    return [
        cost_usd(
            prices.get(str(c.get("model"))),
            int(c.get("input_tokens", 0) or 0),
            int(c.get("output_tokens", 0) or 0),
        )
        for c in calls
    ]


# The process-wide ledger domain engines arm.
LEDGER = CallLedger()
