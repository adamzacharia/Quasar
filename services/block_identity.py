"""Deterministic per-block identity for eval feedback (Feature 4).

A "block" is one rendered card in an assistant turn: the text answer, a data
table, a figure, a paper grid, a notebook. The frontend message model is flat
(one message -> one type) and the store already spawns a separate message per
card, so a block is exactly a card-message.

The id is minted ONCE, at the moment the card is emitted, and persisted into
`messages.metadata` (rich_meta) alongside the card. History replay reads it
back; it never recomputes it. That ordering matters: sse.py's emission order
depends on eager-vs-done paths and dedup (`_all_results`, `_eagerly_emitted`),
so a recomputed ordinal could drift and orphan a rating that was already keyed
to the old id. A stored id cannot drift.

The ordinal is therefore only required to be unique-and-stable WITHIN a single
turn's emission, which a per-kind monotonic counter guarantees.
"""

from __future__ import annotations

import hashlib
from typing import Dict

# Kinds that carry a rating. `web_sources` is deliberately absent: it is a
# provider-attribution strip, not a science block, and the eval-mode gate must
# not demand a star on it.
RATEABLE_BLOCK_KINDS = ("text", "data", "plotly", "image", "papers", "notebook")


def compute_block_id(
    conversation_id: str,
    run_id: str,
    block_kind: str,
    ordinal: int,
) -> str:
    """sha1 over the turn coordinates. Pure — same inputs, same id, forever."""
    raw = "|".join((
        str(conversation_id or ""),
        str(run_id or ""),
        str(block_kind or ""),
        str(int(ordinal)),
    ))
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


class BlockIdAllocator:
    """Hands out stable block ids for one turn, one counter per kind.

    Per-kind (rather than one global counter) so that interleaving between
    kinds cannot shift a block's ordinal: a figure emitted eagerly mid-stream
    must not renumber the data card that lands at the end of the turn.
    """

    def __init__(self, conversation_id: str, run_id: str):
        self.conversation_id = conversation_id or ""
        self.run_id = run_id or ""
        self._ordinals: Dict[str, int] = {}

    def next(self, block_kind: str) -> str:
        """Mint the next id for `block_kind`. Call ONLY when actually emitting —
        a skipped/deduped card must not burn an ordinal."""
        ordinal = self._ordinals.get(block_kind, 0)
        self._ordinals[block_kind] = ordinal + 1
        return compute_block_id(self.conversation_id, self.run_id, block_kind, ordinal)

    def peek(self, block_kind: str) -> int:
        """How many ids of this kind have been handed out (tests/diagnostics)."""
        return self._ordinals.get(block_kind, 0)
