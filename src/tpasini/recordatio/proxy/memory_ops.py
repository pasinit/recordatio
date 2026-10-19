"""Execute an approved memory-feedback op against the canonical store.

The nightly analyst proposes reversible memory-hygiene ops (tombstone /
merge / correct); a human approves them from the dashboard, and this module
performs the actual mutation. Every op is user-scoped and its target unit
ids must be COMPLETE UUIDs that exist in one of ``user_id``'s scopes. A
truncated/malformed uuid, or one that doesn't resolve exactly, makes the op
INVALID — the analyst occasionally emits a shortened id, and rather than
guess which unit it meant, an invalid op is refused (and the callers
auto-reject it) so one user's approval can never touch another's memory and
no mutation is ever applied to a guessed target.

Reversibility (the reason these are safe to one-click apply):
- ``tombstone`` sets ``valid_until`` (soft delete; the row is retained).
- ``merge`` tombstones the duplicates and leaves the canonical untouched.
- ``correct`` adds a new unit carrying the fixed content and tombstones the
  old one (supersede, never overwrite — the original stays recoverable and
  the new unit's provenance points back at it).
"""
import uuid
from datetime import UTC, datetime
from typing import Any

from tpasini.recordatio.models import KnowledgeUnit
from tpasini.recordatio.storage.base import SQLiteStore


class MemoryOpError(ValueError):
    """The op could not be applied (bad target, wrong scope, missing field)."""


class InvalidOpError(MemoryOpError):
    """The op is malformed — an incomplete/invalid unit uuid, or one that
    doesn't exist in scope. Distinct from other errors so callers can
    auto-reject (dismiss) it rather than leave it pending."""


def _is_complete_uuid(s: str) -> bool:
    """True only for a full canonical UUID (units are ``uuid4()`` strings).

    Rejects the analyst's occasional truncated ids (e.g. an 8-char prefix);
    ``uuid.UUID`` also normalises case/format, so we compare round-tripped.
    """
    try:
        return str(uuid.UUID(s)) == s.lower()
    except (ValueError, AttributeError, TypeError):
        return False


def _require_unit(store: SQLiteStore, user_id: str, unit_id: str) -> tuple[KnowledgeUnit, str]:
    if not unit_id:
        raise InvalidOpError("op target is missing a unit id")
    if not _is_complete_uuid(unit_id):
        raise InvalidOpError(f"unit id {unit_id!r} is not a complete uuid — invalid suggestion")
    pair = store.get_unit_for_user(user_id, unit_id)
    if pair is None:
        raise InvalidOpError(f"unit {unit_id!r} not found in {user_id}'s scopes")
    return pair


def apply_memory_op(store: SQLiteStore, user_id: str, op_type: str, target: dict[str, Any]) -> str:
    """Apply one op; return a human-readable result string for the audit log.

    Raises:
        MemoryOpError: unknown ``op_type`` or a target that doesn't resolve
            to unit(s) in ``user_id``'s scopes. Nothing is mutated in that
            case, so the caller can leave the op ``pending``.
    """
    now = datetime.now(UTC)
    if op_type == "tombstone":
        _unit, agent_id = _require_unit(store, user_id, str(target.get("unit_id", "")))
        store.invalidate(str(target["unit_id"]), now, agent_id)
        return f"tombstoned {target['unit_id']} (valid_until set)"

    if op_type == "merge":
        canonical_id = str(target.get("canonical_id", ""))
        _require_unit(store, user_id, canonical_id)  # canonical must be a valid, in-scope uuid
        duplicate_ids = target.get("duplicate_ids") or []
        if not isinstance(duplicate_ids, list) or not duplicate_ids:
            raise InvalidOpError("merge needs a non-empty duplicate_ids list")
        merged: list[str] = []
        for dup_id in duplicate_ids:
            if str(dup_id) == canonical_id:
                continue  # never tombstone the canonical onto itself
            _unit, agent_id = _require_unit(store, user_id, str(dup_id))
            store.invalidate(str(dup_id), now, agent_id)
            merged.append(str(dup_id))
        return f"merged {len(merged)} duplicate(s) onto {canonical_id}: {', '.join(merged)}"

    if op_type == "correct":
        unit_id = str(target.get("unit_id", ""))
        new_content = target.get("new_content")
        if not isinstance(new_content, str) or not new_content.strip():
            raise InvalidOpError("correct needs non-empty new_content")
        old, agent_id = _require_unit(store, user_id, unit_id)
        replacement = KnowledgeUnit(
            type=old.type,
            content=new_content,
            asserted_at=now,
            valid_from=now,
            provenance=f"correction of {unit_id}",
        )
        # The analyst re-corrects the same original source id each time, so a
        # naive add+invalidate leaves N sibling corrections all descending from
        # one source with none superseding another (the duplicate-cluster
        # pathology). Supersede any prior active correction of this source first
        # so at most one active unit ever descends from it.
        superseded = store.active_correction_ids(unit_id, agent_id)
        store.add(replacement, agent_id)
        for prior_id in superseded:
            store.invalidate(prior_id, now, agent_id)
        store.invalidate(unit_id, now, agent_id)
        return f"superseded {unit_id} with {replacement.id}"

    raise InvalidOpError(f"unknown op_type {op_type!r}")


__all__ = ["apply_memory_op", "MemoryOpError", "InvalidOpError"]
