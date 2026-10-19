"""User-level identity store (IDENTITY_DESIGN.md rev. 2).

The principal — the human user behind a set of agents — is a single entity
per user, system-wide, living in the reserved scope ``{user_id}/_user``
(S-RULE). This module is the one place that knows the reserved-scope grammar:
routing (extraction), retrieval composition, the ensure API, and the
migration all use these helpers so the convention cannot drift.
"""

import logging
import uuid
from datetime import UTC, datetime

from tpasini.recordatio.entity import Entity
from tpasini.recordatio.entity_resolver import default_canonical_key, normalize_label
from tpasini.recordatio.models import EntityAlias
from tpasini.recordatio.storage.base import SQLiteStore

logger = logging.getLogger(__name__)

# Reserved agent slug: "{user_id}/_user" is the user-level store. The slug is
# rejected for normal traffic (config API, proxy headers) — only the routing
# layer, the ensure API, and the migration write it.
RESERVED_USER_SLUG = "_user"

# Namespace for deterministic mirror-entity ids (uuid5). Fixed forever: the
# same (user_scope, type, label) must map to the same id across processes and
# across the migration/live-write boundary.
RECORDATIO_NS = uuid.UUID("9a52a191-2c30-5a8c-9c5d-2f6f6e2b62d1")

PRINCIPAL_LABEL = "User"


def user_scope(agent_id: str) -> str:
    """The user-level store scope for an agent scope.

    ``user_id/agent_name -> user_id/_user``; survives the empty slug
    (``tommaso/``) and underscore slugs (``tommaso/_maintenance``). A bare
    scope (no ``/``) maps to ``<scope>/_user`` — the whole id is its user
    segment.
    """
    return agent_id.partition("/")[0] + "/" + RESERVED_USER_SLUG


def is_user_scope(agent_id: str) -> bool:
    return agent_id.partition("/")[2] == RESERVED_USER_SLUG


def is_reserved_slug(slug: str) -> bool:
    return slug == RESERVED_USER_SLUG


def visibility(agent_id: str) -> tuple[str, ...]:
    """Composed read visibility for an agent scope: ``{S, user_scope(S)}``.

    Never another user's scope; never another agent's private scope. A read
    under the user scope itself sees only the user scope.
    """
    scope = user_scope(agent_id)
    return (agent_id,) if agent_id == scope else (agent_id, scope)


def principal_canonical_key(scope: str) -> str:
    """Canonical key of the principal in its user scope.

    ``stable_user_mention``'s existing grammar applied to the reserved scope:
    ``agent_user:{user_id}/_user``. The unique index
    ``(agent_id, type, canonical_key)`` makes this one row per user.
    """
    return f"agent_user:{scope}"


def mirror_entity_id(scope: str, entity_type: str | None, label: str) -> str:
    """Deterministic id for a non-principal endpoint mirrored into the user
    scope (§2.3): cross-agent dedup by construction."""
    return str(uuid.uuid5(RECORDATIO_NS, f"{scope}|{entity_type or ''}|{normalize_label(label)}"))


def get_principal(store: SQLiteStore, scope: str) -> Entity | None:
    """The user's principal entity, or ``None`` pre-ensure/pre-migration."""
    return store.get_entity_by_canonical_key(principal_canonical_key(scope), scope)


def ensure_principal(
    store: SQLiteStore,
    user_id: str,
    *,
    display_name: str | None = None,
    aliases: list[str] | None = None,
    now: datetime | None = None,
) -> tuple[Entity, bool]:
    """Idempotent ensure of the user's principal (§5).

    Creates the principal only if absent; upserts ``display_name``/``aliases``
    through the collision-guarded alias path on every call. Returns
    ``(principal, created)``. Both the login-time API and the extraction
    ensure-on-write safety net converge here, so divergence is impossible by
    construction.
    """
    scope = f"{user_id}/{RESERVED_USER_SLUG}"
    now = now or datetime.now(UTC)
    principal = get_principal(store, scope)
    created = False
    if principal is None:
        entity = Entity(
            label=PRINCIPAL_LABEL,
            type="person",
            canonical_label=display_name or PRINCIPAL_LABEL,
            canonical_key=principal_canonical_key(scope),
            asserted_at=now,
        )
        store.add_entity(entity, scope)
        principal = get_principal(store, scope) or entity
        created = True
    for alias in [display_name, *(aliases or [])]:
        if alias:
            add_principal_alias(store, principal.id, alias, scope, source="principal-ensure", now=now)
    return principal, created


def add_principal_alias(
    store: SQLiteStore,
    principal_id: str,
    alias: str,
    scope: str,
    *,
    agent_scope: str | None = None,
    source: str = "identity",
    now: datetime | None = None,
) -> bool:
    """Add a standing alias to the principal, collision-guarded (§2.1).

    The alias is NOT added (and the conflict is logged) when an entity other
    than the principal — in the asserting agent's scope or in the user scope —
    already carries that normalized label or alias. Load-bearing: a genuine
    third-party namesake must not be swallowed by a noisy ``has_name`` fact.
    """
    norm = normalize_label(alias)
    if not norm or norm == normalize_label(PRINCIPAL_LABEL):
        return False
    check_scopes = [scope] + ([agent_scope] if agent_scope and agent_scope != scope else [])
    for check in check_scopes:
        for entity in store.list_entities(check):
            if entity.id != principal_id and normalize_label(entity.label) == norm:
                logger.info(
                    "principal alias %r blocked: entity %s (%r) in scope %s carries that label",
                    alias, entity.id, entity.label, check,
                )
                return False
        for row in store.find_entity_aliases(alias, check):
            if row.entity_id != principal_id:
                logger.info(
                    "principal alias %r blocked: already an alias of entity %s in scope %s",
                    alias, row.entity_id, check,
                )
                return False
    existing = {normalize_label(a.alias) for a in store.list_entity_aliases(principal_id, scope)}
    if norm in existing:
        return True
    store.upsert_entity_alias(
        EntityAlias(entity_id=principal_id, alias=alias, source=source, created_at=now or datetime.now(UTC)),
        scope,
    )
    return True


def principal_binding_labels(store: SQLiteStore, scope: str, user_entity_label: str) -> dict[str, str]:
    """Normalized label -> original text for every label that binds a mention
    to the principal (E-RULE): the configured user label plus the principal's
    standing alias set in the user scope."""
    labels: dict[str, str] = {}
    norm_user = normalize_label(user_entity_label)
    if norm_user:
        labels[norm_user] = user_entity_label
    principal = get_principal(store, scope)
    if principal is not None:
        for alias_row in store.list_entity_aliases(principal.id, scope):
            norm = normalize_label(alias_row.alias)
            if norm:
                labels.setdefault(norm, alias_row.alias)
    return labels


def mention_binds_principal(label: str, mention_type: str | None, binding_labels: dict[str, str]) -> bool:
    """The E-RULE precision bounds: full-label normalized equality only (never
    substring) and a type guard — a ``concept``/``role``-typed mention is an
    ordinary entity."""
    if mention_type not in (None, "", "person"):
        return False
    norm = normalize_label(label)
    return bool(norm) and norm in binding_labels


def ensure_mirror_entity(
    store: SQLiteStore, source: Entity, scope: str, *, source_agent_id: str | None = None
) -> str:
    """Mirror a non-principal endpoint into the user scope (§2.3); returns the
    id the user-level relation must point at.

    Mirrors carry label/type/description and keep their own ``wikidata_id``
    (third parties are public-matchable). Deterministic ids make repeated
    mirroring from any agent converge on one row. If the deterministic insert
    is blocked by the per-scope label-unique index (same LOWER(label), other
    id), the existing user-scope entity is the mirror.
    """
    mid = mirror_entity_id(scope, source.type, source.label)
    existing = store.get_entity_by_id(mid, scope)
    if existing is not None:
        return mid
    store.add_entity(
        Entity(
            id=mid,
            label=source.label,
            type=source.type,
            subtype=source.subtype,
            canonical_label=source.canonical_label or source.label,
            canonical_key=default_canonical_key(source.type, source.label),
            wikidata_id=source.wikidata_id,
            description=source.description,
            wikipedia_url=source.wikipedia_url,
            asserted_at=source.asserted_at,
        ),
        scope,
        source_agent_id=source_agent_id,
    )
    if store.get_entity_by_id(mid, scope) is not None:
        return mid
    by_label = store.get_entity_by_label(source.label, scope)
    return by_label.id if by_label is not None else mid
