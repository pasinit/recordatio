"""Backward compatibility with the project's former name, Memento.

Deployed clients and configs still use ``MEMENTO_*`` env vars and
``x-memento-*`` request headers / body flags. Both keep working until every
caller has moved to the ``RECORDATIO_*`` / ``x-recordatio-*`` names; then this
module can be deleted.
"""

import os
from collections.abc import Mapping, MutableMapping
from typing import Any

_LEGACY_ENV_PREFIX = "MEMENTO_"
_ENV_PREFIX = "RECORDATIO_"
_LEGACY_KEY_PREFIX = "x-memento-"
_KEY_PREFIX = "x-recordatio-"


def alias_legacy_env(environ: MutableMapping[str, str] = os.environ) -> None:
    """Copy each ``MEMENTO_X`` env var to ``RECORDATIO_X`` unless already set."""
    for key, value in list(environ.items()):
        if key.startswith(_LEGACY_ENV_PREFIX):
            environ.setdefault(_ENV_PREFIX + key[len(_LEGACY_ENV_PREFIX):], value)


def header(headers: Mapping[str, str], suffix: str, default: str | None = None) -> str | None:
    """Read ``x-recordatio-<suffix>``, falling back to ``x-memento-<suffix>``."""
    value = headers.get(_KEY_PREFIX + suffix)
    if value is None:
        value = headers.get(_LEGACY_KEY_PREFIX + suffix)
    return default if value is None else value


def pop_flag(body: dict[str, Any], suffix: str, default: Any = None) -> Any:
    """Pop ``x-recordatio-<suffix>`` and ``x-memento-<suffix>`` from a request body.

    Both keys are always removed so neither is forwarded upstream; the new
    name wins when both are present.
    """
    legacy = body.pop(_LEGACY_KEY_PREFIX + suffix, default)
    return body.pop(_KEY_PREFIX + suffix, legacy)
