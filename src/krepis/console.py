"""console.py — the canonical console deep-link URL chokepoint.

The Nous Ergon dashboard ("console") renders one page per producer surface
(EOD report, model zoo, signal-quality analysis, …). Each producer's
digest email deep-links to its page so the operator can jump from the
2-line summary to the full render.

Three producers had independently hand-rolled the same one-liner — a base
URL + a page slug + an optional ``?date=YYYY-MM-DD`` query — each with its
own copy of the base literal. The 3rd adoption is the consolidation signal
(alpha-engine-config#1300): lift the builder + the base constant here so
the host lives in exactly one place.

That single home is what made the 2026-08-12 repair a one-line change: the
``console.nousergon.ai`` hostname was handed to nousergon-console v2, and
every digest deep-link in the fleet 404'd until this constant moved to
``dashboard.nousergon.ai`` (alpha-engine-config#6140).

Usage (the per-repo slug stays local — it's the cross-repo contract with
the dashboard page's ``url_path``)::

    from krepis.console import console_url

    # No date — landing page:
    console_url("model-zoo")
    # -> "https://dashboard.nousergon.ai/model-zoo"

    # Date-keyed deep-link:
    console_url("eod-report", date="2026-06-22")
    # -> "https://dashboard.nousergon.ai/eod-report?date=2026-06-22"

    # Base override (e.g. a debug host); a trailing slash is tolerated:
    console_url("model-zoo", base="https://console.example.com/")
    # -> "https://console.example.com/model-zoo"

The default base is env-overridable via ``CONSOLE_BASE_URL`` so a deploy
that points at a vanity / staging host doesn't need a code change; an
explicit ``base=`` argument still wins over the environment.
"""

from __future__ import annotations

import os

# The default console host. Resolved at call time from the
# ``CONSOLE_BASE_URL`` env var when set (deploy-time override for a vanity /
# staging host), else this literal. The literal lives here ONCE — producers
# no longer carry their own copy.
#
# This host is ``dashboard.nousergon.ai``, NOT ``console.nousergon.ai``.
# Every slug built here is a Streamlit page ``url_path`` in
# alpha-engine-dashboard, and Streamlit is served at
# ``dashboard.nousergon.ai`` (:8501). ``console.nousergon.ai`` was cut over
# to nousergon-console v2 (:5180), which is a fleet ENTITY INDEX serving
# only ``/``, ``/search``, ``/registry/<name>`` and ``/component/<id>`` — it
# 404s every slug in this module. Do not point this back at
# ``console.nousergon.ai`` for a page slug; if a v2 entity route is ever
# needed, build it against the v2 host explicitly rather than moving this
# default (alpha-engine-config#6140).
DEFAULT_CONSOLE_BASE_URL = "https://dashboard.nousergon.ai"

#: Env var that overrides :data:`DEFAULT_CONSOLE_BASE_URL` at call time.
CONSOLE_BASE_URL_ENV = "CONSOLE_BASE_URL"


def console_url(slug: str, *, date: str | None = None, base: str | None = None) -> str:
    """Build a console deep-link: ``{base}/{slug}`` (+ ``?date=`` if given).

    Args:
        slug: The dashboard page slug (its ``url_path``), e.g. ``"eod-report"``.
            Kept caller-side — it's the cross-repo contract with the page.
        date: Optional ``YYYY-MM-DD`` string. When given, appended as a
            ``?date=`` query so the page opens the exact cycle. When ``None``
            the bare landing URL is returned.
        base: Optional base-URL override. Precedence: this arg, else the
            ``CONSOLE_BASE_URL`` env var, else :data:`DEFAULT_CONSOLE_BASE_URL`.
            A trailing slash is tolerated and stripped.

    Returns:
        The console URL string.
    """
    resolved = base or os.environ.get(CONSOLE_BASE_URL_ENV) or DEFAULT_CONSOLE_BASE_URL
    resolved = resolved.rstrip("/")
    if date:
        return f"{resolved}/{slug}?date={date}"
    return f"{resolved}/{slug}"
