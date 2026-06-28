"""
Shared structured logging + Flow Doctor integration.

Replaces near-identical copies of ``log_config.py`` in alpha-engine-data
and alpha-engine/executor. Consumers call :func:`setup_logging` once at
process startup; subsequent call sites retrieve the Flow Doctor instance
via :func:`get_flow_doctor`.

Modes:

- Text (default): human-readable single-line log format.
- JSON: activated by ``ALPHA_ENGINE_JSON_LOGS=1``. Emits one JSON object
  per log record, including tracebacks for errors.

Flow Doctor is **default-on** (since 0.58.0): it activates whenever a
``flow_doctor_yaml`` path is provided to :func:`setup_logging` — passing a
yaml IS the opt-in. ``FLOW_DOCTOR_DISABLED=1`` (or ``FLOW_DOCTOR_ENABLED=0``)
is the kill switch; a test context (``PYTEST_CURRENT_TEST``) auto-disables
unless ``FLOW_DOCTOR_ALLOW_IN_TESTS=1``. This inverts the prior opt-in-per-
runtime default, whose failure mode was silently-dark runtimes. ERROR-level
records (including ``logger.exception``) fire the FlowDoctorHandler, which
dispatches per the yaml config (email + GitHub issue with dedup + rate
limits); wrap entrypoints in :func:`guard_entrypoint` / :func:`monitor_handler`
to also capture uncaught crashes.

In a deployed runtime (Lambda, or ``ALPHA_ENGINE_DEPLOYED=1`` on EC2/SF/spot)
a missing install / yaml / secret fails loud; in local dev / CI the same
conditions log a WARNING and skip activation. Requires the ``flow_doctor``
optional extra (``nousergon-lib[flow_doctor]``).
"""

from __future__ import annotations

import contextlib
import functools
import json
import logging
import os
import re
import sys
from datetime import datetime, timezone
from typing import Callable, Optional

# ``${VAR}`` interpolation tokens in a flow-doctor.yaml. flow-doctor
# resolves these from ``os.environ`` eagerly at ``FlowDoctor.from_config()``
# time — before any lazy ``get_secret()`` consumer-site call runs — so
# the seed below must populate them first.
_FD_VAR_RE = re.compile(r"\$\{([A-Z][A-Z0-9_]*)\}")

# Singleton populated by setup_logging() when FLOW_DOCTOR_ENABLED=1.
# ``Optional[object]`` typing avoids forcing a flow_doctor import here.
_fd_instance: Optional[object] = None


class JSONFormatter(logging.Formatter):
    """Emit log records as single-line JSON objects."""

    def format(self, record: logging.LogRecord) -> str:
        entry = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "module": record.module,
            "func": record.funcName,
            "msg": record.getMessage(),
        }
        if record.exc_info and record.exc_info[0] is not None:
            entry["exc"] = self.formatException(record.exc_info)
        if hasattr(record, "ctx"):
            entry["ctx"] = record.ctx
        return json.dumps(entry, default=str)


# Default-active secrets redaction patterns. Applied at the logging-handler
# layer by SecretsRedactingFilter so every log record reaching stdout / CW
# Logs / flow-doctor has the matching substrings replaced. Conservative
# by design — false positives (over-redaction) are visible; false negatives
# (leaked keys) reach public CW Logs.
#
# Pattern list is closed-form: every alpha-engine secret class known to leak
# at the data-collector / research / predictor / backtester log sites. Adding
# a new pattern is a one-line addition; opt-out per-record/per-attach is
# possible via ``record.no_redact = True`` (rare — see ``SecretsRedactingFilter``).
#
# Origin: 2026-05-24 audit on alpha-engine-research-runner CW Logs surfaced
# the FMP API key in plaintext inside HTTP-error WARNING lines (the FMP
# /stable 402 paid-tier errors mid-Research). alpha-engine-data #255 had
# shipped a repo-local scrubber for the same defect class earlier; lifting
# to the lib chokepoint per [[feedback_lift_invariants_to_chokepoint_after_second_recurrence]]
# closes the recurrence permanently across every repo that consumes
# ``krepis.logging.setup_logging``.
_SECRET_REDACTION_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    # URL-query-string credentials: `?apikey=...&symbol=X` /
    # `?api_key=...` / `?key=...` / `?token=...`. Conservative length
    # floor (16 chars) avoids redacting short ID-like tokens that look
    # alphanumeric but aren't secrets.
    (
        re.compile(
            r"([?&](?:apikey|api_key|key|token|access_token|auth_token)=)"
            r"([A-Za-z0-9_\-\.]{16,})"
        ),
        r"\1<REDACTED>",
    ),
    # AWS Access Key ID — exactly 16 alphanumerics after AKIA prefix.
    (re.compile(r"\bAKIA[A-Z0-9]{16}\b"), "<AWS_ACCESS_KEY_REDACTED>"),
    # Anthropic API keys — sk-ant-{api,sid,oat}-...
    (re.compile(r"\bsk-ant-[A-Za-z0-9_\-]{16,}"), "<ANTHROPIC_KEY_REDACTED>"),
    # OpenAI-style secret keys — sk-... 32+ chars (covers project keys).
    (re.compile(r"\bsk-[A-Za-z0-9_\-]{32,}"), "<OPENAI_KEY_REDACTED>"),
    # Authorization: Bearer <token> headers, case-insensitive.
    (
        re.compile(r"(authorization:\s*bearer\s+)([A-Za-z0-9_\-\.]+)", re.IGNORECASE),
        r"\1<REDACTED>",
    ),
    # GitHub personal access tokens — classic (ghp_*) and fine-grained
    # (github_pat_*). Both have well-known prefixes; min-length 30
    # guards against the prefix-alone false-positive.
    (re.compile(r"\b(?:ghp_|gho_|ghu_|ghs_|ghr_)[A-Za-z0-9_]{30,}"), "<GITHUB_TOKEN_REDACTED>"),
    (re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}"), "<GITHUB_TOKEN_REDACTED>"),
)


class SecretsRedactingFilter(logging.Filter):
    """logging.Filter that redacts known secret-shaped substrings from
    every log record's formatted message.

    Attached by default to the handler created by :func:`setup_logging`.
    Opt-out via the ``ALPHA_ENGINE_DISABLE_LOG_REDACTION=1`` env var (for
    cases where redaction obscures debugging of a non-secret pattern that
    matches a redaction regex by coincidence).

    Per-record opt-out is also possible via ``record.no_redact = True``
    on a record where the redaction is known-safe to skip. Use rarely —
    the default-active posture is a security property.

    Never raises. A regex compilation or replacement error is caught and
    the original record passes through unmodified (better to log
    something than nothing). The catch is intentionally broad: a logging
    filter that crashes is worse than one that silently no-ops.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        if getattr(record, "no_redact", False):
            return True
        try:
            # Render the message with its args so substitution sees the
            # actual emitted text (otherwise a `logger.warning("...%s...", key)`
            # would pass `record.msg` containing only the format string).
            rendered = record.getMessage()
            redacted = rendered
            for pattern, replacement in _SECRET_REDACTION_PATTERNS:
                redacted = pattern.sub(replacement, redacted)
            if redacted != rendered:
                # Replace the format string + clear args so downstream
                # formatters re-emit the redacted text rather than
                # re-interpolating.
                record.msg = redacted
                record.args = ()
        except Exception:  # noqa: BLE001 - filter MUST NOT crash logging
            pass
        return True


def get_flow_doctor():
    """Return the shared flow-doctor instance, or None if not initialized."""
    return _fd_instance


@contextlib.contextmanager
def guard_entrypoint():
    """Wrap a pipeline entrypoint body so an uncaught crash reaches flow-doctor.

    flow-doctor's logging handler only sees ``logger.error()/exception()``
    records — a bare ``raise`` that propagates out (the fleet's fail-loud
    posture) crashes the process *without* a report. Wrapping ``main()`` in
    this captures that path and re-raises (never swallows). No-ops cleanly when
    flow-doctor is inactive (local dev / disabled), so it's safe to add
    unconditionally::

        from krepis.logging import setup_logging, guard_entrypoint
        setup_logging("executor", flow_doctor_yaml=_FD_YAML)
        def main():
            with guard_entrypoint():
                run_pipeline()
    """
    fd = _fd_instance
    if fd is None:
        yield
        return
    with fd.guard():
        yield


def monitor_handler(func: Callable) -> Callable:
    """Decorator form of :func:`guard_entrypoint` for Lambda handlers.

    ``fd`` is resolved at call time (not decoration time) so it works when the
    handler is decorated at import — before ``setup_logging()`` runs — and
    no-ops when flow-doctor is inactive::

        @monitor_handler
        def handler(event, context):
            ...
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        fd = _fd_instance
        if fd is None:
            return func(*args, **kwargs)
        try:
            return func(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001 - reported then re-raised
            try:
                fd.report(exc)
            except Exception:
                pass
            raise
    return wrapper


def _seed_flow_doctor_secrets(yaml_path: str) -> None:
    """Populate the flow-doctor ``${VAR}`` secrets into ``os.environ``.

    flow-doctor resolves every ``${VAR}`` in its yaml from ``os.environ``
    eagerly inside ``FlowDoctor.from_config()``, before any consumer-site
    :func:`krepis.secrets.get_secret` call has had a chance to
    run. With the legacy ``ssm_secrets.load_secrets()`` bulk-load shim
    retired (PR 9g), systemd/Step-Functions-launched entrypoints have no
    ``.env`` source, so those ``${VAR}`` refs would resolve to nothing
    and flow-doctor's email + GitHub dispatch would silently misfire.

    This is the single chokepoint every repo reaches flow-doctor
    through, so seeding here closes the gap system-wide with no
    per-repo code. The var set is derived from the yaml itself rather
    than hardcoded — each repo's flow-doctor.yaml carries a different
    ``${VAR}`` set, and a yaml-added secret must not silently re-open
    the gap.

    Invariants:

    - **Local dev** (not deployed): a var already present in ``os.environ``
      wins — never overwritten, and the secrets backend is never touched
      for it. An explicitly-set shell var is the developer's intent.
    - **Deployed** (``_is_deployed()``): SSM is authoritative. A value
      already in ``os.environ`` is *overwritten* with the live SSM value
      when one resolves — a legacy ``~/.alpha-engine.env`` sourced before
      the entrypoint can otherwise pin a STALE secret (e.g. a rotated PAT)
      that wins over SSM. If SSM resolves nothing the existing value is
      left untouched (graceful SSM-miss fallback).
    - A genuinely unresolvable secret with no prior value is left
      **unset**, so flow-doctor's own ``ConfigError`` fires loudly rather
      than being masked with ``""`` (see ``feedback_no_silent_fails``).
    - A secrets-backend hiccup never blocks logging setup; it is logged
      at WARNING and any existing value is preserved.
    """
    try:
        with open(yaml_path, "r", encoding="utf-8") as fh:
            yaml_text = fh.read()
    except OSError:
        # Missing/unreadable yaml is reported by _attach_flow_doctor's
        # own os.path.exists guard with a clearer message.
        return

    from krepis.secrets import get_secret

    deployed = _is_deployed()
    for var in sorted(set(_FD_VAR_RE.findall(yaml_text))):
        existing = os.environ.get(var)
        # Local dev: an explicitly-set env var wins; never hit the backend.
        # Deployed: fall through so SSM can overwrite a stale sourced value.
        if existing and not deployed:
            continue
        try:
            value = get_secret(var, required=False)
        except Exception as exc:  # noqa: BLE001 - backend hiccup is non-fatal
            logging.getLogger(__name__).warning(
                "flow-doctor secret seed: get_secret(%s) raised %r; leaving %s",
                var, exc,
                "prior value in place" if existing
                else "unset so flow-doctor fails loudly",
            )
            continue
        if value:
            os.environ[var] = value
        # value is None: leave any existing value untouched (graceful
        # SSM-miss fallback when deployed); if there was none it stays
        # unset so flow-doctor's ConfigError fires loudly.


def _is_deployed() -> bool:
    """True when running in a deployed runtime (Lambda or marked EC2/SF/spot).

    Governs flow-doctor's ``strict`` posture: deployed → fail loud on a
    misconfigured/secret-missing flow-doctor (you WANT that surfaced in
    prod); not deployed (local dev / CI) → graceful WARN + skip so a
    developer who never set the secrets isn't blocked.

    ``AWS_LAMBDA_FUNCTION_NAME`` is set automatically by the Lambda
    runtime (free signal); ``ALPHA_ENGINE_DEPLOYED=1`` is the explicit
    marker added to EC2 systemd / Step Function / spot env blocks.
    """
    return bool(
        os.environ.get("AWS_LAMBDA_FUNCTION_NAME")
        or os.environ.get("ALPHA_ENGINE_DEPLOYED") == "1"
    )


def _flow_doctor_should_activate(yaml_path: Optional[str]) -> bool:
    """Decide whether flow-doctor activates (default-on when a yaml is given).

    Precedence (first match wins):
    1. ``FLOW_DOCTOR_DISABLED=1`` or ``FLOW_DOCTOR_ENABLED=0`` → off (kill switch).
    2. ``FLOW_DOCTOR_ENABLED=1`` → on. Explicit opt-in wins even under pytest —
       preserves the pre-0.58 contract (existing suites that assert activation).
    3. Test context → off, unless ``FLOW_DOCTOR_ALLOW_IN_TESTS=1``. Guards only
       the NEW default-on path so a consumer's suite never fires real
       issues/telegram by merely importing. Two signals, either suffices:
       ``PYTEST_CURRENT_TEST`` (set per-test by pytest) OR ``pytest`` already
       imported (``sys.modules``). The latter is load-bearing: entrypoints call
       ``setup_logging`` at module top, so under pytest the handler attaches at
       COLLECTION time — before any test runs, when PYTEST_CURRENT_TEST is not
       yet set. 2026-06-11: an alpha-engine-data test run leaked real alert
       emails + GitHub issues for synthetic fixture tickers through exactly
       this import-time gap.
    4. Default (unset): on **iff** a ``flow_doctor_yaml`` was provided — passing
       a yaml IS the opt-in. This inverts the old opt-in-per-runtime default
       whose failure mode was silently-dark runtimes (predictor/backtester/
       research were wired but never exported FLOW_DOCTOR_ENABLED=1).
    """
    if os.environ.get("FLOW_DOCTOR_DISABLED", "0") == "1":
        return False
    enabled = os.environ.get("FLOW_DOCTOR_ENABLED")
    if enabled == "0":
        return False
    if enabled == "1":
        return True
    in_test_context = bool(os.environ.get("PYTEST_CURRENT_TEST")) or (
        "pytest" in sys.modules
    )
    if in_test_context and (
        os.environ.get("FLOW_DOCTOR_ALLOW_IN_TESTS", "0") != "1"
    ):
        return False
    return bool(yaml_path)


def _attach_flow_doctor(
    yaml_path: str,
    exclude_patterns: list[str] | None = None,
    strict: bool = True,
) -> None:
    """Initialize the shared flow-doctor instance and attach a log handler.

    ``exclude_patterns`` is a list of regex strings forwarded to
    ``FlowDoctorHandler(exclude_patterns=...)``. Log records whose
    rendered message matches any pattern are dropped before entering
    the flow-doctor dispatch pipeline (email / GitHub issue). Use for
    benign ERROR-level noise that would otherwise dedup-spam on-call.

    ``strict`` (deployed runtimes) re-raises a missing-install / missing-yaml
    / missing-secret as a hard failure — a silently-degraded error monitor
    defeats the purpose. When not strict (local dev / CI), the same conditions
    log a WARNING and skip activation so a developer who never configured
    flow-doctor isn't blocked.
    """
    global _fd_instance
    _log = logging.getLogger("krepis.logging")

    try:
        import flow_doctor
    except ImportError as exc:
        msg = (
            "flow-doctor is not installed but a flow_doctor_yaml was provided. "
            "Install via nousergon-lib[flow_doctor] or add flow-doctor[diagnosis] "
            f"to requirements: {exc}"
        )
        if strict:
            raise RuntimeError(msg) from exc
        _log.warning("flow-doctor inactive (dev): %s", msg)
        return

    if not os.path.exists(yaml_path):
        msg = f"flow-doctor config not found at {yaml_path}"
        if strict:
            raise RuntimeError(msg)
        _log.warning("flow-doctor inactive (dev): %s", msg)
        return

    _seed_flow_doctor_secrets(yaml_path)
    # flow-doctor >=0.6.0 (the pinned floor, see pyproject ``flow_doctor``
    # extra) constructs via ``FlowDoctor.from_config()``; the deprecated
    # ``flow_doctor.init()`` free function was removed in 0.6.0. The fleet
    # floor is now flow-doctor>=0.6.0, so the transitional <0.6 init()
    # fallback has been dropped (config#647).
    #
    # ``strict`` flows into from_config: in prod a missing token raises a
    # ConfigError (fail loud); in dev FlowDoctor degrades to a no-op (_healthy
    # = False) with a stderr WARN instead of crashing the developer's run.
    try:
        _fd_instance = flow_doctor.FlowDoctor.from_config(
            config_path=yaml_path, strict=strict
        )
    except Exception as exc:  # noqa: BLE001 - strict re-raise below
        if strict:
            raise
        _log.warning("flow-doctor inactive (dev): construction failed: %s", exc)
        return

    handler_kwargs: dict = {"level": logging.ERROR}
    if exclude_patterns:
        handler_kwargs["exclude_patterns"] = exclude_patterns
    handler = flow_doctor.FlowDoctorHandler(_fd_instance, **handler_kwargs)
    logging.getLogger().addHandler(handler)


def setup_logging(
    name: str,
    flow_doctor_yaml: str | None = None,
    exclude_patterns: list[str] | None = None,
) -> None:
    """Configure the root logger for an Alpha Engine entrypoint.

    :param name: Logger name shown in the text-mode prefix
        (``"%(asctime)s %(levelname)s [{name}] %(message)s"``). Typically
        the module name (``"data-collector"``, ``"executor"``, etc.).
    :param flow_doctor_yaml: Absolute or CWD-relative path to the
        flow-doctor yaml config. Required if ``FLOW_DOCTOR_ENABLED=1``;
        ignored otherwise.
    :param exclude_patterns: Optional list of regex strings. When
        ``FLOW_DOCTOR_ENABLED=1``, these are forwarded to
        ``FlowDoctorHandler`` so matching ERROR-level records are
        dropped before the flow-doctor dispatch pipeline. Use sparingly
        — this silences *alerts*, not logs. The records still appear in
        stdout / JSON logs; only flow-doctor's email + GitHub issue
        routing is suppressed. Example: the executor passes
        ``[r"Error 10197"]`` to suppress benign IB Gateway noise when
        the iOS app steals the live-data session.

    Env vars consulted:

    - ``ALPHA_ENGINE_JSON_LOGS`` — ``"1"`` enables JSON formatter.
    - ``FLOW_DOCTOR_ENABLED`` — ``"1"`` attaches FlowDoctorHandler.
    """
    json_mode = os.environ.get("ALPHA_ENGINE_JSON_LOGS", "0") == "1"

    handler = logging.StreamHandler()
    if json_mode:
        handler.setFormatter(JSONFormatter())
    else:
        handler.setFormatter(logging.Formatter(
            f"%(asctime)s %(levelname)s [{name}] %(message)s"
        ))

    # Attach secrets-redacting filter by default. Closes the FMP-API-key /
    # AWS-key / Anthropic-key / GitHub-PAT plaintext-log class system-wide
    # (every consumer of krepis.logging.setup_logging inherits
    # the filter on its next lib pin bump). Opt-out via
    # ALPHA_ENGINE_DISABLE_LOG_REDACTION=1 for the rare debugging scenario
    # where a redaction regex matches a non-secret pattern.
    if os.environ.get("ALPHA_ENGINE_DISABLE_LOG_REDACTION", "0") != "1":
        handler.addFilter(SecretsRedactingFilter())

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(logging.INFO)

    # An explicit FLOW_DOCTOR_ENABLED=1 with no yaml is a misconfiguration the
    # operator wants surfaced loudly (it can't be silently default-off anymore).
    if os.environ.get("FLOW_DOCTOR_ENABLED") == "1" and not flow_doctor_yaml:
        raise RuntimeError(
            "FLOW_DOCTOR_ENABLED=1 but setup_logging() was not given a "
            "flow_doctor_yaml path"
        )

    # Default-on: flow-doctor activates whenever a yaml is provided, unless a
    # kill switch / test context says otherwise (see _flow_doctor_should_activate).
    # strict (fail loud on missing install/yaml/secret) applies in a deployed
    # runtime OR whenever the operator explicitly set FLOW_DOCTOR_ENABLED=1 — an
    # explicit opt-in wants its misconfig surfaced. The default-on path in local
    # dev / CI stays lenient (WARN + skip) so it never blocks a developer.
    if flow_doctor_yaml and _flow_doctor_should_activate(flow_doctor_yaml):
        strict = _is_deployed() or os.environ.get("FLOW_DOCTOR_ENABLED") == "1"
        _attach_flow_doctor(
            flow_doctor_yaml,
            exclude_patterns=exclude_patterns,
            strict=strict,
        )
