"""
Web Push notifications (VAPID) — the fleet's foundation-layer "reach the
operator regardless of what's open" sender.

**Why this exists.** Telegram (:mod:`krepis.telegram`) reaches a phone but is
a single flat channel with no click-through to a specific app screen. A
browser-tab-local ``Notification`` only fires while that tab's process is
alive. Web Push is the one channel that (a) reaches a specific installed
PWA/site regardless of whether its tab/app is currently open, and (b) is a
W3C standard with mature client libraries on every platform — no bespoke
per-consumer protocol. First consumer: symposion's turn-finished notification
(a browser tab-local ``Notification`` can't fire once the tab is fully
closed); flow-doctor's ``WebPushNotifierConfig`` is the second.

**Public API:**

- :func:`send_push` — primitive single-push send. Returns ``bool``, never
  raises (mirrors :func:`krepis.telegram.send_message`'s fire-and-forget
  contract) — a failed push must never block the caller's pipeline.

**Subscription shape.** ``subscription`` is the standard W3C
``PushSubscription.toJSON()`` object a browser hands back after
``pushManager.subscribe(...)``: ``{"endpoint": str, "keys": {"p256dh": str,
"auth": str}}``. Callers own persisting/looking these up — this module is
stateless, just the send primitive.

**VAPID identity.** One shared keypair for every krepis-based sender
(resolved via :func:`krepis.secrets.get_secret`, so ``/alpha-engine/
WEBPUSH_VAPID_PUBLIC_KEY`` + ``.../WEBPUSH_VAPID_PRIVATE_KEY``) — every
Python producer in the fleet pushes under the same identity, the same trust
boundary already accepted for the shared Telegram bot. A consumer that needs
an independent identity (e.g. a Node service that can't import krepis, or an
app whose subscriptions must not be replayable by a different krepis
consumer) passes explicit ``vapid_public_key``/``vapid_private_key`` instead.
Generate a keypair with the standard ``vapid`` CLI (ships with ``py_vapid``,
a ``pywebpush`` dependency: ``pip install py_vapid && vapid --gen``) or
Node's ``web-push generate-vapid-keys`` — both produce interoperable
standard EC P-256 keys. This module deliberately does not reimplement key
generation: a hand-rolled generator is a correctness risk with no upside
over the already-tested tooling the libraries ship.

**Optional dependency.** ``pywebpush`` is not a base krepis dependency —
install ``krepis[webpush]``. Calling :func:`send_push` without it installed
logs a warning and returns ``False`` rather than raising ``ImportError``,
matching :mod:`krepis.telegram`'s "misconfigured is a no-op, not a crash"
contract.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Final, Mapping

from krepis import fleet_events
from krepis.secrets import get_secret

logger = logging.getLogger(__name__)

VAPID_SUBJECT_DEFAULT: Final[str] = "mailto:ops@nousergon.com"

try:
    from pywebpush import WebPushException, webpush as _webpush
except ImportError:  # pragma: no cover - exercised via the "not installed" test
    _webpush = None
    WebPushException = None  # type: ignore[assignment,misc]


def send_push(
    subscription: Mapping[str, Any],
    *,
    title: str,
    body: str,
    url: str | None = None,
    tag: str | None = None,
    vapid_public_key: str | None = None,
    vapid_private_key: str | None = None,
    vapid_subject: str | None = None,
) -> bool:
    """Send a single Web Push notification to one subscription.

    Loads ``WEBPUSH_VAPID_PUBLIC_KEY`` / ``WEBPUSH_VAPID_PRIVATE_KEY`` /
    ``WEBPUSH_VAPID_SUBJECT`` via :func:`krepis.secrets.get_secret`
    (``required=False``) when the corresponding ``vapid_*`` kwarg isn't
    passed explicitly — same override-beats-secret-lookup shape as
    :func:`krepis.telegram.send_message`'s ``bot_token``/``chat_id``, so a
    caller with its own independent VAPID identity (e.g. a per-app keypair)
    can route through this transport without touching the process-global
    secret.

    The payload delivered to the client's service worker ``push`` handler is
    ``json.dumps({"title": title, "body": body, "url": url, "tag": tag})`` —
    ``url`` and ``tag`` are included even when ``None`` so the client-side
    handler can rely on the key always being present.

    :param subscription: The W3C ``PushSubscription.toJSON()`` object —
        ``{"endpoint": str, "keys": {"p256dh": str, "auth": str}}``.
    :param title: Notification title.
    :param body: Notification body text.
    :param url: Optional URL the client's ``notificationclick`` handler
        should open/focus.
    :param tag: Optional notification tag — the client may use this to
        collapse repeated notifications instead of stacking them.
    :param vapid_public_key: Optional explicit VAPID public key (skips
        secret lookup). ``pywebpush`` itself only needs the private key to
        sign the send, but this is accepted for symmetry with
        ``vapid_private_key`` and so callers can pass a subscription+keypair
        pair together without a mismatched partial override.
    :param vapid_private_key: Optional explicit VAPID private key (skips
        secret lookup).
    :param vapid_subject: Optional explicit VAPID JWT ``sub`` claim (a
        ``mailto:`` or ``https:`` URL identifying the sender) — skips
        secret lookup, falls back to :data:`VAPID_SUBJECT_DEFAULT`.
    :returns: ``True`` if the push service accepted the request, ``False``
        otherwise (``pywebpush`` not installed, no VAPID private key
        configured, or the send failed). Never raises.
    """
    if _webpush is None:
        logger.warning(
            "Web Push not sent — pywebpush is not installed (install krepis[webpush])"
        )
        return False

    private_key = vapid_private_key or get_secret("WEBPUSH_VAPID_PRIVATE_KEY", required=False)
    if not private_key:
        logger.warning("Web Push not configured — WEBPUSH_VAPID_PRIVATE_KEY is MISSING")
        return False
    subject = (
        vapid_subject
        or get_secret("WEBPUSH_VAPID_SUBJECT", required=False)
        or VAPID_SUBJECT_DEFAULT
    )
    # vapid_public_key isn't passed to pywebpush (it derives the public key
    # from the private key for the signed JWT) - accepted as a parameter
    # purely so callers can hand a matched (public, private) pair without a
    # confusing "why doesn't public_key do anything" gap, and reserved for
    # a future subscription-vs-key validation check.
    del vapid_public_key

    payload = json.dumps({"title": title, "body": body, "url": url, "tag": tag})

    ok = False
    try:
        _webpush(
            subscription_info=dict(subscription),
            data=payload,
            vapid_private_key=private_key,
            vapid_claims={"sub": subject},
        )
        ok = True
    except WebPushException as exc:
        status = getattr(getattr(exc, "response", None), "status_code", "?")
        logger.warning("Web Push send failed (%s): %s", status, exc)

    if not fleet_events.emission_suppressed():
        fleet_events.emit_alert_event(
            origin="webpush.send_push",
            body=title,
            severity_raw=None,
            dedup_key=None,
            channels={"webpush": ok},
        )

    return ok
