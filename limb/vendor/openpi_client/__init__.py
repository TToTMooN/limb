"""Vendored subset of openpi-client — only the bits limb needs to talk to an
OpenPI native policy server (pi0, pi0.5, pi0-FAST served via
``openpi/scripts/serve_policy.py``).

Source: https://github.com/Physical-Intelligence/openpi/tree/main/packages/openpi-client
License: Apache 2.0 (see NOTICE.md)

Why vendored: openpi-client isn't on PyPI; the upstream package lives inside
the openpi repo as a path source. Pulling that as a dependency means anyone
running limb needs a sibling openpi clone, which is awkward. The bit we use
is tiny (~130 lines) and stable.

If you need the full openpi-client API (e.g. ``runtime/``, ``image_tools``),
install the upstream package and import from ``openpi_client`` directly —
this vendor copy only covers websocket policy clients.
"""

from limb.vendor.openpi_client.websocket_client_policy import WebsocketClientPolicy  # noqa: F401

__all__ = ["WebsocketClientPolicy"]
