"""ViserApp — central coordinator for viser-based visualization panels.

Owns a single ViserServer and provides a registration API for panels.
Each panel is a self-contained component responsible for its own GUI
folder and/or scene elements. Panels are added declaratively::

    app = ViserApp()
    app.add_panel("cameras", CameraPanel(image_size=224))
    app.add_panel("urdf", URDFPanel(bimanual=True))
    app.add_panel("session", SessionPanel())

    # In the control loop:
    app.update(obs)

    # On shutdown:
    app.close()

Threading model
---------------
- **Main thread** calls ``update(obs)`` synchronously from the control loop
  (~100 Hz). All panel ``update()`` methods run on the main thread.
- **Viser server thread** (internal to viser) handles WebSocket I/O and fires
  ``on_click`` / ``on_update`` GUI callbacks on its own thread.  Panels that
  receive callbacks (SessionPanel, RecordingPanel) must use thread-safe
  structures (``deque``, ``threading.Lock``) to bridge to the main thread.
- **Portal subprocess** (YamViserAgent) has its own ViserServer + panels in a
  separate process. The main-process ViserApp is independent.

Data flow::

    Main thread (100 Hz control loop)
    │
    ├─ agent.act(obs)          Portal RPC
    ├─ session.step(obs, act)  trigger poll (deque.popleft) + recording
    ├─ env.step(action)        Portal RPC
    └─ viser_app.update(obs)   sequential fan-out to panels:
         ├─ CameraPanel.update(obs)      extract RGB, update thumbnails
         ├─ URDFPanel.update(obs)        update joint visualization
         └─ RecordingPanel.update(obs)   write frames if recording

    Viser server thread (async WebSocket)
    │
    ├─ SessionPanel on_click   → deque.append(signal)    [thread-safe]
    └─ RecordingPanel on_click → Lock + toggle recording [thread-safe]
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Protocol, runtime_checkable

import viser
from loguru import logger


@runtime_checkable
class ViserPanel(Protocol):
    """A self-contained viser GUI/scene component.

    Panels are attached to a ViserServer and manage their own GUI elements
    (sidebar folders) and/or scene elements (3D viewport objects).
    """

    def attach(self, server: viser.ViserServer) -> None:
        """Create GUI/scene elements on the given server."""
        ...

    def detach(self) -> None:
        """Clean up resources (release writers, close handles, etc.)."""
        ...


@runtime_checkable
class ObservablePanel(ViserPanel, Protocol):
    """A panel that receives observations from the control loop.

    ``update()`` is always called from the main thread, synchronously,
    in panel registration order.
    """

    def update(self, obs: Any) -> None:
        """Process a new observation (update images, URDF, etc.)."""
        ...


class ViserApp:
    """Central viser server owner with panel registration.

    Parameters
    ----------
    port : int or None
        Port for the viser web server. None = viser default (8080).
    viser_server : ViserServer or None
        Use an existing server instead of creating one (e.g. from an agent).
    """

    def __init__(
        self,
        port: Optional[int] = None,
        viser_server: Optional[viser.ViserServer] = None,
    ) -> None:
        if viser_server is not None:
            self.server = viser_server
        elif port is not None:
            self.server = viser.ViserServer(port=port)
        else:
            self.server = viser.ViserServer()
        # Ordered dict (Python 3.7+) — panels update in registration order.
        # Register CameraPanel before RecordingPanel so RGB is extracted
        # before frames are written.
        self._panels: Dict[str, ViserPanel] = {}

    def add_panel(self, name: str, panel: ViserPanel) -> None:
        """Register and attach a panel to the viser server."""
        if name in self._panels:
            logger.warning("Replacing existing panel: {}", name)
            self._panels[name].detach()
        self._panels[name] = panel
        panel.attach(self.server)

    def get_panel(self, name: str) -> Optional[ViserPanel]:
        """Get a registered panel by name."""
        return self._panels.get(name)

    def update(self, obs: Any) -> None:
        """Fan out an observation to all ObservablePanel instances.

        Called from the main thread only. Panels update in registration
        order, which matters for data dependencies (e.g. CameraPanel
        extracts RGB before RecordingPanel writes frames).
        """
        for panel in self._panels.values():
            if isinstance(panel, ObservablePanel):
                panel.update(obs)

    def close(self) -> None:
        """Detach all panels and clean up."""
        for panel in self._panels.values():
            panel.detach()
        self._panels.clear()
