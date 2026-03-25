"""Loop status panel — minimal Hz and step count display in the viser sidebar."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import viser


@dataclass
class LoopStatusPanel:
    """Shows control loop Hz and step count in the viser GUI.

    Implements ViserPanel: attach(server), detach().
    Also provides update_loop(hz, step) for StatusDisplay compatibility.
    """

    _server: Optional[viser.ViserServer] = field(default=None, init=False, repr=False)
    _hz_handle: object = field(default=None, init=False, repr=False)

    def attach(self, server: viser.ViserServer) -> None:
        self._server = server
        self._hz_handle = server.gui.add_text("Loop", initial_value="-- Hz | step 0", disabled=True)

    def update_loop(self, hz: float, step: int) -> None:
        if self._hz_handle is not None:
            self._hz_handle.value = f"{hz:.1f} Hz | step {step}"

    # StatusDisplay compat stubs
    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

    def detach(self) -> None:
        pass
