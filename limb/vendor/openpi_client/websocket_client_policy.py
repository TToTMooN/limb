"""Vendored from openpi-client (Apache 2.0). See limb/vendor/openpi_client/__init__.py.

Changes from upstream: imports are rewritten to use limb's vendored package
path; the ``typing_extensions.override`` decorator is dropped to avoid a tiny
extra dep; and ``infer()`` reconnects + retries once when the websocket has
died (upstream raises forever after a serve-side error — froze the 30 Hz
loop on-robot 2026-07-08).
"""

import logging
import time
from typing import Dict, Optional, Tuple

import websockets.sync.client

from limb.vendor.openpi_client import base_policy as _base_policy
from limb.vendor.openpi_client import msgpack_numpy


class WebsocketClientPolicy(_base_policy.BasePolicy):
    """Implements the Policy interface by communicating with a server over websocket.

    See WebsocketPolicyServer for a corresponding server implementation.
    """

    def __init__(self, host: str = "0.0.0.0", port: Optional[int] = None, api_key: Optional[str] = None) -> None:
        if host.startswith("ws"):
            self._uri = host
        else:
            self._uri = f"ws://{host}"
        if port is not None:
            self._uri += f":{port}"
        self._packer = msgpack_numpy.Packer()
        self._api_key = api_key
        self._ws, self._server_metadata = self._wait_for_server()

    def get_server_metadata(self) -> Dict:
        return self._server_metadata

    def _wait_for_server(self) -> Tuple[websockets.sync.client.ClientConnection, Dict]:
        logging.info(f"Waiting for server at {self._uri}...")
        while True:
            try:
                headers = {"Authorization": f"Api-Key {self._api_key}"} if self._api_key else None
                conn = websockets.sync.client.connect(
                    self._uri, compression=None, max_size=None, additional_headers=headers
                )
                metadata = msgpack_numpy.unpackb(conn.recv())
                return conn, metadata
            except ConnectionRefusedError:
                logging.info("Still waiting for server...")
                time.sleep(5)

    def infer(self, obs: Dict) -> Dict:
        data = self._packer.pack(obs)
        try:
            self._ws.send(data)
            response = self._ws.recv()
        except Exception as e:
            # RECONNECT-ON-FAILURE (local change, on-robot freeze 2026-07-08 18:08): a
            # serve-side error closes the websocket; upstream openpi-client then raises
            # on EVERY infer forever and the chunk buffer never refills. Reconnect
            # (blocks this thread until the serve answers) and retry the request once.
            logging.warning(f"policy-server websocket lost ({type(e).__name__}: {e}); reconnecting...")
            try:
                self._ws.close()
            except Exception:
                pass
            self._ws, self._server_metadata = self._wait_for_server()
            self._ws.send(data)
            response = self._ws.recv()
        if isinstance(response, str):
            # we're expecting bytes; if the server sends a string, it's an error.
            raise RuntimeError(f"Error in inference server:\n{response}")
        return msgpack_numpy.unpackb(response)

    def reset(self) -> None:
        pass
