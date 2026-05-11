# Vendored: openpi-client (subset)

This directory contains a minimal subset of [openpi-client](https://github.com/Physical-Intelligence/openpi/tree/main/packages/openpi-client),
vendored into limb so that talking to an OpenPI native policy server (pi0,
pi0.5, pi0-FAST) doesn't require a sibling openpi clone or an extra path
dependency.

## What's here

- `__init__.py` — re-export `WebsocketClientPolicy`
- `base_policy.py` — abstract `BasePolicy` (12 lines)
- `msgpack_numpy.py` — pickle-free msgpack codec for numpy arrays (57 lines)
- `websocket_client_policy.py` — the actual websocket client (~60 lines)

What is **not** vendored: `image_tools`, the `runtime/` package, tests. If
you need those, install upstream `openpi-client` and import from `openpi_client`
directly.

## Source

- Upstream: https://github.com/Physical-Intelligence/openpi
- Version at time of vendoring: openpi-client 0.1.0 (see upstream pyproject)
- Files copied: `src/openpi_client/{__init__,base_policy,msgpack_numpy,websocket_client_policy}.py`

## Local modifications

- Import paths rewritten from `openpi_client.X` to `limb.vendor.openpi_client.X`.
- Dropped the `typing_extensions.override` decorator on the two BasePolicy
  implementations — avoids a tiny extra dep; runtime behaviour unchanged.

Everything else is byte-for-byte identical to upstream.

## License

Apache License 2.0 — copy below from the openpi repository.
limb retains its own license; the Apache 2.0 license applies only to the
files in this directory.

```
Copyright 2024 Physical Intelligence

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```
