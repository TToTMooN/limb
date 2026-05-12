"""Pedal-only DAgger phase trigger smoke test.

Stands up a DAggerEvents inbox + FootPedalPhaseTrigger, prints every
accepted phase transition.  No robots, no policy server, no agent.
Use this to verify your foot-pedal bindings and evdev permissions before
launching the full DAgger stack.

Press the LEFT pedal for ``pause_resume`` (AUTONOMOUS<->PAUSED).
Press the RIGHT pedal for ``correction``  (PAUSED<->CORRECTING).
Ctrl+C to exit.

Usage:
    uv run scripts/diagnostics/test_dagger_phase.py
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import tyro
from loguru import logger

from limb.agents.dagger.phase import DAggerEvents
from limb.agents.dagger.phase_trigger import FootPedalPhaseTrigger


@dataclass
class Args:
    device_path: str = "auto"
    vendor_id: int = 0x3553
    product_id: int = 0xB001
    pause_resume_key: str = "KEY_A"
    correction_key: str = "KEY_B"
    poll_hz: float = 100.0


def main(args: Args) -> None:
    logger.info(
        "DAgger phase trigger diagnostic — pedal {} -> pause_resume, pedal {} -> correction",
        args.pause_resume_key,
        args.correction_key,
    )

    events = DAggerEvents()
    trigger = FootPedalPhaseTrigger(
        device_path=args.device_path,
        vendor_id=args.vendor_id,
        product_id=args.product_id,
        pause_resume_key=args.pause_resume_key,
        correction_key=args.correction_key,
    )
    trigger.start(events)

    period = 1.0 / args.poll_hz
    last_phase = events.phase
    logger.info("Initial phase: {}", last_phase.value)
    logger.info("Press pedals; Ctrl+C to exit.")

    try:
        while True:
            transition = events.consume_transition()
            if transition is not None:
                old_phase, new_phase = transition
                logger.info(
                    "Phase transition: {} -> {}",
                    old_phase.value.upper(),
                    new_phase.value.upper(),
                )
                last_phase = new_phase
            else:
                # Periodic phase echo — silent if nothing changed.
                _ = last_phase
            time.sleep(period)
    except KeyboardInterrupt:
        logger.info("Interrupted, shutting down...")
    finally:
        trigger.close()


if __name__ == "__main__":
    main(tyro.cli(Args))
