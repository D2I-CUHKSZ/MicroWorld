"""Parallel simulation CLI entrypoint.

This keeps compatibility with the existing script while providing a
standardized package-level command entry.
"""

import asyncio
import sys

from lightworld.simulation.parallel_simulation_main import main as _async_main
from lightworld.simulation.parallel_simulation_main import setup_signal_handlers


def main() -> int:
    setup_signal_handlers()
    try:
        asyncio.run(_async_main())
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except SystemExit:
        pass
    finally:
        try:
            from multiprocessing import resource_tracker

            resource_tracker._resource_tracker._stop()
        except Exception:
            pass
        print("Simulation process exited")
    return 0


if __name__ == "__main__":
    sys.exit(main())
