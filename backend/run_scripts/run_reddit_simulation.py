
import asyncio
import sys

from run_parallel_simulation import main as parallel_main


def main():
    if "--reddit-only" not in sys.argv and "--twitter-only" not in sys.argv:
        sys.argv.insert(1, "--reddit-only")
    return asyncio.run(parallel_main())


if __name__ == "__main__":
    sys.exit(main())
