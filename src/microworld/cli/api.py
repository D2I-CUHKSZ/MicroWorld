"""Backend API server CLI entrypoint."""

import os
import sys

from microworld import create_app
from microworld.config.settings import Config


def main() -> int:
    """Start Flask backend API service."""
    if sys.platform == "win32":
        os.environ.setdefault("PYTHONIOENCODING", "utf-8")
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    errors = Config.validate()
    if errors:
        print("Configuration errors:")
        for err in errors:
            print(f"  - {err}")
        print("\nPlease check settings in the .env file")
        return 1

    app = create_app()
    host = os.environ.get("FLASK_HOST", "0.0.0.0")
    port = int(os.environ.get("FLASK_PORT", 5001))
    debug = Config.DEBUG
    app.run(host=host, port=port, debug=debug, threaded=True)
    return 0
