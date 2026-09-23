"""Desktop entry point — this is what the packaged `findme.exe` runs.

Starts the local findme server on a free port, opens the default browser to it, and
keeps running until the window is closed. No terminal knowledge required.
"""

from __future__ import annotations

import os
import socket
import threading
import time
import webbrowser


def _free_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def main() -> None:
    # Use the model bundled with the frozen build (offline, no download).
    from app import config

    config.setup_model_home()

    port = int(os.environ.get("FINDME_PORT") or _free_port())
    url = f"http://127.0.0.1:{port}"

    def _open_browser() -> None:
        # Wait until the server actually accepts connections before opening the browser.
        # The first launch can be slow (the OS scans the large bundle), so poll patiently.
        deadline = time.time() + 240
        while time.time() < deadline:
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=1):
                    break
            except OSError:
                time.sleep(0.5)
        webbrowser.open(url)

    if not os.environ.get("FINDME_NO_BROWSER"):
        threading.Thread(target=_open_browser, daemon=True).start()

    bar = "=" * 56
    print(bar)
    print("  findme is starting… (the first launch can take a minute)")
    print(f"  Your browser will open automatically at:  {url}")
    print("  Keep this window open while you use findme.")
    print(bar, flush=True)

    import uvicorn

    from app.main import app

    # Use the pure-Python asyncio loop + h11 (not uvloop/httptools): uvloop's compiled
    # event loop stalls inside a PyInstaller-frozen build.
    try:
        uvicorn.run(app, host="127.0.0.1", port=port, log_level="info", loop="asyncio", http="h11")
    except Exception:
        import traceback

        traceback.print_exc()
        input("\nfindme failed to start. Press Enter to close this window.")


if __name__ == "__main__":
    main()
