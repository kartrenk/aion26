"""Fixtures for E2E webapp tests.

Provides a session-scoped webapp server subprocess and
per-test Playwright page fixtures.
"""

import subprocess
import sys
import time
from pathlib import Path

import pytest

try:
    import requests
except ImportError:
    requests = None

WEBAPP_SCRIPT = str(Path(__file__).parent.parent.parent / "scripts" / "train_webapp_pro.py")
WEBAPP_PORT = 5099
WEBAPP_URL = f"http://localhost:{WEBAPP_PORT}"
STARTUP_TIMEOUT = 60  # seconds


@pytest.fixture(scope="session")
def webapp_server():
    """Start the webapp as a subprocess and wait for /health to respond."""
    if requests is None:
        pytest.skip("requests not installed (install e2e extras)")

    proc = subprocess.Popen(
        [sys.executable, WEBAPP_SCRIPT, "serve", "--port", str(WEBAPP_PORT)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    # Poll /health until the server is ready
    deadline = time.time() + STARTUP_TIMEOUT
    while time.time() < deadline:
        try:
            resp = requests.get(f"{WEBAPP_URL}/health", timeout=2)
            if resp.status_code == 200:
                yield WEBAPP_URL
                proc.terminate()
                proc.wait(timeout=10)
                return
        except (requests.ConnectionError, requests.Timeout):
            pass

        # Check if process crashed
        if proc.poll() is not None:
            stderr = proc.stderr.read().decode(errors="replace")
            raise RuntimeError(
                f"Webapp process exited with code {proc.returncode} "
                f"before becoming ready.\nStderr:\n{stderr[:2000]}"
            )

        time.sleep(1)

    # Timed out
    proc.terminate()
    proc.wait(timeout=10)
    stderr = proc.stderr.read().decode(errors="replace")
    raise RuntimeError(
        f"Webapp did not respond on /health within {STARTUP_TIMEOUT}s.\n"
        f"Stderr:\n{stderr[:2000]}"
    )


@pytest.fixture
def base_url(webapp_server):
    """Return the webapp base URL for requests-based tests."""
    return webapp_server
