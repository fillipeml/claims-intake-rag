"""The ASGI entry point.

`uvicorn claims_rag.app:app` needs a module-level application, while `create_app` is a factory
so tests can hand it a corpus. This is the one line that joins them, and it reads the fixtures
path from the environment so the container can mount a different corpus without a rebuild.
"""

from __future__ import annotations

import os

from .api import create_app

app = create_app(fixtures=os.environ.get("CLAIMS_FIXTURES", "fixtures"))
