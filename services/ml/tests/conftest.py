from __future__ import annotations

import sys
import types


if "yaml" not in sys.modules:
    sys.modules["yaml"] = types.SimpleNamespace(safe_load=lambda *_args, **_kwargs: {})
