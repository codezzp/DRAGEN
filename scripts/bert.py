from __future__ import annotations

import sys

import _bootstrap  # noqa: F401
from scripts_compat import run_encode_text_roberta


if __name__ == "__main__":
    raise SystemExit(run_encode_text_roberta(sys.argv[1:]))
