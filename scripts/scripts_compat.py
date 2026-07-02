from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Sequence


def run_encode_text_roberta(argv: Sequence[str]) -> int:
    print(
        "scripts/bert.py 已废弃；正式入口是 scripts/10_encode_text_roberta.py。"
        "本兼容入口会转发参数，但请在实验记录中使用正式脚本名。",
        flush=True,
    )
    script = Path(__file__).resolve().parent / "10_encode_text_roberta.py"
    spec = importlib.util.spec_from_file_location("encode_text_roberta", script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {script}")
    module = importlib.util.module_from_spec(spec)
    old_argv = sys.argv
    try:
        sys.argv = [str(script), *argv]
        spec.loader.exec_module(module)
        return int(module.main())
    finally:
        sys.argv = old_argv
