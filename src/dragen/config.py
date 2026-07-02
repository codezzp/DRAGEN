"""Configuration helpers for experiment scripts."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

try:
    import yaml
except ImportError:  # pragma: no cover - depends on the runtime environment.
    yaml = None  # type: ignore[assignment]


def load_config(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    text = Path(path).read_text(encoding="utf-8")
    if text.lstrip().startswith("{"):
        data = json.loads(text)
    elif yaml is not None:
        data = yaml.safe_load(text) or {}
    else:
        data = parse_simple_yaml(text)
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return data


def apply_config(parser: argparse.ArgumentParser, args: argparse.Namespace, argv: Sequence[str] | None = None) -> argparse.Namespace:
    config = load_config(getattr(args, "config", None))
    if not config:
        return args
    cli_dests = cli_provided_dests(parser, argv if argv is not None else sys.argv[1:])
    values = flatten_config(config)
    for key, value in values.items():
        if value is None or key in cli_dests or not hasattr(args, key):
            continue
        setattr(args, key, value)
    setattr(args, "resolved_config", build_resolved_config(args, config))
    return args


def cli_provided_dests(parser: argparse.ArgumentParser, argv: Sequence[str]) -> set[str]:
    option_to_dest: dict[str, str] = {}
    for action in parser._actions:
        for option in action.option_strings:
            option_to_dest[option] = action.dest
    provided: set[str] = set()
    for token in argv:
        option = token.split("=", 1)[0]
        if option in option_to_dest:
            provided.add(option_to_dest[option])
    return provided


def flatten_config(config: Mapping[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in config.items():
        if key not in {"data", "model", "train", "loss", "output", "logging", "checkpoint", "tables", "analysis"}:
            out[key] = value
    merge_section(out, config.get("data"), {"pack_dir": "pack_dir", "input_variant": "input_variant", "artifact_dir": "artifact_dir"})
    merge_section(
        out,
        config.get("model"),
        {
            "hidden_dim": "hidden_dim",
            "role_num": "role_num",
            "top_k_global": "top_k_global",
            "dropout": "dropout",
            "use_global_prior": "use_global_prior",
            "use_adaptive_sampling": "use_adaptive_sampler",
            "use_adaptive_sampler": "use_adaptive_sampler",
            "use_memory": "use_memory",
            "use_temporal_memory": "use_memory",
            "use_gate": "use_gate",
            "use_prior_observation_gate": "use_gate",
            "use_uncertainty": "use_uncertainty",
            "use_role": "use_role",
        },
    )
    merge_section(
        out,
        config.get("train"),
        {
            "epochs": "epochs",
            "batch_size": "batch_size",
            "lr": "lr",
            "weight_decay": "weight_decay",
            "seed": "seed",
            "device": "device",
            "max_train_samples": "max_train_samples",
            "max_valid_samples": "max_valid_samples",
            "max_test_samples": "max_test_samples",
            "eval_every": "eval_every",
        },
    )
    merge_section(
        out,
        config.get("loss"),
        {
            "lambda_jump": "lambda_jump",
            "lambda_struct": "lambda_struct",
            "lambda_align": "lambda_align",
            "lambda_uncertainty": "lambda_uncertainty",
            "lambda_role": "lambda_role",
            "lambda_sampler_edge": "lambda_sampler_edge",
            "lambda_sampler_hub": "lambda_sampler_hub",
            "lambda_sampler_temp": "lambda_sampler_temp",
        },
    )
    merge_section(out, config.get("output"), {"out_dir": "out_dir", "main_out": "main_out", "risk_out": "risk_out", "ablation_out": "ablation_out"})
    merge_section(out, config.get("logging"), {"tensorboard": "tensorboard", "tb_log_dir": "tb_log_dir"})
    merge_section(out, config.get("checkpoint"), {"resume": "resume", "save_every_epoch": "save_every_epoch"})
    merge_section(
        out,
        config.get("tables"),
        {
            "run_dirs": "run_dirs",
            "out_dir": "out_dir",
            "ablation_run_dirs": "ablation_run_dirs",
            "full_run_dir": "full_run_dir",
        },
    )
    merge_section(out, config.get("analysis"), {"artifact_dir": "artifact_dir", "out_dir": "out_dir"})
    return out


def merge_section(out: dict[str, Any], section: Any, mapping: Mapping[str, str]) -> None:
    if not isinstance(section, Mapping):
        return
    for source, dest in mapping.items():
        if source in section:
            out[dest] = section[source]


def build_resolved_config(args: argparse.Namespace, source_config: Mapping[str, Any]) -> dict[str, Any]:
    data = {k: v for k, v in vars(args).items() if k != "resolved_config"}
    return {"source_config": dict(source_config), "resolved_args": data}


def write_run_metadata(out_dir: Path, args: argparse.Namespace) -> None:
    reports = out_dir / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    resolved = getattr(args, "resolved_config", {"resolved_args": vars(args)})
    write_yaml(reports / "resolved_config.yaml", resolved)
    (reports / "command.txt").write_text(" ".join(quote_arg(arg) for arg in sys.argv) + "\n", encoding="utf-8")
    with (reports / "git_info.json").open("w", encoding="utf-8") as f:
        json.dump(git_info(), f, ensure_ascii=False, indent=2)
        f.write("\n")


def write_yaml(path: Path, data: Mapping[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        if yaml is not None:
            yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)
        else:
            json.dump(data, f, ensure_ascii=False, indent=2)
            f.write("\n")


def quote_arg(arg: str) -> str:
    return json.dumps(arg) if any(ch.isspace() for ch in arg) else arg


def git_info() -> dict[str, Any]:
    return {
        "branch": run_git(["rev-parse", "--abbrev-ref", "HEAD"]),
        "commit": run_git(["rev-parse", "HEAD"]),
        "dirty": bool(run_git(["status", "--short"])),
    }


def run_git(args: list[str]) -> str:
    try:
        return subprocess.check_output(["git", *args], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return ""


def parse_simple_yaml(text: str) -> dict[str, Any]:
    lines = [line.rstrip() for line in text.splitlines() if line.strip() and not line.lstrip().startswith("#")]
    root: dict[str, Any] = {}
    stack: list[tuple[int, Any]] = [(-1, root)]
    for idx, line in enumerate(lines):
        indent = len(line) - len(line.lstrip(" "))
        item = line.strip()
        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        if item.startswith("- "):
            if not isinstance(parent, list):
                raise ValueError(f"Unsupported YAML list placement near: {line}")
            parent.append(parse_scalar(item[2:].strip()))
            continue
        key, sep, value = item.partition(":")
        if not sep:
            raise ValueError(f"Unsupported YAML line: {line}")
        key = key.strip()
        value = value.strip()
        if value:
            parent[key] = parse_scalar(value)
            continue
        container: Any = [] if next_significant_line_is_list(lines, idx, indent) else {}
        parent[key] = container
        stack.append((indent, container))
    return root


def next_significant_line_is_list(lines: Sequence[str], idx: int, indent: int) -> bool:
    for line in lines[idx + 1 :]:
        next_indent = len(line) - len(line.lstrip(" "))
        if next_indent <= indent:
            return False
        return line.strip().startswith("- ")
    return False


def parse_scalar(value: str) -> Any:
    if value in {"", "null", "None", "~"}:
        return None
    if value == "true":
        return True
    if value == "false":
        return False
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value.strip("'\"")
