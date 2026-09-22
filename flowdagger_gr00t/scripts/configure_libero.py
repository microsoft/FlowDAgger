#!/usr/bin/env python3
"""Create a noninteractive LIBERO config for a source checkout."""

import argparse
from pathlib import Path

import yaml


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("libero_root", help="Path to the cloned LIBERO repository")
    parser.add_argument(
        "--config-dir",
        default="~/.libero",
        help="Directory for config.yaml (default: ~/.libero)",
    )
    args = parser.parse_args()

    root = Path(args.libero_root).expanduser().resolve()
    package_root = root / "libero" / "libero"
    required = [package_root / "bddl_files", package_root / "init_files", package_root / "assets"]
    missing = [str(path) for path in required if not path.is_dir()]
    if missing:
        raise SystemExit(f"Not a valid LIBERO checkout; missing: {', '.join(missing)}")

    config_dir = Path(args.config_dir).expanduser().resolve()
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / "config.yaml"
    config = {
        "benchmark_root": str(package_root),
        "bddl_files": str(package_root / "bddl_files"),
        "init_states": str(package_root / "init_files"),
        "datasets": str(root / "libero" / "datasets"),
        "assets": str(package_root / "assets"),
    }
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    print(f"Wrote {config_path}")
    print(f"export LIBERO_CONFIG_PATH={config_dir}")
    print(f"export PYTHONPATH={root}:${{PYTHONPATH:-}}")


if __name__ == "__main__":
    main()
