#!/usr/bin/env python3
"""Patch robosuite 1.4's mass-matrix call for the MuJoCo 3 Python API."""

from importlib import metadata, util
from pathlib import Path


OLD = "            mujoco.mj_fullM(self.sim.model._model, mass_matrix, self.sim.data.qM)"
NEW = """            try:
                mujoco.mj_fullM(self.sim.model._model, mass_matrix, self.sim.data.qM)
            except TypeError:
                mj_data = getattr(self.sim.data, \"_data\", self.sim.data)
                mujoco.mj_fullM(self.sim.model._model, mj_data, mass_matrix)"""


def main():
    try:
        version = metadata.version("robosuite")
    except metadata.PackageNotFoundError:
        raise SystemExit("robosuite is not installed; run pip install -r requirements.txt") from None
    if version != "1.4.1":
        raise SystemExit(f"Expected robosuite 1.4.1, found {version}")

    spec = util.find_spec("robosuite")
    if spec is None or spec.origin is None:
        raise SystemExit("robosuite is not importable")
    target = Path(spec.origin).parent / "controllers" / "base_controller.py"
    source = target.read_text(encoding="utf-8")
    if NEW in source:
        print(f"Already patched: {target}")
        return
    if OLD not in source:
        raise SystemExit(f"Expected mj_fullM call not found in {target}")
    target.write_text(source.replace(OLD, NEW, 1), encoding="utf-8")
    print(f"Patched: {target}")


if __name__ == "__main__":
    main()
