"""Verify that all workbench dependencies are installed and functional.

Checks each component and reports pass/fail with actionable error messages.

Usage:
    python scripts/verify_install.py
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
RESET = "\033[0m"

results: list[tuple[str, bool, str]] = []


def check(name: str, passed: bool, detail: str = ""):
    status = f"{GREEN}PASS{RESET}" if passed else f"{RED}FAIL{RESET}"
    print(f"  [{status}] {name}" + (f" — {detail}" if detail else ""))
    results.append((name, passed, detail))


def main():
    print("Verifying workbench installation...\n")

    # 1. PyTorch + CUDA
    try:
        import torch

        has_cuda = torch.cuda.is_available()
        gpu_name = torch.cuda.get_device_name(0) if has_cuda else "N/A"
        check("PyTorch", True, f"{torch.__version__}, CUDA={has_cuda}, GPU={gpu_name}")
        if not has_cuda:
            check("CUDA GPU", False, "No CUDA device found")
    except ImportError:
        check("PyTorch", False, "pip install torch --index-url https://download.pytorch.org/whl/cu128")

    # 2. RDKit
    try:
        from rdkit import Chem

        mol = Chem.MolFromSmiles("C")
        check("RDKit", mol is not None, f"rdkit {Chem.rdBase.rdkitVersion}")
    except ImportError:
        check("RDKit", False, "pip install rdkit")

    # 3. pymatgen
    try:
        from pymatgen.core import Structure

        check("pymatgen", True)
    except ImportError:
        check("pymatgen", False, "pip install pymatgen")

    # 4. REINVENT 4
    reinvent_bin = shutil.which("reinvent")
    if reinvent_bin:
        try:
            import reinvent.version

            check("REINVENT 4", True, f"v{reinvent.version.__version__} at {reinvent_bin}")
        except ImportError:
            check("REINVENT 4", True, f"CLI at {reinvent_bin}")
    else:
        check("REINVENT 4", False, "See INSTALL.md Step 4")

    # 5. REINVENT priors
    prior_path = os.environ.get("REINVENT_PRIOR_PATH", "")
    if not prior_path:
        # Try default location
        candidates = [
            REPO_ROOT.parent / "REINVENT4" / "priors" / "reinvent.prior",
        ]
        for c in candidates:
            if c.exists():
                prior_path = str(c)
                break
    if prior_path and Path(prior_path).exists():
        size_mb = Path(prior_path).stat().st_size / 1e6
        check("REINVENT prior", True, f"{prior_path} ({size_mb:.0f} MB)")
    else:
        check("REINVENT prior", False, "Download from Zenodo — see INSTALL.md Step 4")

    # 6. MatterSim
    try:
        from mattersim.forcefield import MatterSimCalculator

        check("MatterSim", True)
    except ImportError:
        check("MatterSim", False, "pip install mattersim")

    # 7. MatterGen
    mattergen_bin = os.environ.get(
        "MATTERGEN_BIN",
        str(REPO_ROOT.parent / "mattergen-upstream" / ".venv" / "bin" / "mattergen-generate"),
    )
    if Path(mattergen_bin).exists():
        check("MatterGen", True, mattergen_bin)
    else:
        check("MatterGen", False, "See INSTALL.md Step 5 (separate venv required)")

    # 8. xTB
    xtb_bin = os.environ.get("XTB_BIN", "")
    if not xtb_bin:
        # Try to find it in conda envs
        try:
            result = subprocess.run(
                ["conda", "info", "--envs"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            for line in result.stdout.splitlines():
                if "xtb_env" in line:
                    env_path = line.split()[-1]
                    candidate = Path(env_path) / "bin" / "xtb"
                    if candidate.exists():
                        xtb_bin = str(candidate)
        except (FileNotFoundError, subprocess.TimeoutExpired):
                pass
    if xtb_bin and Path(xtb_bin).exists():
        check("xTB", True, xtb_bin)
    else:
        check("xTB", False, "conda create -n xtb_env -c conda-forge xtb -y")

    # 9. ChEMBL reference
    chembl_path = os.environ.get(
        "CHEMBL_REFERENCE_PATH", str(REPO_ROOT / "data" / "chembl_reference.pkl")
    )
    if Path(chembl_path).exists():
        size_mb = Path(chembl_path).stat().st_size / 1e6
        check("ChEMBL reference", True, f"{chembl_path} ({size_mb:.0f} MB)")
    else:
        check("ChEMBL reference", False, "python scripts/build_chembl_reference.py — see INSTALL.md Step 7")

    # 10. Materials Project API
    mp_key = os.environ.get("MP_API_KEY", "")
    if mp_key:
        check("Materials Project API", True, "MP_API_KEY is set")
    else:
        check(
            "Materials Project API",
            False,
            f"{YELLOW}Optional{RESET} — export MP_API_KEY=... (register at materialsproject.org)",
        )

    # Summary
    passed = sum(1 for _, p, _ in results if p)
    total = len(results)
    failed = total - passed
    print(f"\n{'=' * 50}")
    if failed == 0:
        print(f"{GREEN}All {total} checks passed.{RESET} Workbench is ready.")
    else:
        print(f"{passed}/{total} passed, {RED}{failed} failed{RESET}.")
        print("See INSTALL.md for setup instructions.")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
