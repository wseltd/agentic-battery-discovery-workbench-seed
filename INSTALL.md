# Installation Guide

Complete setup for the Agentic Molecular and Materials Discovery Workbench,
including all external models, data, and tools needed for end-to-end operation.

## Prerequisites

- **Python 3.10+** (3.13 tested)
- **NVIDIA GPU** with CUDA-capable driver (CUDA 12.x+)
- **conda** (Miniconda or Anaconda) for xTB installation
- **git-lfs** (`sudo apt install git-lfs && git lfs install`)
- ~10 GB free disk for models, data, and dependencies

## Quick Start

```bash
git clone <this-repo>
cd Agentic-Molecular-and-Materials-Discovery-Workbench
./scripts/setup.sh          # installs everything
source .venv/bin/activate
```

Or follow the manual steps below.

---

## Step 1: Core Python Environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
```

## Step 2: PyTorch with CUDA

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

Verify:
```bash
python -c "import torch; print(torch.cuda.get_device_name(0))"
```

## Step 3: MatterSim (ML Force Field)

```bash
pip install mattersim
```

Model weights download automatically on first use (~50 MB).

Verify:
```bash
python -c "from mattersim.forcefield import MatterSimCalculator; print('MatterSim OK')"
```

## Step 4: REINVENT 4 (Molecular Generator)

REINVENT pins `torch==2.9.1` in its dependencies. Since we need a newer PyTorch
for GPU compatibility and MatterSim, we install REINVENT without its dependency
pins and let our existing stack satisfy the requirements. This is safe because
REINVENT uses standard PyTorch APIs that are stable across versions.

```bash
# Clone REINVENT 4
git clone --depth 1 https://github.com/MolecularAI/REINVENT4.git ../REINVENT4

# Install without dependency pins
pip install --no-deps -e ../REINVENT4

# Install REINVENT's other dependencies (skip torch, it's already installed)
pip install \
  "chemprop>=1.5.2" "descriptastorus>=2.6.1" "mmpdb>=2.1,<3" \
  "molvs>=0.1.1" "pumas>=1.3.0" "tensorboard>=2,<3" "tomli>=2.0" \
  "apted>=1.0.3" "funcy>=2,<3" "polars<2" "tenacity>=8.2,<9" \
  "requests_mock>=1.10,<2"
```

### Download model priors

```bash
mkdir -p ../REINVENT4/priors
wget -O ../REINVENT4/priors/reinvent.prior \
  "https://zenodo.org/records/15641297/files/reinvent.prior?download=1"
```

Additional priors (optional, ~80 MB each):
```bash
for prior in libinvent linkinvent mol2mol_similarity; do
  wget -O ../REINVENT4/priors/${prior}.prior \
    "https://zenodo.org/records/15641297/files/${prior}.prior?download=1"
done
```

Verify:
```bash
reinvent --help
```

## Step 5: MatterGen (Crystal Generator)

MatterGen requires `torch==2.2.1+cu118` and `numpy<2`, which conflict with our
stack. It runs in a **separate Python environment** and is called via its CLI.
This is the recommended approach from Microsoft's own documentation.

```bash
# Clone MatterGen
cd .. && git clone https://github.com/microsoft/mattergen.git mattergen-upstream

# Create isolated Python 3.10 environment
cd mattergen-upstream
uv venv .venv --python 3.10    # or: python3.10 -m venv .venv
source .venv/bin/activate
pip install -e .

# Upgrade PyTorch for modern GPUs (Ampere, Ada, Blackwell)
# MatterGen's pinned torch 2.2+cu118 lacks kernels for newer GPUs
pip install --force-reinstall torch==2.11.0+cu128 torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu128
pip install --force-reinstall --no-deps torch_scatter torch_sparse torch_cluster \
  -f https://data.pyg.org/whl/torch-2.11.0+cu128.html

# Downgrade numpy (MatterGen's GemNet uses removed numpy.math)
pip install "numpy<2"

deactivate
cd ../Agentic-Molecular-and-Materials-Discovery-Workbench
```

Model weights download automatically from HuggingFace on first generation.

Verify:
```bash
../mattergen-upstream/.venv/bin/mattergen-generate /tmp/mattergen_verify \
  --pretrained_name=mattergen_base --batch_size=1 --num_batches=1
```

## Step 6: xTB (Semi-Empirical QC)

xTB requires Fortran libraries not available via pip. Install via conda:

```bash
conda create -n xtb_env -c conda-forge xtb -y
```

Verify:
```bash
$(conda info --envs | grep xtb_env | awk '{print $NF}')/bin/xtb --version
```

## Step 7: ChEMBL Reference Database

The novelty checker needs a pre-built fingerprint database from ChEMBL (~2.4M compounds).

```bash
# Download ChEMBL compound data (~230 MB compressed)
mkdir -p data
wget -O data/chembl_35_chemreps.txt.gz \
  "https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/releases/chembl_35/chembl_35_chemreps.txt.gz"
gunzip data/chembl_35_chemreps.txt.gz

# Build reference pickle (~15 min, produces ~260 MB file)
python scripts/build_chembl_reference.py
```

## Step 8: Materials Project API Key

Required for energy-above-hull calculations (convex hull stability screening).

1. Register at [materialsproject.org](https://materialsproject.org)
2. Go to Dashboard → API → copy your key
3. Set the environment variable:

```bash
export MP_API_KEY="your_key_here"
# Add to ~/.bashrc for persistence
```

## Step 9: Environment Configuration

Create a `.env` file (or copy from `.env.example`):

```bash
cp .env.example .env
# Edit .env with your paths
```

Required variables:
```
REINVENT_PRIOR_PATH=/path/to/REINVENT4/priors/reinvent.prior
MATTERGEN_BIN=/path/to/mattergen-upstream/.venv/bin/mattergen-generate
XTB_BIN=/path/to/miniconda3/envs/xtb_env/bin/xtb
CHEMBL_REFERENCE_PATH=data/chembl_reference.pkl
MP_API_KEY=your_materials_project_api_key
```

## Step 10: Verify Full Installation

```bash
python scripts/verify_install.py
```

This checks all components: PyTorch+CUDA, RDKit, REINVENT, MatterGen,
MatterSim, xTB, ChEMBL reference, and Materials Project API.

---

## Dependency Architecture

```
Main venv (.venv) — Python 3.13, PyTorch 2.11+cu128
├── REINVENT 4         (--no-deps install, shares main torch)
├── MatterSim 1.2.2   (pip install, uses main torch)
├── RDKit              (pip install)
├── pymatgen           (pip install)
└── mp-api             (pip install)

Separate venv (mattergen-upstream/.venv) — Python 3.10
└── MatterGen          (torch 2.11+cu128, numpy<2)
    Called via CLI: mattergen-generate

Conda env (xtb_env)
└── xTB                (Fortran binary)
    Called via subprocess
```

MatterGen needs a separate environment because it pins `numpy<2` and
`pytorch-lightning==2.0.6`, which conflict with MatterSim and the modern
scientific Python stack. The workbench calls MatterGen through its CLI
(`mattergen-generate`) and reads the output CIF files with pymatgen.

xTB is a Fortran binary that doesn't integrate with pip. The workbench
generates input files (XYZ geometry + run script) and calls xTB via subprocess.

## Troubleshooting

**MatterGen: "no kernel image is available for execution on the device"**
Your GPU is too new for the bundled PyTorch. Follow Step 5 to upgrade
PyTorch and PyG extensions in the MatterGen venv.

**MatterGen: "numpy.math" AttributeError**
numpy 2.x removed `numpy.math`. Downgrade numpy in the MatterGen venv:
`pip install "numpy<2"`

**xTB: "xtb-python not found"**
xTB cannot be pip-installed. Use conda (Step 6).

**ChEMBL build: MorganGenerator deprecation warnings**
Harmless. RDKit deprecated the old fingerprint API but it still works.

**REINVENT: torch version mismatch warning**
Expected. REINVENT pins torch 2.9.1 but works correctly with 2.11.
