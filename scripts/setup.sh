#!/usr/bin/env bash
# Full installation script for the Agentic Molecular and Materials Discovery Workbench.
# See INSTALL.md for manual steps and troubleshooting.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PARENT_DIR="$(dirname "$REPO_DIR")"

GREEN='\033[92m'
RED='\033[91m'
YELLOW='\033[93m'
RESET='\033[0m'

info()  { echo -e "${GREEN}[INFO]${RESET}  $*"; }
warn()  { echo -e "${YELLOW}[WARN]${RESET}  $*"; }
fail()  { echo -e "${RED}[FAIL]${RESET}  $*"; exit 1; }

# -------------------------------------------------------------------
# Pre-flight checks
# -------------------------------------------------------------------
info "Checking prerequisites..."

command -v python3 >/dev/null 2>&1 || fail "python3 not found"
command -v conda   >/dev/null 2>&1 || warn "conda not found — xTB installation will be skipped"
command -v git     >/dev/null 2>&1 || fail "git not found"
command -v wget    >/dev/null 2>&1 || fail "wget not found (apt install wget)"

python3 -c "import sys; assert sys.version_info >= (3, 10)" 2>/dev/null \
  || fail "Python 3.10+ required"

nvidia-smi >/dev/null 2>&1 || warn "No NVIDIA GPU detected — GPU-dependent steps may fail"

# -------------------------------------------------------------------
# Step 1: Core venv
# -------------------------------------------------------------------
info "Step 1/9: Setting up Python virtual environment..."
cd "$REPO_DIR"
if [ ! -d .venv ]; then
    python3 -m venv .venv
fi
source .venv/bin/activate
pip install --quiet -e '.[dev]'

# -------------------------------------------------------------------
# Step 2: PyTorch + CUDA
# -------------------------------------------------------------------
info "Step 2/9: Installing PyTorch with CUDA 12.8..."
if python -c "import torch" 2>/dev/null; then
    info "  PyTorch already installed, skipping"
else
    pip install torch torchvision torchaudio \
      --index-url https://download.pytorch.org/whl/cu128
fi

# -------------------------------------------------------------------
# Step 3: MatterSim
# -------------------------------------------------------------------
info "Step 3/9: Installing MatterSim..."
if python -c "from mattersim.forcefield import MatterSimCalculator" 2>/dev/null; then
    info "  MatterSim already installed, skipping"
else
    pip install mattersim
fi

# -------------------------------------------------------------------
# Step 4: REINVENT 4
# -------------------------------------------------------------------
info "Step 4/9: Installing REINVENT 4..."
REINVENT_DIR="$PARENT_DIR/REINVENT4"

if [ ! -d "$REINVENT_DIR" ]; then
    git clone --depth 1 https://github.com/MolecularAI/REINVENT4.git "$REINVENT_DIR"
fi

if ! command -v reinvent >/dev/null 2>&1; then
    pip install --no-deps -e "$REINVENT_DIR"
    pip install --quiet \
      "chemprop>=1.5.2" "descriptastorus>=2.6.1" "mmpdb>=2.1,<3" \
      "molvs>=0.1.1" "pumas>=1.3.0" "tensorboard>=2,<3" "tomli>=2.0" \
      "apted>=1.0.3" "funcy>=2,<3" "polars<2" "tenacity>=8.2,<9" \
      "requests_mock>=1.10,<2"
fi

# Download prior
info "  Downloading REINVENT prior model..."
mkdir -p "$REINVENT_DIR/priors"
if [ ! -f "$REINVENT_DIR/priors/reinvent.prior" ]; then
    wget -q -O "$REINVENT_DIR/priors/reinvent.prior" \
      "https://zenodo.org/records/15641297/files/reinvent.prior?download=1"
fi
info "  Prior at $REINVENT_DIR/priors/reinvent.prior"

# -------------------------------------------------------------------
# Step 5: MatterGen (separate venv)
# -------------------------------------------------------------------
info "Step 5/9: Setting up MatterGen (separate environment)..."
MATTERGEN_DIR="$PARENT_DIR/mattergen-upstream"

if [ ! -d "$MATTERGEN_DIR" ]; then
    git clone https://github.com/microsoft/mattergen.git "$MATTERGEN_DIR"
fi

if [ ! -f "$MATTERGEN_DIR/.venv/bin/mattergen-generate" ]; then
    cd "$MATTERGEN_DIR"

    # Need Python 3.10 for MatterGen
    if command -v python3.10 >/dev/null 2>&1; then
        python3.10 -m venv .venv
    elif command -v uv >/dev/null 2>&1; then
        uv venv .venv --python 3.10
    else
        warn "Python 3.10 not found — install it or use: uv venv .venv --python 3.10"
        warn "Skipping MatterGen setup"
        cd "$REPO_DIR"
        SKIP_MATTERGEN=1
    fi

    if [ -z "${SKIP_MATTERGEN:-}" ]; then
        .venv/bin/pip install --quiet -e .

        # Upgrade PyTorch for modern GPUs
        .venv/bin/pip install --quiet --force-reinstall \
          torch==2.11.0+cu128 torchvision torchaudio \
          --index-url https://download.pytorch.org/whl/cu128
        .venv/bin/pip install --quiet --force-reinstall --no-deps \
          torch_scatter torch_sparse torch_cluster \
          -f https://data.pyg.org/whl/torch-2.11.0+cu128.html
        .venv/bin/pip install --quiet "numpy<2"
    fi
    cd "$REPO_DIR"
fi
info "  MatterGen CLI at $MATTERGEN_DIR/.venv/bin/mattergen-generate"

# -------------------------------------------------------------------
# Step 6: xTB
# -------------------------------------------------------------------
info "Step 6/9: Installing xTB..."
if command -v conda >/dev/null 2>&1; then
    XTB_ENV=$(conda info --envs 2>/dev/null | grep xtb_env | awk '{print $NF}' || true)
    if [ -z "$XTB_ENV" ]; then
        conda create -n xtb_env -c conda-forge xtb -y
        XTB_ENV=$(conda info --envs | grep xtb_env | awk '{print $NF}')
    fi
    info "  xTB at $XTB_ENV/bin/xtb"
else
    warn "  conda not available — skipping xTB. Install manually: conda create -n xtb_env -c conda-forge xtb -y"
fi

# -------------------------------------------------------------------
# Step 7: ChEMBL reference
# -------------------------------------------------------------------
info "Step 7/9: Building ChEMBL reference database..."
cd "$REPO_DIR"
mkdir -p data

if [ ! -f data/chembl_reference.pkl ]; then
    if [ ! -f data/chembl_35_chemreps.txt ]; then
        info "  Downloading ChEMBL data (~230 MB)..."
        wget -q -O data/chembl_35_chemreps.txt.gz \
          "https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/releases/chembl_35/chembl_35_chemreps.txt.gz"
        gunzip data/chembl_35_chemreps.txt.gz
    fi
    info "  Building fingerprint pickle (~15 min)..."
    python scripts/build_chembl_reference.py
else
    info "  ChEMBL reference already exists, skipping"
fi

# -------------------------------------------------------------------
# Step 8: .env file
# -------------------------------------------------------------------
info "Step 8/9: Creating .env configuration..."
if [ ! -f .env ]; then
    XTB_BIN="${XTB_ENV:-/path/to/conda}/bin/xtb"
    cat > .env << ENVEOF
REINVENT_PRIOR_PATH=$REINVENT_DIR/priors/reinvent.prior
MATTERGEN_BIN=$MATTERGEN_DIR/.venv/bin/mattergen-generate
XTB_BIN=$XTB_BIN
CHEMBL_REFERENCE_PATH=data/chembl_reference.pkl
MP_API_KEY=
ENVEOF
    warn "  Created .env — edit it to add your MP_API_KEY"
else
    info "  .env already exists, skipping"
fi

# -------------------------------------------------------------------
# Step 9: Verify
# -------------------------------------------------------------------
info "Step 9/9: Verifying installation..."
python scripts/verify_install.py

echo ""
info "Setup complete. Activate with: source .venv/bin/activate"
