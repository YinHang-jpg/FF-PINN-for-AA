# ENVIRONMENT.md — Runtime Environment and Installation

This document specifies the environment under which the
FF-PINN code accompanying the manuscript was tested, and gives
step-by-step installation instructions.

---

## 1. Python version

- **Recommended:** Python **3.10** (the version used for the
  manuscript benchmarks).
- **Tested:** 3.9, 3.10, 3.11. 3.12 should also work but is not
  exercised in CI.

Both Windows 10 and Ubuntu 22.04 are supported. macOS works for
CPU runs but has not been benchmarked.

---

## 2. Direct dependencies

The minimal scientific stack is captured by
`other_files/requirements.txt`:

```text
numpy>=1.20.0
matplotlib>=3.3.0
numba>=0.56.0
joblib>=1.1.0
psutil>=5.8.0
pytest>=6.0.0
pytest-benchmark>=3.4.0
```

Deep-learning, ODE / interpolation, and progress-bar dependencies
are installed separately so that the user can pick the right
PyTorch CUDA build for their machine:

```text
torch>=1.13         # CPU or CUDA build (see https://pytorch.org)
scipy>=1.9
tqdm>=4.64
pandas>=1.3
```

The `review/examples/run_example.py` demo requires
`torch`, `numpy`, `scipy`, `pandas`, and `matplotlib`.

---

## 3. Installation steps

### 3.1 Quick install (CPU only)

```bash
git clone <this-repository-url> FF-PINN
cd FF-PINN

python -m venv .venv
# Windows:  .venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

pip install --upgrade pip
pip install -r other_files/requirements.txt
pip install torch scipy tqdm
```

### 3.2 GPU install (CUDA 11.8 example)

```bash
pip install --upgrade pip
pip install -r other_files/requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cu118
pip install scipy tqdm
```

For CUDA 12.x replace `cu118` with `cu121`. Check
[pytorch.org](https://pytorch.org) for the exact wheel matching
your driver.

### 3.3 PYTHONPATH

All drivers in `other_files/`, `results/`, and `review/examples/`
import from the project sub-packages `mechanisms.*` and
`initialization.*`. Run them with the **repository root** as the
working directory or export `PYTHONPATH`:

```bash
# Linux / macOS
export PYTHONPATH=$(pwd)

# Windows (cmd)
set PYTHONPATH=%CD%

# Windows (PowerShell)
$env:PYTHONPATH = (Get-Location).Path
```

### 3.4 Verify the installation

```bash
python review/examples/run_example.py
```

Expected output is described in
[`examples/expected_output.txt`](examples/expected_output.txt).

---

## 4. `environment.yml` (Conda alternative)

If you prefer Conda, the following self-contained spec reproduces
the reference environment (CPU build):

```yaml
name: ffpinn
channels:
  - pytorch
  - conda-forge
  - defaults
dependencies:
  - python=3.10
  - numpy>=1.20
  - scipy>=1.9
  - matplotlib>=3.3
  - numba>=0.56
  - joblib>=1.1
  - psutil>=5.8
  - tqdm>=4.64
  - pytorch>=1.13
  - cpuonly
  - pip
  - pip:
      - pytest>=6.0.0
      - pytest-benchmark>=3.4.0
```

For a GPU build, replace `cpuonly` with the appropriate
`pytorch-cuda=<version>` package.

---

## 5. Optional: COMSOL / FEM cross-check

The COMSOL comparison shown in `validation/` requires COMSOL
Multiphysics 6.x with the Acoustics Module. **No COMSOL is needed**
to run the PINN code itself or the demo in
`review/examples/run_example.py`.

---

## 6. Known platform notes

- On Windows, several scripts call
  `sys.stdout.reconfigure(encoding="utf-8")` so that diagnostic
  prints render correctly.
- The repository is UTF-8 throughout. Configure your terminal
  accordingly (PowerShell:
  `[Console]::OutputEncoding = [System.Text.Encoding]::UTF8`).
- If `torch.compile` is unavailable on your platform (older Triton),
  the drivers silently fall back to eager mode; this is the default
  setting checked into `PINN_main.py`.
