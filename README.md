## Reproduction code and data (Zenodo archive)

This repository contains:
- `figures/`: scripts (`.py`), datasets (`.npy`) to reproduce the figures (`.pdf`) in the paper.
- `demo/`: a minimal demo which generates the information lattice for a given many-body state (separate from figure reproduction).
- `utils/plotting.py`: shared plotting utilities used by both.

## Setup

The code can be run using standard Python libraries.
Requires `Python >= 3.9`. Scripts should be run from the root folder. Example:
```bash
python figures/fig07.py
```

For successful imports and dependencies it is necessary to run
```bash
python -m pip install -e .
```

Optionally create a virtual environment with

```bash
python -m venv .venv

# macOS/Linux
. .venv/bin/activate

# Windows (PowerShell)
.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
```

or run in an existing environment (e.g., Conda). 