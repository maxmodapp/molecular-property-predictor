# Third-Party Notices

This project includes and depends on third-party software.

## pKaSolver

- Location: `external/pkasolver`
- License: MIT
- Copyright: Fritz Mayr, Marcus Wieder, Oliver Wieder, Thierry Langer
- Upstream project: pKaSolver

This repository includes a minimal runtime subset of pKaSolver needed for local pKa inference. The vendored `query.py` was modified to load this project's flat model layout from `models/pka`.

## Dimorphite-DL

- Location: `external/pkasolver/pkasolver/dimorphite_dl`
- License: see `external/pkasolver/pkasolver/dimorphite_dl/LICENSE.txt`

Dimorphite-DL is included because pKaSolver uses it for protonation state generation.

## Runtime Dependencies

The application also depends on packages installed in the Python environment, including RDKit, PyTorch, PyTorch Geometric, TensorFlow/Keras, PySide6, CairoSVG and SVGUtils. These packages are not vendored here.

## Models

The pKa checkpoints in `models/pka` are project-specific fine-tuned model files. The log P Keras model in `models/logp` is also project-specific.
