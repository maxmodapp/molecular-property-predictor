# Molecular Property Predictor

Desktop application for local molecular property prediction from SMILES, InChI, or structures drawn in an embedded molecular editor.

This project applies machine learning and deep learning techniques to molecular property prediction, combining neural network inference, graph-based molecular representations, cheminformatics preprocessing, pKa microstate generation, molecular visualization, and desktop software engineering into a single application.

It was built as a practical tool for predicting molecular log P and pKa values while keeping the full inference workflow local.

## What The App Does

- Accepts molecular input as SMILES, InChI, or a drawn structure.
- Standardizes and renders the input molecule with RDKit.
- Predicts log P using a trained Keras neural network model.
- Predicts microstate pKa values using a fine-tuned ensemble of graph neural network models.
- Shows individual protonation/deprotonation transitions with their predicted pKa ranges.
- Generates a final pKa map over the molecule.
- Provides a desktop UI built with PySide6.
- Can be packaged as a portable Windows executable with PyInstaller.

## Screenshots

Add screenshots under `docs/screenshots/`.

Recommended screenshots:

1. `docs/screenshots/main-input.png`

   Main app view after entering a SMILES or InChI. This should show the molecular input area, the rendered molecule, and the prediction panel.

2. `docs/screenshots/pka-results.png`

   pKa tab with the list of microstate pKa predictions and the protonated/deprotonated SMILES transitions.

3. `docs/screenshots/pka-map.png`

   Full-width pKa visual output, including the microstate reaction image and the final molecular pKa map.

4. `docs/screenshots/logp-results.png`

   log P tab showing the predicted log P value.

You can display them in GitHub like this:

```md
## Screenshots

![Main input](docs/screenshots/main-input.png)
![pKa results](docs/screenshots/pka-results.png)
![pKa map](docs/screenshots/pka-map.png)
![log P results](docs/screenshots/logp-results.png)
```

## Prediction Pipeline

### Molecular Input

The app supports three input workflows:

- Direct SMILES input.
- InChI input converted into a molecule.
- Structure drawing through an embedded molecular editor.

The resulting molecule is parsed with RDKit, converted to a consistent representation, and rendered in the interface before running the selected predictions.

### log P Prediction

The log P model is stored in:

```text
models/logp/model_logp.keras
models/logp/logp_norm.npz
```

The application loads the trained Keras model, applies the same normalization convention used during training, runs local inference, and converts the output back to the original log P scale.

### pKa Prediction

The pKa workflow uses graph-based molecular representations and an ensemble of fine-tuned neural network checkpoints. Each candidate microstate is evaluated by the model ensemble, and the app reports both the predicted pKa and an uncertainty range derived from the ensemble spread.

The pKa checkpoints are stored directly in:

```text
models/pka/
|-- fine_tuned_model_0.pt
|-- fine_tuned_model_1.pt
|-- ...
|-- fine_tuned_model_10.pt
`-- fine_tuned_best_model.pt
```

The pKa inference engine loads those checkpoints, evaluates the generated protonation/deprotonation states, and returns:

- The protonated SMILES.
- The deprotonated SMILES.
- The predicted pKa value.
- The prediction range.
- A reaction visualization for each microstate.
- A final molecular pKa map.

## Technical Highlights

- Machine learning workflow for molecular property prediction.
- Deep learning based inference with trained neural network models.
- Keras neural network inference for log P estimation.
- Graph neural network ensemble for microstate pKa prediction.
- Molecular graph featurization using atom and bond descriptors.
- Ensemble aggregation to estimate prediction spread and report pKa ranges.
- Local model loading and inference without external prediction APIs.
- RDKit-based molecule parsing, conversion, and rendering.
- Integrated molecular drawing workflow.
- Desktop GUI architecture with separate prediction workers to keep the UI responsive.
- Portable application build with PyInstaller.
- Repository structure prepared for GitHub and portfolio review.

## Repository Structure

```text
molecular-property-predictor/
|-- app/
|   |-- desktop_app_final.py        # Main PySide6 desktop app
|   |-- pka_prediction_engine.py    # App-owned pKa integration layer
|   |-- desktop_app_final.spec      # PyInstaller build spec
|   |-- build_exe.ps1               # Windows build script
|   |-- jsme_editor_embed_fixed.html
|   |-- app_icon.ico
|   `-- app_icon.png
|-- models/
|   |-- logp/
|   |   |-- model_logp.keras
|   |   `-- logp_norm.npz
|   `-- pka/
|       |-- fine_tuned_model_0.pt
|       |-- ...
|       |-- fine_tuned_model_10.pt
|       `-- fine_tuned_best_model.pt
|-- external/
|   `-- pkasolver/                  # Minimal runtime components used for pKa inference
|-- docs/
|   `-- screenshots/
|-- requirements.txt
|-- environment.yml
|-- THIRD_PARTY_NOTICES.md
|-- LICENSE
`-- README.md
```

## Tech Stack

- Python 3.9
- PySide6
- RDKit
- TensorFlow / Keras
- PyTorch
- PyTorch Geometric
- CairoSVG
- SVGUtils
- PyInstaller

## Run From Source

Create and activate a Python 3.9 environment with the required dependencies installed. Then run:

```powershell
cd app
python desktop_app_final.py
```

To validate the pKa prediction engine without launching the UI:

```powershell
cd app
python desktop_app_final.py --smoke-test-pkasolver
```

Expected output:

```text
pKaSolver smoke test OK: 1 microestado(s).
```

## Build A Windows Executable

From an environment with the dependencies installed:

```powershell
cd app
.\build_exe.ps1 -OutputRoot D:\PPS_PrediccionPropiedades -CreateZip
```

The build script uses the PyInstaller spec in `app/desktop_app_final.spec` and includes the app assets, local models, and runtime files needed for prediction.

## Model Files And Git LFS

The model files are large binary artifacts. This repository includes `.gitattributes` rules for Git LFS:

```text
*.pt filter=lfs diff=lfs merge=lfs -text
*.keras filter=lfs diff=lfs merge=lfs -text
*.npz filter=lfs diff=lfs merge=lfs -text
```

Before pushing the repository:

```powershell
git lfs install
git add .
git commit -m "Initial molecular property predictor app"
git push
```

## Third-Party Components

The pKa workflow uses selected runtime components from pKaSolver and Dimorphite-DL for molecular microstate handling and pKa-related utilities. The application-level integration, model layout, UI, prediction presentation, packaging, and local inference workflow are organized in this repository.

See `THIRD_PARTY_NOTICES.md` for dependency attribution and license notes.

## Project Status

This is a portfolio-ready desktop inference application built from a molecular property prediction workflow. Current predictions include log P and microstate pKa. The structure is prepared for future expansion to additional molecular properties.
