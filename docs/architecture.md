# Architecture

The application accepts molecular structures in SMILES or InChI format, validates and preprocesses them, extracts molecular features, and sends those features to a trained machine-learning model for prediction.

## Main components

1. **User interface** — molecule input, validation messages, and prediction results.
2. **Molecule parser** — parses and sanitizes SMILES/InChI strings.
3. **Feature generator** — calculates descriptors or molecular fingerprints.
4. **Prediction engine** — loads the trained model and generates predictions.
5. **Results view** — displays predicted properties, units, and warnings.

The UI and prediction engine should remain separate so that models can be updated without redesigning the interface.
