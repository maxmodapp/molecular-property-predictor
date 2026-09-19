# Datasets

Datasets should contain molecular structures and the target properties used for training.

## Recommended fields

- molecule identifier
- SMILES
- InChI or InChIKey, when available
- target property
- units
- data source
- quality or provenance metadata

## Preparation

Before training, remove duplicates, validate structures, standardize molecular representations, handle salts and stereochemistry consistently, check units, and prevent data leakage between train and test partitions.

Dataset snapshots and transformations should be versioned so that experiments can be reproduced.
