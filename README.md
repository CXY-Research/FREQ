# FREQ

Official implementation of **Mitigating pair-frequency bias and qualifier-primary relation mismatch in inductive hyper-relational knowledge graph completion**, published in *Neurocomputing* (2026).

📄 [Paper](https://doi.org/10.1016/j.neucom.2026.135254)

## Overview

FREQ extends MAYPL with two components without adding learnable parameters:

- **IDF:** frequency-conditioned pair encoding before pair normalization.
- **RCQW:** relation-conditioned weighting of normalized qualifier messages.

Experiments cover WD20K100v1, WD20K100v2, WP-IND, and MFB-IND.

## Installation

Requires Python 3.9, PyTorch 2.0.1, and a CUDA-capable GPU.

```bash
pip install -r requirements.txt
```

## Datasets

Download the preprocessed datasets from the [MAYPL repository](https://github.com/bdi-lab/maypl/tree/main/data) and place each dataset under `data/<dataset_name>/`.

## Usage

Train FREQ:

```bash
python run_experiment.py train --dataset WD20K100v1 --variant freq --seed 0
```

Evaluate a checkpoint, replacing `<epoch>` with its epoch number:

```bash
python run_experiment.py test --dataset WD20K100v1 --variant freq --seed 0 --epoch <epoch>
```

## Citation

If you use this code, please cite:

> Xinyu Chen, Nan Wu, Yi Yang, Jia Wang, Fangfang Han, Haoran Yu, and Yongming Cai. Mitigating pair-frequency bias and qualifier-primary relation mismatch in inductive hyper-relational knowledge graph completion. Neurocomputing, 2026. https://doi.org/10.1016/j.neucom.2026.135254

## Acknowledgments

This implementation is based on [MAYPL](https://github.com/bdi-lab/maypl), the official implementation of *Structure Is All You Need* (ICML 2025).

## License

This project follows the upstream **CC BY-NC-SA 4.0** license. See `LICENSE.txt`.
