# nmr_gnn_transformer

This repository implements a hybrid **GNN + Transformer encoder-decoder** model for predicting molecular structure (as SMILES) from NMR spectral data. It builds on and extends prior work in graph-based NMR modeling.

---

## Dataset

To run the code, you can use the .npz file in 'data' folder or else you’ll need to manually download the dataset:

- **Source:** [NMRShiftDB2](https://nmrshiftdb.nmr.uni-koeln.de/portal/js_pane/P-Help)
- **File Required:** `nmrshiftdb2withsignals.sd`

Once downloaded, place the file in a `data/` folder at the root of the repository.

---

## Model Overview

This project includes a custom hybrid model:
- A **Graph Neural Network (GNN)** encoder extracts features from molecular graphs.
- A **Transformer decoder** generates SMILES sequences from encoded representations.

The goal is to learn the mapping: **NMR spectra → molecular structure**.

---

## Acknowledgments

This project builds on code from:

**Han et al., 2022**  
*Scalable graph neural network for NMR chemical shift prediction*  
**Phys. Chem. Chem. Phys., 2022, 24, 26870–26878**  

```bibtex
@article{nmr_sgnn,
  title={Scalable graph neural network for {NMR} chemical shift prediction},
  author={Han, Jongmin and Kang, Hyungu and Kang, Seokho and Kwon, Youngchun and Lee, Dongseon and Choi, Youn-Suk},
  journal={Physical Chemistry Chemical Physics},
  volume={24},
  pages={26870--26878},
  year={2022},
  doi={10.1039/D2CP04542G}
}
