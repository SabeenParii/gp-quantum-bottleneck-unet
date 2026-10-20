# gp-quantum-bottleneck-unet

Source code for the study **"Global versus Patch-Wise Quantum Bottlenecks in U-Net Mammographic Mass Segmentation: Matched Classical Controls, Severing Tests, and External Validation."**

A 4-qubit, 2-layer parameterized quantum circuit is placed in the bottleneck slot of a U-Net in two ways:

- **global (G):** one 4-dimensional code per image (global average pooling), and
- **patch-wise (P):** one 4-dimensional code per cell of a 4 × 4 grid, with shared weights.

At each placement the circuit (Q) is compared with an equal-parameter classical layer, Dense(4, tanh) (Cl), and with an identity mapping (Id). Two reference models are trained as well: a hybrid U-Net with an uncompressed bottleneck (`full`) and a 31M-parameter classical U-Net (`unet`).

Models are trained on CBIS-DDSM mass ROIs with patient-level 5-fold cross-validation and three seeds. They are tested without fine-tuning on DMID mass crops. Severing tests check whether each network uses its bottleneck code.

## Data

Only public, de-identified datasets are used, and no images are included in this repository.

- **CBIS-DDSM** is available from The Cancer Imaging Archive. Notebook 01 downloads it through the public TCIA API.
- **DMID** is the Digital Mammography Dataset for Breast Cancer Diagnosis Research, available from its authors' figshare record. Notebooks 03a and 03b download it.

| File | Content |
| --- | --- |
| `data/cbis_folds.csv` | The 400 CBIS-DDSM mass ROIs (208 patients) with their patient-level fold (0–4), side, view, and crop size |
| `data/dmid_crops.csv` | The 310 DMID mass crops. `used_after_dedup = True` marks the 291 unique crops (237 images) used in the external test. |

## Repository layout

```
src/qseg/          shared package: conv blocks, circuit layer, state-vector simulator,
                   loss, metrics, augmentation, training loop
src/qseg2.py       global/patch-wise bottlenecks, controls, severing layer, CV runner
notebooks/         Colab notebooks, in run order (see below)
scripts/           sensitivity analysis (learning-rate schedule)
figures/           scripts and aggregated CSVs that reproduce the figures
data/              fold assignments and DMID crop list
```

## How to run (Google Colab)

1. Create the folder `MyDrive/quantum-placement/code/` on Google Drive. Copy `src/qseg/` and `src/qseg2.py` into it.
2. Run the notebooks in this order. Each notebook says which runtime it needs.

| Notebook | Purpose | Runtime |
| --- | --- | --- |
| `01_cbis_ddsm_prep` | Download CBIS-DDSM mass ROIs, make the 256 × 256 crops, and write the patient-level folds | CPU |
| `02_train_models` | Self-test, then train the 8 models × 5 folds with seed 0 (up to 200 epochs, early stopping). Each run also stores its CBIS severing tests and the circuit gradient audit. | T4 GPU |
| `02b_extra_seeds` | Seeds 1 and 2 for the six G/P models | T4 GPU |
| `02c_sensitivity_lr_train` | Sensitivity runs with the learning-rate schedule on the training loss (seed 0) | T4 GPU |
| `03a_external_explore` | Explore the DMID files | CPU |
| `03b_dmid_roi_prep` | Make the DMID mass crops from the ROI masks | CPU |
| `03c_dmid_external_test` | External test of every trained model on DMID, including severing tests. Also counts eligible CDD-CESM cases (exploratory, not used in the paper). | CPU |
| `03d_dmid_test_sensitivity_runs` | DMID test of the sensitivity runs | CPU |
| `05_analysis` | Collect all runs and seeds, de-duplicate the DMID crops, and summarize training dynamics | CPU |
| `06_hierarchical_stats` | Hierarchical bootstrap over seeds and patients/images, with Holm correction | CPU |
| `scripts/07_sensitivity_analysis.py` | Compare the main runs with the sensitivity runs. Run it in a Colab cell after mounting Drive. | CPU |

Every notebook can be resumed: finished runs are skipped. Results are written under `MyDrive/quantum-placement/results/`.

## Figures

Run the figure scripts from inside `figures/`. Output goes to `figures/out/`.

| Paper figure | Script |
| --- | --- |
| Fig. 1 (study overview) | `fig1_overview.py` |
| Fig. 2 (architecture) | `fig2_arch.py` |
| Fig. 3 (seed-to-seed variation) | `figs_extra.py` → `fig6_seeds` |
| Fig. 4 (paired differences) | `fig3_forest.py` |
| Fig. 5 (example segmentations) | `fig5_examples.py` (needs the trained models and data on Drive) |
| Fig. 6 (severing tests) | `fig6_severing_v2.py` |
| Fig. 7 (training dynamics) | `figs_extra.py` → `fig7_training` |

## Environment

The reported runs used TensorFlow 2.21.0 on an NVIDIA T4 GPU in Google Colab. The quantum circuit is simulated exactly as a state vector inside TensorFlow. See `requirements.txt` for the other packages.

## License

MIT, see `LICENSE`.
