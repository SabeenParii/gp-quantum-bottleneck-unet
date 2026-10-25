# gp-quantum-bottleneck-unet

Code, data tables and per-image results for the study **"Simulated Quantum Bottlenecks in U-Net Mammographic Mass Segmentation: No Resolved Dice Benefit and Training-Dependent Encoding Effects."**

A simulated 4-qubit, 2-layer parameterized quantum circuit is placed in the bottleneck slot of a U-Net in two ways:

- **global (G):** one 4-value code per image (global average pooling), and
- **patch-wise (P):** one 4-value code per cell of a 4 × 4 grid, with one shared set of circuit weights.

At each placement the circuit (Q) is compared with a classical Dense(4, tanh) layer with the same number of effective weights (Cl) and with an identity mapping (Id) in the same slot. Two classical reference models are trained as well: a U-Net with an uncompressed bottleneck (`full`) and a wider classical U-Net (`unet`).

The lesion position and scale are randomized within each ROI crop. Models are trained on CBIS-DDSM mass ROIs with patient-level five-fold cross-validation (hybrids: three seeds and two learning-rate schedules; 190 runs in total) and tested without fine-tuning on the external DMID dataset. Code-replacement tests (mean code, other image's code) measure how sensitive each network is to its bottleneck code.

## Data

Only public, de-identified datasets are used. **No images are included in this repository.**

- **CBIS-DDSM** is available from The Cancer Imaging Archive. Notebook 11 downloads it through the public TCIA API.
- **DMID** (Digital Mammography Dataset for Breast Cancer Diagnosis Research) is available from its authors' figshare record. Notebooks 11 and 18b–18d download it.

| File | Content |
| --- | --- |
| `data/cbis_folds_randomized_crops.csv` | The 400 CBIS-DDSM mass ROIs (208 patients): patient-level fold (0–4) and the parameters of every randomized crop (crop side, scale, shift, padding, mask fraction) |
| `data/cbis_case_descriptors.csv` | Public case descriptors of all 1,696 CBIS-DDSM mass ROIs; `selected` marks the 400 used |
| `data/dmid_records.csv` | All 348 listed DMID masses with coordinates: assigned mask components under the corrected (competitive) rule, status (retained, duplicate, skipped), crop parameters, and the assignment under the original rule for comparison |
| `data/dmid_uncertain_records.csv` | The 24 retained DMID records judged uncertain after visual review, with the reason |
| `data/dmid_duplicates.csv` | Records removed as duplicates (identical full-resolution component set) and the retained record |
| `data/dmid_records_original_rule.csv` | DMID crop list under the original assignment rule (310 records, 291 retained) |

## Results

| File | Content |
| --- | --- |
| `results/per_image_results.csv.gz` | Per-image results of all 190 runs on CBIS-DDSM (own test fold) and on DMID (301 crops, corrected rule): Dice, IoU, HD95, and Dice after code replacement (`dice_const`, `dice_swap0`–`dice_swap9`) |
| `results/per_image_results_original_rule.csv.gz` | Same, with DMID scored under the original assignment rule (291 crops) |
| `results/no_network_baselines*.csv` | Mean-mask and centered-disc baselines without a network (main cohort, original rule, without uncertain records) |
| `results/run_records.csv` | One row per run: epochs, best epoch, training time, GPU, learning-rate history summary, gradient audit of the circuit weights |
| `results/train_val_soft_dice.csv` | Train and validation soft Dice of every run, both measured in inference mode |
| `results/stats/` | Output of the analysis scripts (contrasts, Holm-adjusted p-values, code-replacement tests, sensitivity analyses) |

Column codes: `set` / `schedule` = learning-rate schedule (`train`: reduce on training loss, primary; `val`: reduce on validation loss); `model` = `g_q`, `g_cl`, `g_id`, `p_q`, `p_cl`, `p_id` (placement_block), `full`, `unet`; `dataset` = `CBIS` or `DMID`.

## Repository layout

```
src/qseg/          shared package: conv blocks, circuit layer, state-vector simulator,
                   loss, metrics, augmentation, training loop
src/qseg2.py       global/patch-wise bottlenecks, controls, code-replacement layer, CV runner
notebooks/         Colab notebooks of the reported study, in run order (see below)
analysis/          statistics (crossed hierarchical bootstrap, Holm correction)
figures/           figure scripts
data/              folds, crop parameters, DMID reference records
results/           per-image results of all runs and the statistics
preliminary/       earlier pipeline with mask-centered crops (preliminary analysis; not the reported experiment)
```

## How to run

**Training and evaluation (Google Colab).** Create `MyDrive/quantum-placement/code/` on Google Drive and copy `src/qseg/` and `src/qseg2.py` into it. Then run the notebooks in this order. Every notebook can be resumed; finished runs are skipped.

| Notebook | Purpose | Runtime |
| --- | --- | --- |
| `11_randomized_crops` | Download CBIS-DDSM and DMID, make the randomized ROI crops, compute the no-network baselines | CPU |
| `12_pilot` | Pilot on fold 0, seed 0 (learning-rate schedule choice); not part of the 190 runs | T4 GPU |
| `13_train_session_A`, `13_train_session_B` | Train the 190 runs (two copies can run at the same time) | T4 GPU |
| `14_evaluation` | Evaluate every run in-domain and on DMID, including the code-replacement tests | T4 GPU |
| `15_quick_table`, `16_slim_export`, `17_run_records` | Summary table, compact per-image table, run records | CPU |
| `18b_dmid_audit_fullres` / `18c_dmid_audit_lowmem`, `18d_dmid_sheets` | Full-resolution DMID reference audit and overlay sheets (no model output); CBIS-DDSM descriptors | CPU |
| `18_dmid_audit_sheets` | Overlay sheets of the 256 × 256 DMID crops | CPU |
| `19_matched_train_val_dice` | Train and validation soft Dice under the same (inference) conditions | T4 GPU |
| `20_dmid_corrected_rescore` | Corrected DMID reference assignment and re-scoring of all saved models (no retraining) | T4 GPU |
| `21_dmid_corrected_baselines` | No-network baselines on the corrected DMID cohort | CPU |

**Statistics (any computer, from the repository root).**

```
python analysis/hierarchical_stats.py                       # main contrasts, model summaries, code-replacement drops
python analysis/additional_analyses.py                      # interaction, between-setting changes, collapsed run, seeds
python analysis/fold0_sensitivity.py                        # all contrasts without fold 0, both schedules
python analysis/severing_tests.py                           # code-replacement tests with Holm correction
python analysis/hierarchical_stats.py --exclude-uncertain --out results/stats_excl_uncertain
python analysis/hierarchical_stats.py --results results/per_image_results_original_rule.csv.gz --out results/stats_original_rule
python figures/figs_results.py                              # Figs. 4-6 (Fig. 3 needs the crop QC grid from notebook 11)
```

The bootstrap uses 10,000 replicates with a fixed random seed, so the statistics are reproduced exactly from the per-image results.

## Environment

The reported runs used TensorFlow 2.21.0 on NVIDIA T4 or L4 GPUs in Google Colab. The quantum circuit is simulated exactly as a state vector inside TensorFlow. See `requirements.txt` for the other packages.

## License

MIT, see `LICENSE`.
