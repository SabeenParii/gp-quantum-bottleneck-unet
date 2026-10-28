# gp-quantum-bottleneck-unet

Code, data tables and per-image results for the study **"Simulated Quantum Bottlenecks in U-Net Mammographic Mass Segmentation: No Detected Dice Benefit and Training-Dependent Encoding Effects."**

A simulated 4-qubit, 2-layer parameterized quantum circuit is placed in the bottleneck slot of a U-Net in two ways:

- **global (G):** one 4-value code per image (global average pooling), and
- **patch-wise (P):** one 4-value code per cell of a 4 × 4 grid, with one shared set of circuit weights.

At each placement the circuit (Q) is compared with a classical Dense(4, tanh) layer with the same number of effective weights (Cl) and with an identity mapping (Id) in the same slot. Two classical reference models are trained as well: a U-Net with an uncompressed bottleneck (`full`) and a wider classical U-Net (`unet`). The hybrid and `full` models use dropout (0.1–0.3); `unet` has no dropout.

The lesion position and scale are randomized within each ROI crop. Models are trained on CBIS-DDSM mass ROIs with patient-level five-fold cross-validation (hybrids: three seeds and two learning-rate schedules; 190 runs in total) and tested without fine-tuning on the external DMID dataset. Code-replacement tests (mean training code, another training image's code) measure how sensitive each network is to its bottleneck code; they do not show how the code is used.

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
| `results/per_image_results.csv.gz` | Per-image results of all 190 runs on CBIS-DDSM (own test fold) and on DMID (301 crops, corrected rule): Dice, IoU, HD95, and Dice after code replacement (`dice_const`, `dice_swap0`–`dice_swap9`). Full precision (notebook 22); all statistics are computed from this file |
| `results/per_image_results_original_rule.csv.gz` | Same, with DMID scored under the original assignment rule (291 crops) |
| `results/no_network_baselines*.csv` | Mean-mask and centered-disc baselines without a network (main cohort, original rule, without uncertain records) |
| `results/run_records.csv` | One row per run: epochs, best epoch, training time, GPU, learning-rate history summary, gradient audit of the circuit weights |
| `results/train_val_soft_dice.csv` | Train and validation soft Dice of every run, both measured in inference mode |
| `results/finite_shots_per_image.csv.gz` | Finite-shot evaluation of the 60 circuit runs (notebook 23): per-image Dice with exact expectation values and with 100, 1,000 and 10,000 shots (three draws each), share of flipped pixels and mean absolute probability change |
| `results/replacement_v2_per_image.csv.gz` | Code-replacement tests under the revised protocol (notebook 24; 180 hybrid runs, 68,580 rows): Dice with the normal code recomputed in the same evaluation (`dice`), with the exact mean of the 240 training codes (`dice_const`) and with ten fixed donors (`dice_swap0`–`dice_swap9`), share of flipped pixels and mean absolute probability change. These values are used for Table VIII and Fig. 6 |
| `results/finite_shots_joint_per_image.csv.gz` | Finite-shot evaluation with joint sampling of the four measured qubits from common bitstrings (notebook 24), same layout as `finite_shots_per_image.csv.gz` |
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
| `22_full_precision_export` | Re-exports the per-image results of notebooks 14 and 20 without rounding (the compact exports of notebooks 16 and 20 round to four decimals; they are not used for the statistics) | CPU |
| `23_finite_shot_evaluation` | Re-evaluates the 60 trained circuit models with finite-shot estimates of the circuit outputs (binomial sampling, 100 / 1,000 / 10,000 shots, three draws; no retraining) | T4 GPU |
| `24_revision_analyses` | (A) code-replacement tests under the revised protocol for all 180 hybrid runs; (B) finite-shot evaluation with joint bitstring sampling; (C) prediction examples (Supplementary Fig. S1). No retraining | T4 GPU |

**Statistics (any computer, from the repository root).**

```
python analysis/hierarchical_stats.py                       # main contrasts, model summaries, code-replacement drops
python analysis/additional_analyses.py                      # interaction, between-setting changes, collapsed run, seeds
python analysis/fold0_sensitivity.py                        # all contrasts without fold 0, both schedules
python analysis/severing_tests.py                           # code-replacement tests, original donor scheme (development; Suppl. Table S5 comparison)
python analysis/replacement_v2.py                           # code-replacement tests, revised protocol (Table VIII, Fig. 6, Suppl. Table S5)
python analysis/finite_shots.py                             # finite-shot evaluation of the circuit models
python analysis/finite_shots.py --joint                     # the same with joint bitstring sampling
python analysis/hierarchical_stats.py --exclude-uncertain --out results/stats_excl_uncertain
python analysis/hierarchical_stats.py --results results/per_image_results_original_rule.csv.gz --out results/stats_original_rule
python figures/figs_results.py                              # Figs. 4-6 (Fig. 3 needs the crop QC grid from notebook 11)
```

The bootstrap uses 10,000 replicates with a fixed random seed, so the statistics are reproduced exactly from the per-image results. p-values are approximate bootstrap p-values (twice the smaller share of replicates on either side of zero, floor 1/10,000); Holm correction is applied within each family (nine model contrasts, or twelve code-replacement tests, per schedule and dataset; the eight seed-0 comparisons with the reference models per dataset).

**Collapsed-run sensitivity analysis.** The collapsed run (primary schedule, global identity, fold 0, seed 1) is kept in all main analyses. Its sensitivity analysis removes the run from **both** models of every contrast (`additional_analyses.py`, rows `collapsed run removed (paired)` in `additional_analyses.csv`). This is the only collapsed-run exclusion in the release.

**Code-replacement protocol (revised, used in the paper).** For each fold the bank holds the bottleneck codes of all 240 unique training images. The constant test uses their exact mean (per 4 × 4 cell for the patch-wise models). Swap draw `r` (0–9) of test case `case_id` uses the training image with index `default_rng(int(md5(f"{case_id}|{r}")[:8], 16)).integers(240)` in the sorted list of the fold's training identifiers. Donors therefore do not depend on activations, batch size or image order, and are the same for every model and seed of a fold, which allows paired comparisons of the drops between models (exploratory, `replacement_v2_drop_contrasts.csv`). Replacement CIs are conditional on these fixed donors; the draw-to-draw spread is in `replacement_v2_tests.csv` (`draw_sd`).

**Original donor scheme (development; kept for comparison).** In `Sever` (src/qseg2.py, notebooks 14 and 20) the bank was 256 codes sampled with replacement from the training images, and the bank index of each test image came from a stateless generator seeded by a fixed constant and the sum of the absolute codes of the batch. These values remain in `per_image_results.csv.gz` (`dice_const`, `dice_swap*`). The two schemes differ by at most 0.012 in any mean drop, with no change in Holm-significant status (`replacement_v2_vs_v1.csv`).

### Where each result comes from

| Manuscript item | Script | Output (`results/`) |
| --- | --- | --- |
| Table II (cohorts) | — | `data/cbis_case_descriptors.csv`, `data/dmid_records.csv` |
| Tables IV, V; Fig. 4 | `hierarchical_stats.py` | `stats/summary_models.csv`, `stats/external_drop.csv`, `no_network_baselines.csv`; empty rates and HD95 from `per_image_results.csv.gz` |
| Comparisons with full and unet (Section 4.1) | `hierarchical_stats.py` | `stats/contrasts_vs_reference_seed0.csv` |
| Tables VI, VII; Fig. 5 | `hierarchical_stats.py` | `stats/contrasts.csv` |
| Rows "without the collapsed run" (Tables VI, VII); Suppl. Table S8 | `additional_analyses.py` | `stats/additional_analyses.csv` (`collapsed run removed (paired)`) |
| Table VIII; Fig. 6; Suppl. Table S5 | `replacement_v2.py`; `figures/figs_results.py` | `stats/replacement_v2_tests.csv`, `stats/replacement_v2_distribution.csv`, `stats/replacement_v2_drop_contrasts.csv`, `stats/replacement_v2_vs_v1.csv` |
| Section 4.6; Suppl. Table S10 (training dynamics) | — | `run_records.csv`, `train_val_soft_dice.csv` |
| Suppl. Table S2 (DMID reference, three cohorts) | `hierarchical_stats.py` (main, `--results …original_rule…`, `--exclude-uncertain`) | `stats/`, `stats_original_rule/`, `stats_excl_uncertain/` |
| Suppl. Table S3 (development history) | — | `preliminary/`, notebooks 11–24 |
| Suppl. Table S4 (without fold 0) | `fold0_sensitivity.py` | `stats/fold0_excluded_both_schedules.csv` |
| Suppl. Table S6 (exact p-values) | `hierarchical_stats.py`, `replacement_v2.py` | `stats/contrasts.csv`, `stats/replacement_v2_tests.csv` |
| Suppl. Table S7 (seed-specific and per-run) | `additional_analyses.py`, `hierarchical_stats.py` | `stats/seed_contrasts.csv`, `stats/per_run_contrasts.csv` |
| Suppl. Table S9 (changes between settings) | `additional_analyses.py` | `stats/additional_analyses.csv` (`schedule change …`, `dataset change …`) |
| Section 4.7; Suppl. Table S11 (finite shots) | `finite_shots.py` | `stats/finite_shots_drop.csv`, `stats/finite_shots_contrasts.csv`, `stats/finite_shots_exact_check.csv` |
| Section 4.7; Suppl. Table S12 (joint sampling) | `finite_shots.py --joint` | `stats/finite_shots_joint_*.csv` |
| Suppl. Fig. S1 (prediction examples) | notebook 24, cell 5 | figure and case list written by the notebook |

### Levels of reproduction

1. **Statistics, tables and figures from the released per-image results** (any computer, minutes): the commands above. The bootstrap is seeded, so every reported estimate, CI and p-value is reproduced exactly.
2. **Inference from trained models** (notebooks 14, 20, 23, 24; T4 GPU): needs the saved checkpoints. The 190 checkpoints are not included in the release because of their size; they are retained by the author and are available on request. GPU inference is not bit-reproducible, so re-scored Dice values can differ from the released ones in the third or fourth decimal (the notebooks report this difference).
3. **Retraining** (notebooks 11–13; 5 to 28 minutes per run, about 43 GPU hours for all 190 runs on T4 or L4 GPUs, from `run_records.csv`): reproduces the experiment but not the exact weights, because GPU training is not deterministic.

## Changes in v2.1.0

- Code-replacement tests under a revised protocol (all 240 training codes, exact mean, fixed hash-based donors shared by all models and seeds of a fold): notebook 24, `analysis/replacement_v2.py`, `results/replacement_v2_per_image.csv.gz`. These replace the original-scheme values in Table VIII and Fig. 6; the original values are kept for comparison.
- Finite-shot evaluation with joint bitstring sampling (`finite_shots.py --joint`, `results/finite_shots_joint_per_image.csv.gz`).
- Prediction-example figure (notebook 24).
- Preliminary notebook `02d_sensitivity_lr_train_seeds12` (training-loss schedule, seeds 1 and 2, of the earlier experiment), so that the full development history is archived.
- Mapping of results to the manuscript and supplement updated; reproduction levels documented.

## Changes in v2.0.2

- Finite-shot evaluation of the trained circuit models (notebook 23, `analysis/finite_shots.py`, `results/finite_shots_per_image.csv.gz`). The circuit outputs are replaced by binomial N-shot estimates (N = 100, 1,000, 10,000; qubits sampled independently; no hardware noise; no retraining). The exact predictions are recomputed in the same notebook; they agree with the stored per-image Dice to within 0.006 (floating-point differences between evaluation sessions).

## Changes in v2.0.1

- Per-image results are released at full precision (notebook 22) and all statistics were regenerated from them (compared with the earlier four-decimal files, estimates and CI limits changed by at most 0.0002 and p-values by at most 0.0014; no resolved or Holm-significant status changed).
- The earlier one-sided exclusion of the collapsed run in `hierarchical_stats.py` was removed; the paired analysis in `additional_analyses.py` is the only collapsed-run exclusion.
- Documentation of the code-replacement draws (above and in `Sever`).

## Environment

The reported runs used TensorFlow 2.21.0 on NVIDIA T4 or L4 GPUs in Google Colab. The quantum circuit is simulated exactly as a state vector inside TensorFlow; notebook 23 adds a finite-shot readout at test time only. See `requirements.txt` for the other packages. The statistics in `results/stats/` were regenerated with Python 3.13, NumPy 2.5, pandas 3.0 and SciPy 1.18; the analysis scripts use only NumPy and pandas operations that are stable across these versions.

## License

MIT, see `LICENSE`.
