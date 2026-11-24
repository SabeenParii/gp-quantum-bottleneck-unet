"""Training cost of the 190 main runs (Table of computational cost in the journal manuscript).

Reads results/run_records.csv and writes results/stats/train_cost.csv with, per model and LR schedule:
number of runs, trainable plus non-trainable weights, median epochs, median seconds per epoch and
median and total training time. All runs were trained on one NVIDIA T4 GPU.
Inference times come from notebooks/41_compute_cost.ipynb (results/compute_cost/inference_ms.csv).
"""
import os
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
d = pd.read_csv(os.path.join(ROOT, "results", "run_records.csv"))
d["sec_per_epoch"] = d.train_seconds / d.epochs_run
out = (d.groupby(["schedule", "model"])
         .agg(runs=("run", "size"), weights=("params_total", "first"),
              epochs_median=("epochs_run", "median"), sec_per_epoch_median=("sec_per_epoch", "median"),
              train_min_median=("train_seconds", lambda s: s.median() / 60),
              train_h_total=("train_seconds", lambda s: s.sum() / 3600))
         .round(2).reset_index())
both = (d.groupby("model")
          .agg(runs=("run", "size"), weights=("params_total", "first"),
               epochs_median=("epochs_run", "median"), sec_per_epoch_median=("sec_per_epoch", "median"),
               train_min_median=("train_seconds", lambda s: s.median() / 60),
               train_h_total=("train_seconds", lambda s: s.sum() / 3600))
          .round(2).reset_index().assign(schedule="both"))
out = pd.concat([out, both], ignore_index=True)
path = os.path.join(ROOT, "results", "stats", "train_cost.csv")
out.to_csv(path, index=False)
print(out.to_string(index=False))
print("total GPU hours:", round(d.train_seconds.sum() / 3600, 1))
