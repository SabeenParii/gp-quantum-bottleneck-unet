"""Paper 2: placement of the quantum bottleneck (global vs patch-wise) and bottleneck use.

Builds on the Paper 1 package `qseg` (same conv blocks, circuit, simulator, loss, metrics,
augmentation and training settings). New here:

* Global (G) and patch-wise (P) bottlenecks with the same 4->4 block slot, each with
  circuit (Q), equal-parameter classical layer (Cl) or identity (Id). Neither uses the
  67M-parameter dense expansion of Paper 1:
      G: 16x16x512 -> GAP -> Dense(4, tanh) -> block -> Dense(512, relu) -> tile 16x16
      P: 16x16x512 -> AvgPool(4) -> Conv1x1(4, tanh) -> block per cell (16 cells)
                   -> Conv1x1(512, relu) -> nearest up-sampling x4
  G and P have identical parameter counts; the only difference is spatial structure.
* "full": same encoder/decoder, bottleneck map passed on uncompressed (upper reference).
* Sever layer after the block. Test-time modes: 0 = normal, 1 = constant (mean training
  code), 2 = swap (code of a random training image). No retraining needed.
* Patient-level 5-fold CV on CBIS-DDSM mass ROIs (folds.csv from notebook 01).
* Per-image Dice / IoU / HD95, per-parameter circuit gradient audit, resumable runs.
"""
import json
import os
import random
import time

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models

from qseg.config import Config
from qseg.datasets import make_augment_fn
from qseg.evaluation import evaluate_predictions
from qseg.models import TFCircuitLayer, build_unet, conv_block, output_layer, parameter_breakdown
from qseg.train import bce_dice_loss, soft_dice

MODELS2 = {
    "unet": "C-UNet (Paper 1 baseline)",
    "full": "Hybrid U-Net, uncompressed bottleneck",
    "g_q": "Global bottleneck, circuit",
    "g_cl": "Global bottleneck, classical Dense(4)",
    "g_id": "Global bottleneck, identity",
    "p_q": "Patch-wise bottleneck, circuit",
    "p_cl": "Patch-wise bottleneck, classical Dense(4)",
    "p_id": "Patch-wise bottleneck, identity",
}


# ----------------------------------------------------------------------------- layers
class PatchCircuitLayer(TFCircuitLayer):
    """The Paper 1 circuit applied independently to every cell of a (B, h, w, 4) map.
    One shared set of 24 weights (20 effective), exactly as in Paper 1."""

    def call(self, inputs):
        x = tf.cast(inputs, tf.float32)
        shape = tf.shape(x)
        flat = tf.reshape(x, [-1, self.n_qubits])
        out = super().call(flat)
        return tf.reshape(out, shape)

    def compute_output_shape(self, input_shape):
        return input_shape


class Sever(layers.Layer):
    """Identity during training. At test time `mode` = 1 replaces every code by the mean
    training code, `mode` = 2 by the code of a random training image (from `bank`)."""

    def __init__(self, bank_size=256, **kwargs):
        super().__init__(**kwargs)
        self.bank_size = bank_size

    def build(self, input_shape):
        self.mode = self.add_weight(name="mode", shape=(), initializer="zeros", trainable=False)
        self.bank = self.add_weight(name="bank", shape=(self.bank_size,) + tuple(input_shape[1:]),
                                    initializer="zeros", trainable=False)
        super().build(input_shape)

    def call(self, x):
        x = tf.cast(x, tf.float32)
        mode = tf.cast(self.mode, tf.float32)
        bank = tf.cast(self.bank, tf.float32)
        const = tf.broadcast_to(tf.reduce_mean(bank, axis=0, keepdims=True), tf.shape(x))
        # deterministic per batch, different between batches
        s2 = tf.math.floormod(tf.cast(tf.reduce_sum(tf.abs(x)) * 1000.0, tf.int64), 1_000_003)
        idx = tf.random.stateless_uniform([tf.shape(x)[0]], seed=tf.stack([tf.constant(17, tf.int64), s2]),
                                          minval=0, maxval=self.bank_size, dtype=tf.int32)
        swap = tf.gather(bank, idx)
        return tf.where(mode > 1.5, swap, tf.where(mode > 0.5, const, x))

    def compute_output_shape(self, input_shape):
        return input_shape


def _block(z, kind, n_qubits, n_layers, patch):
    if kind == "q":
        cls = PatchCircuitLayer if patch else TFCircuitLayer
        return cls(n_qubits, n_layers, name="quantum_circuit")(z)
    if kind == "cl":
        return layers.Dense(n_qubits, activation="tanh", name="classical_replacement")(z)
    if kind == "id":
        return z
    raise ValueError(kind)


# ----------------------------------------------------------------------------- models
def build_p2_model(name, input_shape=(256, 256, 1), n_qubits=4, n_q_layers=2,
                   filters=(32, 64, 128, 256, 512), dropouts=(0.1, 0.1, 0.2, 0.2, 0.3)):
    if name == "unet":
        return build_unet(input_shape)
    inputs = layers.Input(input_shape, name="input_image")
    skips, x = [], inputs
    for i, (f, d) in enumerate(zip(filters[:-1], dropouts[:-1])):
        x = conv_block(x, f, d, name=f"enc{i + 1}")
        skips.append(x)
        x = layers.MaxPooling2D(2, name=f"pool{i + 1}")(x)
    x = conv_block(x, filters[-1], dropouts[-1], name="bottleneck")      # (16, 16, 512)
    side = input_shape[0] // 16

    if name == "full":
        pass
    elif name.startswith("g_"):
        z = layers.GlobalAveragePooling2D(name="gap")(x)
        z = layers.Dense(n_qubits, activation="tanh", name="pre_dense")(z)
        z = _block(z, name[2:], n_qubits, n_q_layers, patch=False)
        z = Sever(name="sever")(z)
        z = layers.Dense(filters[-1], activation="relu", name="post_dense")(z)
        z = layers.Reshape((1, 1, filters[-1]), name="to_map")(z)
        x = layers.UpSampling2D(side, interpolation="nearest", name="tile")(z)
    elif name.startswith("p_"):
        z = layers.AveragePooling2D(4, name="patch_pool")(x)              # (4, 4, 512)
        z = layers.Conv2D(n_qubits, 1, activation="tanh", name="pre_conv")(z)
        z = _block(z, name[2:], n_qubits, n_q_layers, patch=True)
        z = Sever(name="sever")(z)
        z = layers.Conv2D(filters[-1], 1, activation="relu", name="post_conv")(z)
        x = layers.UpSampling2D(side // 4, interpolation="nearest", name="patch_up")(z)
    else:
        raise ValueError(f"Unknown model: {name}")

    for i, (f, s, d) in enumerate(zip(reversed(filters[:-1]), reversed(skips), reversed(dropouts[:-1]))):
        x = layers.UpSampling2D(2, name=f"up{i + 1}")(x)
        x = conv_block(layers.Concatenate(name=f"concat{i + 1}")([x, s]), f, d, name=f"dec{i + 1}")
    return models.Model(inputs, output_layer(x), name=f"P2_{name}")


# ----------------------------------------------------------------------------- data
def load_cbis(data_dir):
    """All ROI crops as arrays + the folds table (order = folds.csv)."""
    df = pd.read_csv(os.path.join(data_dir, "folds.csv"))
    imgs, msks = [], []
    for rid in df["roi_id"]:
        im = tf.image.decode_png(tf.io.read_file(os.path.join(data_dir, "images", f"{rid}.png")), channels=1)
        mk = tf.image.decode_png(tf.io.read_file(os.path.join(data_dir, "masks", f"{rid}.png")), channels=1)
        imgs.append(im.numpy().astype(np.float32) / 255.0)
        msks.append((mk.numpy() > 127).astype(np.float32))
    return np.stack(imgs), np.stack(msks), df


def fold_split(df, k, n_folds=5):
    """test = fold k, validation = fold k+1, training = the other three (patient-level)."""
    f = df["fold"].to_numpy()
    te = np.where(f == k)[0]
    va = np.where(f == (k + 1) % n_folds)[0]
    tr = np.where((f != k) & (f != (k + 1) % n_folds))[0]
    return tr, va, te


def train_dataset(cfg, x, y, seed):
    ds = tf.data.Dataset.from_tensor_slices((x, y)).shuffle(len(x), seed=seed, reshuffle_each_iteration=True)
    try:
        seeds = tf.data.Dataset.random(seed=seed, rerandomize_each_iteration=True).batch(2)
    except TypeError:
        seeds = tf.data.Dataset.random(seed=seed).batch(2)
    augment = make_augment_fn(cfg)
    ds = tf.data.Dataset.zip((ds, seeds)).map(lambda p, s: augment(p[0], p[1], s), num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(cfg.batch_size).prefetch(tf.data.AUTOTUNE)


def eval_dataset(cfg, x, y):
    return tf.data.Dataset.from_tensor_slices((x, y)).batch(cfg.batch_size).prefetch(tf.data.AUTOTUNE)


# ----------------------------------------------------------------------------- metrics
def hd95(t, p):
    """95th-percentile symmetric Hausdorff distance (pixels at 256x256) between mask borders.
    NaN if either mask is empty."""
    from scipy import ndimage
    t, p = np.asarray(t).astype(bool), np.asarray(p).astype(bool)
    if t.sum() == 0 or p.sum() == 0:
        return float("nan")
    bt = t ^ ndimage.binary_erosion(t)
    bp = p ^ ndimage.binary_erosion(p)
    dt = ndimage.distance_transform_edt(~bt)
    dp = ndimage.distance_transform_edt(~bp)
    d = np.concatenate([dp[bt], dt[bp]])
    return float(np.percentile(d, 95))


def per_image_scores(y, prob, thr=0.5):
    yt = y[..., 0] > 0.5
    yp = prob[..., 0] >= thr
    rows = []
    for t, p in zip(yt, yp):
        tp, fp, fn = np.sum(t & p), np.sum(~t & p), np.sum(t & ~p)
        dice = 1.0 if tp + fp + fn == 0 else 2 * tp / (2 * tp + fp + fn)
        iou = 1.0 if tp + fp + fn == 0 else tp / (tp + fp + fn)
        rows.append((dice, iou, hd95(t, p)))
    return pd.DataFrame(rows, columns=["dice", "iou", "hd95"])


# ----------------------------------------------------------------------------- audits
def circuit_layer(model):
    return model.get_layer("quantum_circuit") if "quantum_circuit" in [l.name for l in model.layers] else None


def gradient_audit(model, x, y, batch=16, n_batches=4):
    """Mean |d loss / d w| per circuit weight over a few batches, shape (layers, qubits, 3)."""
    layer = circuit_layer(model)
    if layer is None:
        return None
    acc = np.zeros(layer.qweights.shape, np.float64)
    for b in range(n_batches):
        xb, yb = x[b * batch:(b + 1) * batch], y[b * batch:(b + 1) * batch]
        if len(xb) == 0:
            break
        with tf.GradientTape() as tape:
            loss = bce_dice_loss(yb, model(xb, training=False))
        acc += np.abs(tape.gradient(loss, layer.qweights).numpy())
    acc /= max(1, b + 1)
    return dict(per_weight=acc.tolist(), n_effective=int((acc > 1e-8).sum()), n_total=int(acc.size))


def set_sever_bank(model, x_train, batch=16, seed=0):
    sev = model.get_layer("sever")
    code_model = models.Model(model.input, sev.input)
    codes = code_model.predict(x_train, batch_size=batch, verbose=0)
    pick = np.random.default_rng(seed).choice(len(codes), sev.bank_size, replace=len(codes) < sev.bank_size)
    sev.bank.assign(codes[pick].astype(np.float32))
    return codes


def predict_with_mode(model, x, mode, batch=16):
    has = "sever" in [l.name for l in model.layers]
    if has:
        model.get_layer("sever").mode.assign(float(mode))
    try:
        return model.predict(x, batch_size=batch, verbose=0)
    finally:
        if has:
            model.get_layer("sever").mode.assign(0.0)


# ----------------------------------------------------------------------------- training
def set_seeds(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


def run_fold(cfg, data, model_name, fold, seed, out_root, min_epochs=15, verbose=1, max_epochs=None):
    """Train one model on one fold; resumable (finished runs are skipped)."""
    x, y, df = data
    out_dir = os.path.join(out_root, f"{model_name}_fold{fold}_seed{seed}")
    os.makedirs(out_dir, exist_ok=True)
    done = os.path.join(out_dir, "summary.json")
    if os.path.exists(done):
        with open(done) as f:
            return json.load(f)

    tf.keras.backend.clear_session()
    tf.keras.mixed_precision.set_global_policy("float32")
    set_seeds(seed)
    tr, va, te = fold_split(df, fold)
    model = build_p2_model(model_name, (cfg.image_size, cfg.image_size, cfg.in_channels), cfg.n_qubits, cfg.n_q_layers)
    model.compile(optimizer=tf.keras.optimizers.Adam(cfg.learning_rate), loss=bce_dice_loss, metrics=[soft_dice],
                  jit_compile=False)
    grad_init = gradient_audit(model, x[tr], y[tr])
    ckpt = os.path.join(out_dir, "best.weights.h5")
    cbs = [
        tf.keras.callbacks.TerminateOnNaN(),
        tf.keras.callbacks.ModelCheckpoint(ckpt, monitor="val_loss", mode="min", save_best_only=True,
                                           save_weights_only=True),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", mode="min", patience=cfg.early_stop_patience,
                                         restore_best_weights=True, start_from_epoch=min_epochs),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=cfg.lr_factor, patience=cfg.lr_patience,
                                             min_lr=cfg.min_lr),
        tf.keras.callbacks.CSVLogger(os.path.join(out_dir, "history.csv")),
    ]
    t0 = time.perf_counter()
    hist = model.fit(train_dataset(cfg, x[tr], y[tr], seed), validation_data=eval_dataset(cfg, x[va], y[va]),
                     epochs=max_epochs or cfg.max_epochs, callbacks=cbs, verbose=verbose)
    train_s = time.perf_counter() - t0
    h = pd.DataFrame(hist.history)
    vl = h["val_loss"].to_numpy()
    if not np.isfinite(vl).any():
        raise RuntimeError(f"{model_name} fold {fold}: validation loss never finite")
    best_epoch = int(np.nanargmin(vl)) + 1
    model.load_weights(ckpt)

    prob = model.predict(x[te], batch_size=cfg.batch_size, verbose=0)
    pid = df["patient"].to_numpy()[te]
    summary, _ = evaluate_predictions(y[te], prob, pid, cfg.threshold, 0, cfg.auc_max_pixels, cfg.ece_bins, seed)
    per = per_image_scores(y[te], prob, cfg.threshold)
    per.insert(0, "patient", pid)
    per.insert(0, "roi_id", df["roi_id"].to_numpy()[te])

    if "sever" in [l.name for l in model.layers]:
        set_sever_bank(model, x[tr], cfg.batch_size, seed)
        for mode, tag in ((1, "const"), (2, "swap")):
            s = per_image_scores(y[te], predict_with_mode(model, x[te], mode, cfg.batch_size), cfg.threshold)
            per[f"dice_{tag}"] = s["dice"].to_numpy()
        summary.update(dice_sever_const=float(per["dice_const"].mean()), dice_sever_swap=float(per["dice_swap"].mean()))

    summary.update(model=model_name, model_label=MODELS2[model_name], fold=fold, seed=seed,
                   epochs_run=int(len(h)), best_epoch=best_epoch, best_val_loss=float(vl[best_epoch - 1]),
                   train_seconds=round(train_s, 1), dice_per_image=float(per["dice"].mean()),
                   iou_per_image=float(per["iou"].mean()), hd95_median=float(np.nanmedian(per["hd95"])),
                   n_test=int(len(te)), n_test_patients=int(len(set(pid))), tf_version=tf.__version__,
                   gpu=str([d.name for d in tf.config.list_physical_devices("GPU")]),
                   grad_audit_init=grad_init, grad_audit_best=gradient_audit(model, x[tr], y[tr]),
                   **parameter_breakdown(model))
    per.to_csv(os.path.join(out_dir, "per_image.csv"), index=False)
    with open(done, "w") as f:
        json.dump(summary, f, indent=2)
    if verbose:
        msg = f"{model_name} fold {fold}: Dice {summary['dice_per_image']:.4f}, best epoch {best_epoch}/{len(h)}"
        if "dice_sever_const" in summary:
            msg += f", severed: const {summary['dice_sever_const']:.4f} swap {summary['dice_sever_swap']:.4f}"
        print(msg)
    return summary


def collect(out_root):
    rows = []
    for d in sorted(os.listdir(out_root)):
        p = os.path.join(out_root, d, "summary.json")
        if os.path.exists(p):
            with open(p) as f:
                s = json.load(f)
            rows.append({k: v for k, v in s.items() if not isinstance(v, (dict, list))})
    return pd.DataFrame(rows)
