"""TinyQuant Lab: CPU-only, reproducible weight-quantization experiments."""
from __future__ import annotations
import argparse
import csv
from dataclasses import dataclass
import hashlib
import html
import io
import json
import os
from pathlib import Path
import platform
import sys
import time

# Fix the experiment's BLAS thread budget before NumPy initializes.
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
import numpy as np

VERSION = "0.1.0"
ALPHAS = np.linspace(0.45, 1.0, 12).tolist()
SEEDS = [17, 43, 91]
DATASET_URL = "https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html"
REFERENCE_URL = "https://proceedings.mlr.press/v267/kim25d.html"


def finite(array, name="array"):
    a = np.asarray(array)
    if not np.issubdtype(a.dtype, np.number) or np.iscomplexobj(a) or not np.all(np.isfinite(a)):
        raise ValueError(f"{name} must contain finite numbers")
    return a


def as_float32(array, name="array"):
    original = finite(array, name)
    if np.any(np.abs(original) > np.finfo(np.float32).max):
        raise ValueError(f"{name} exceeds float32 range")
    return finite(np.asarray(original, dtype=np.float32), name)


def load_data():
    cached = Path(__file__).parent / "dataset" / "digits.csv.gz"
    if cached.exists():
        import gzip
        source = json.loads((cached.parent / "provenance.json").read_text(encoding="utf-8"))
        if hashlib.sha256(cached.read_bytes()).hexdigest() != source["sha256"]:
            raise ValueError("cached dataset fingerprint mismatch")
        table = np.loadtxt(gzip.open(cached, "rb"), delimiter=",")
        x = np.ascontiguousarray(table[:, :-1], dtype=np.float32) / np.float32(16)
        y = np.asarray(table[:, -1], dtype=np.int64)
        loader = "NumPy parser of official bundled digits.csv.gz"
        dataset_version = source["source_release"]
    else:
        from sklearn.datasets import load_digits
        import sklearn
        data = load_digits()
        x = np.ascontiguousarray(data.data, dtype=np.float32) / np.float32(16)
        y = np.asarray(data.target, dtype=np.int64)
        loader = "sklearn.datasets.load_digits"
        dataset_version = sklearn.__version__
        source = {"source_url": DATASET_URL}
    if x.shape != (1797, 64) or len(y) != 1797 or set(np.unique(y)) != set(range(10)):
        raise ValueError("unexpected digits dataset")
    fingerprint = hashlib.sha256(x.tobytes() + y.tobytes()).hexdigest()
    return x, y, {
        "name": "scikit-learn bundled digits", "url": DATASET_URL,
        "upstream_dataset": "UCI Optical Recognition of Handwritten Digits",
        "uci_url": "https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits",
        "dataset_version": dataset_version, "loader": loader, "file_provenance": source, "samples": len(y),
        "features": x.shape[1], "classes": 10, "pixel_normalization": "pixel / 16",
        "normalized_data_sha256": fingerprint,
        "selection": "Complete bundled load_digits dataset; stratified random split, not official UCI partition."
    }


def split_indices(y, seed):
    """Per-class shuffle: ~65% train, ~15% calibration, remaining ~20% test."""
    y = np.asarray(y)
    rng = np.random.default_rng(seed)
    train, calibration, test = [], [], []
    for label in np.unique(y):
        indices = rng.permutation(np.flatnonzero(y == label))
        ntrain, ncal = int(len(indices) * .65), int(len(indices) * .15)
        train.extend(indices[:ntrain])
        calibration.extend(indices[ntrain:ntrain+ncal])
        test.extend(indices[ntrain+ncal:])
    result = [rng.permutation(np.asarray(v, dtype=np.int64)) for v in (train, calibration, test)]
    assert_disjoint(*result, n=len(y))
    return tuple(result)


def assert_disjoint(train, calibration, test, n):
    groups = [np.asarray(v, dtype=np.int64) for v in (train, calibration, test)]
    joined = np.concatenate(groups)
    if any(len(v) == 0 for v in groups):
        raise ValueError("All three partitions must be nonempty")
    if len(joined) != n or len(np.unique(joined)) != n or np.any(joined < 0) or np.any(joined >= n):
        raise ValueError("Partitions must cover each sample exactly once")


def logits(model, x):
    x = as_float32(x, "inputs")
    if x.ndim != 2 or x.shape[1] != model["w1"].shape[1] or len(x) == 0:
        raise ValueError("inputs must be a nonempty sample-by-feature matrix matching the model")
    return np.maximum(x @ model["w1"].T + model["b1"], 0) @ model["w2"].T + model["b2"]


def loss_and_grad(model, x, y, l2=1e-4):
    """Mean cross entropy plus half-L2; independent NumPy backpropagation."""
    z1 = x @ model["w1"].T + model["b1"]
    h = np.maximum(z1, 0)
    scores = h @ model["w2"].T + model["b2"]
    scores = scores - np.max(scores, axis=1, keepdims=True)
    probs = np.exp(scores)
    probs /= probs.sum(axis=1, keepdims=True)
    loss = -np.log(np.maximum(probs[np.arange(len(y)), y], 1e-30)).mean()
    loss += .5 * l2 * sum(float(np.square(model[k], dtype=np.float64).sum()) for k in ("w1", "w2"))
    ds = probs.copy()
    ds[np.arange(len(y)), y] -= 1
    ds /= len(y)
    dh = (ds @ model["w2"]) * (z1 > 0)
    grad = {
        "w2": ds.T @ h + l2 * model["w2"],
        "b2": ds.sum(axis=0),
        "w1": dh.T @ x + l2 * model["w1"],
        "b1": dh.sum(axis=0)
    }
    return float(loss), grad


def train_mlp(x, y, seed, epochs=350, hidden=48, batch=128):
    rng = np.random.default_rng(seed)
    model = {
        "w1": (rng.standard_normal((hidden, x.shape[1])) * np.sqrt(2 / x.shape[1])).astype(np.float32),
        "b1": np.zeros(hidden, np.float32),
        "w2": (rng.standard_normal((10, hidden)) * np.sqrt(2 / hidden)).astype(np.float32),
        "b2": np.zeros(10, np.float32),
    }
    m = {k: np.zeros_like(v) for k, v in model.items()}
    v = {k: np.zeros_like(a) for k, a in model.items()}
    history = []
    step = 0
    for epoch in range(epochs):
        order = rng.permutation(len(y))
        lr = .008 * (0.35 if epoch >= epochs * .7 else 1)
        for start in range(0, len(y), batch):
            chosen = order[start:start+batch]
            _, grads = loss_and_grad(model, x[chosen], y[chosen])
            step += 1
            for key in model:
                m[key] *= .9
                m[key] += .1 * grads[key]
                v[key] *= .999
                v[key] += .001 * np.square(grads[key])
                update = (m[key] / (1 - .9**step)) / (np.sqrt(v[key] / (1 - .999**step)) + 1e-8)
                model[key] -= np.float32(lr) * update
        if epoch % 25 == 0 or epoch == epochs - 1:
            loss, _ = loss_and_grad(model, x, y)
            history.append({"epoch": epoch + 1, "train_loss": loss, "train_accuracy": accuracy(logits(model, x), y)})
    for key in model:
        finite(model[key], key)
    return model, history


def accuracy(scores, y):
    return float(np.mean(np.argmax(scores, axis=1) == y))


@dataclass
class QuantRow:
    values: np.ndarray
    scales: np.ndarray
    bits: int
    alpha: float

    def reconstruct(self):
        return self.values.astype(np.float32) * self.scales[:, None]


def quantize_rows(weights, bits, alpha=1.0):
    w = as_float32(weights, "weights")
    if w.ndim != 2 or w.size == 0:
        raise ValueError("weights must be a nonempty matrix")
    if bits not in (3, 4, 8):
        raise ValueError("bits must be 3, 4 or 8")
    if not np.isfinite(alpha) or not 0 < alpha <= 1:
        raise ValueError("alpha must be finite and in (0,1]")
    qmax = (1 << (bits - 1)) - 1
    peaks = np.max(np.abs(w), axis=1) * np.float32(alpha)
    minimum_scale = np.nextafter(np.float32(0), np.float32(1))
    scales = np.where(peaks > 0, np.maximum(peaks / np.float32(qmax), minimum_scale), np.float32(1)).astype(np.float32)
    values = np.clip(np.rint(w / scales[:, None]), -qmax, qmax).astype(np.int8)
    return QuantRow(values, scales, bits, float(alpha))


def pack(values, bits):
    a = finite(values, "quantized values")
    if bits not in (3, 4, 8):
        raise ValueError("unsupported bit width")
    qmax = (1 << (bits - 1)) - 1
    if not np.all(a == np.rint(a)) or np.any(a < -qmax) or np.any(a > qmax):
        raise ValueError("quantized values out of symmetric range")
    codes = (a.astype(np.int16).reshape(-1) + qmax).astype(np.uint8)
    bitmatrix = ((codes[:, None] >> np.arange(bits, dtype=np.uint8)) & 1).astype(np.uint8)
    return np.packbits(bitmatrix.reshape(-1), bitorder="little")


def unpack(payload, count, bits):
    if bits not in (3, 4, 8) or not isinstance(count, (int, np.integer)) or count <= 0:
        raise ValueError("invalid packed dimensions")
    data = np.asarray(payload)
    if data.dtype != np.uint8 or data.ndim != 1 or len(data) != (int(count)*bits+7)//8:
        raise ValueError("packed payload must have exact uint8 byte length")
    raw = np.unpackbits(data, bitorder="little")
    if np.any(raw[count*bits:]):
        raise ValueError("nonzero padding bits")
    codes = (raw[:count*bits].reshape(count, bits).astype(np.uint16) * (1 << np.arange(bits, dtype=np.uint16))).sum(axis=1)
    qmax = (1 << (bits-1)) - 1
    if np.any(codes > 2*qmax):
        raise ValueError("reserved packed code")
    return (codes.astype(np.int16) - qmax).astype(np.int8)


def reconstructed_model(model, q1, q2):
    return {"w1": q1.reconstruct(), "w2": q2.reconstruct(), "b1": model["b1"], "b2": model["b2"]}


def choose_weight(model, bits, alphas=ALPHAS):
    rows, trace = [], []
    for key in ("w1", "w2"):
        candidates = [quantize_rows(model[key], bits, alpha) for alpha in alphas]
        errors = [float(np.mean(np.square(c.reconstruct().astype(np.float64) - model[key]))) for c in candidates]
        best = int(np.argmin(errors))
        rows.append(candidates[best])
        trace.append({"layer": key, "alphas": list(alphas), "weight_mse": errors, "chosen_alpha": float(alphas[best])})
    return rows[0], rows[1], trace


def choose_calibration(model, calibration_x, bits, alphas=ALPHAS):
    """The selection interface deliberately has no test-data or labels argument."""
    calibration_x = as_float32(calibration_x, "calibration inputs")
    if len(calibration_x) == 0:
        raise ValueError("calibration partition cannot be empty")
    reference = logits(model, calibration_x)
    # Evaluate identical clipping candidates; exhaustive 12x12 two-layer grid.
    # Every candidate minimizes final output-logit MSE on calibration only.
    candidates1 = [quantize_rows(model["w1"], bits, a) for a in alphas]
    candidates2 = [quantize_rows(model["w2"], bits, a) for a in alphas]
    best = (float("inf"), 0, 0)
    grid = []
    for i, q1 in enumerate(candidates1):
        for j, q2 in enumerate(candidates2):
            pred = logits(reconstructed_model(model, q1, q2), calibration_x)
            error = float(np.mean(np.square(pred.astype(np.float64) - reference)))
            grid.append({"alpha1": float(alphas[i]), "alpha2": float(alphas[j]), "calibration_logit_mse": error})
            if error < best[0]:
                best = (error, i, j)
    return candidates1[best[1]], candidates2[best[2]], grid


def model_archive(model, q1=None, q2=None):
    arrays = {"b1": model["b1"].astype(np.float32), "b2": model["b2"].astype(np.float32)}
    if q1 is None:
        arrays.update({"w1": model["w1"].astype(np.float32), "w2": model["w2"].astype(np.float32)})
        info = {"kind": "float32", "version": VERSION}
    else:
        for key, row in (("w1", q1), ("w2", q2)):
            arrays[key+"_packed"] = pack(row.values, row.bits)
            arrays[key+"_scales"] = row.scales.astype(np.float32)
        info = {"kind": "packed", "version": VERSION, "bits": q1.bits,
                "w1_shape": list(q1.values.shape), "w2_shape": list(q2.values.shape),
                "alpha1": q1.alpha, "alpha2": q2.alpha}
    arrays["metadata"] = np.frombuffer(json.dumps(info, sort_keys=True).encode(), dtype=np.uint8)
    return arrays


def archive_info(arrays):
    info = json.loads(arrays["metadata"].tobytes().decode())
    if info.get("kind") not in ("float32", "packed"):
        raise ValueError("unknown archive kind")
    return info


def validate_archive(arrays):
    info = archive_info(arrays)
    for key in ("b1", "b2"):
        if arrays[key].dtype != np.float32:
            raise ValueError("archive biases must use float32")
    if info["kind"] == "float32":
        expected = {"w1", "w2", "b1", "b2", "metadata"}
        if set(arrays) != expected:
            raise ValueError("unexpected float archive fields")
        w1, w2 = finite(arrays["w1"]), finite(arrays["w2"])
        if w1.dtype != np.float32 or w2.dtype != np.float32 or w1.ndim != 2 or w2.ndim != 2:
            raise ValueError("archive weights must be float32 matrices")
    else:
        expected = {"w1_packed", "w2_packed", "w1_scales", "w2_scales", "b1", "b2", "metadata"}
        if set(arrays) != expected:
            raise ValueError("unexpected quantized archive fields")
        for key in ("w1", "w2"):
            shape = info[key+"_shape"]
            if len(shape) != 2 or any(not isinstance(v, int) or v <= 0 for v in shape):
                raise ValueError("invalid matrix shape")
            scale = finite(arrays[key+"_scales"])
            if scale.dtype != np.float32 or scale.shape != (shape[0],) or np.any(scale <= 0):
                raise ValueError("invalid scales")
            unpack(arrays[key+"_packed"], int(np.prod(shape)), info["bits"])
        w1, w2 = info["w1_shape"], info["w2_shape"]
    sh1 = w1.shape if isinstance(w1, np.ndarray) else tuple(w1)
    sh2 = w2.shape if isinstance(w2, np.ndarray) else tuple(w2)
    if sh1[0] != sh2[1] or arrays["b1"].shape != (sh1[0],) or arrays["b2"].shape != (sh2[0],):
        raise ValueError("incompatible layer shapes")
    finite(arrays["b1"])
    finite(arrays["b2"])
    return info


def save_model(path, arrays):
    validate_archive(arrays)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)


def load_model(path):
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: archive[key].copy() for key in archive.files}
    validate_archive(arrays)
    return arrays


def archive_logits(arrays, x):
    """In-memory packed inference includes unpacking/dequantizing every call."""
    info = archive_info(arrays)
    if info["kind"] == "float32":
        return logits(arrays, x)
    model = {"b1": arrays["b1"], "b2": arrays["b2"]}
    for key in ("w1", "w2"):
        shape = info[key+"_shape"]
        values = unpack(arrays[key+"_packed"], int(np.prod(shape)), info["bits"]).reshape(shape)
        model[key] = values.astype(np.float32) * arrays[key+"_scales"][:, None]
    return logits(model, x)


def timing(arrays, x, repeats=41):
    for _ in range(5):
        archive_logits(arrays, x)
    elapsed = []
    for _ in range(repeats):
        start = time.perf_counter_ns()
        archive_logits(arrays, x)
        elapsed.append((time.perf_counter_ns() - start) / 1e6)
    return {"median_ms": float(np.median(elapsed)), "p25_ms": float(np.percentile(elapsed,25)),
            "p75_ms": float(np.percentile(elapsed,75)), "repeats": repeats,
            "batch_samples": len(x), "scope": "in-memory complete batch forward pass; packed includes unpack + dequantize + float32 matmul; disk loading excluded"}


def json_write(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def prepare_seed(out, seed, epochs=350):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    x, y, provenance = load_data()
    tr, cal, te = split_indices(y, seed)
    np.savez(out / "split.npz", train=tr, calibration=cal, test=te)
    start = time.perf_counter()
    model, history = train_mlp(x[tr], y[tr], seed, epochs=epochs)
    save_model(out / "float32.npz", model_archive(model))
    metadata = {"seed": seed, "epochs": epochs, "hidden": 48, "architecture": [64,48,10],
                "dtype": "float32", "optimizer": "NumPy Adam", "learning_rate": ".008, multiplied by .35 after 70% of fixed epochs",
                "l2": .0001, "training_seconds": time.perf_counter()-start,
                "training_seconds_scope": "MLP training and baseline NPZ serialization",
                "split_sizes": {"train":len(tr), "calibration":len(cal), "test":len(te)},
                "split_sha256": hashlib.sha256(tr.tobytes()+cal.tobytes()+te.tobytes()).hexdigest(),
                "data":provenance, "history":history,
                "baseline_train_accuracy":accuracy(logits(model,x[tr]),y[tr]),
                "baseline_calibration_accuracy":accuracy(logits(model,x[cal]),y[cal]),
                "baseline_test_accuracy":accuracy(logits(model,x[te]),y[te]),
                "selection_policy":"Fixed training epochs; no test or calibration early stopping. Calibration used only for clipping selection. Test used only for final evaluation."}
    json_write(out / "metadata.json", metadata)
    return metadata


def experiment_seed(out):
    out = Path(out)
    x,y,provenance = load_data()
    metadata = json.loads((out/"metadata.json").read_text(encoding="utf-8"))
    if metadata["data"]["normalized_data_sha256"] != provenance["normalized_data_sha256"]:
        raise ValueError("dataset differs from training fingerprint")
    with np.load(out/"split.npz",allow_pickle=False) as split:
        tr,cal,te = split["train"],split["calibration"],split["test"]
    assert_disjoint(tr,cal,te,n=len(y))
    arrays = load_model(out/"float32.npz")
    model = {k:arrays[k] for k in ("w1","w2","b1","b2")}
    reference = logits(model,x[te])
    float_payload = sum(arrays[k].nbytes for k in ("w1","w2","b1","b2"))
    float_weights = model["w1"].nbytes+model["w2"].nbytes
    baseline_timing = timing(arrays,x[te])
    records = []
    records.append({"seed":metadata["seed"],"method":"float32","bits":32,"alpha1":1.,"alpha2":1.,
                    "accuracy":accuracy(reference,y[te]),"accuracy_delta_pp":0.,"test_logit_mse":0.,
                    "calibration_logit_mse":0.,"weight_mse":0.,"packed_weight_bytes":float_weights,
                    "payload_bytes":float_payload,"npz_file_bytes":(out/"float32.npz").stat().st_size,
                    "weight_compression":1.,"payload_compression":1.,"file_compression":1.,
                    "median_ms":baseline_timing["median_ms"],"latency_ratio_vs_float":1.,
                    "timing":baseline_timing,"model_path":str((out/"float32.npz").name)})
    trace={}
    for bits in (8,4,3):
        for method in ("weight-mse","calibration-logits"):
            if method=="weight-mse":
                q1,q2,selection=choose_weight(model,bits)
            else:
                q1,q2,selection=choose_calibration(model,x[cal],bits)
            name=f"{method}-{bits}bit.npz"
            packed=model_archive(model,q1,q2)
            save_model(out/name,packed)
            loaded=load_model(out/name)
            expected=logits(reconstructed_model(model,q1,q2),x[te])
            actual=archive_logits(loaded,x[te])
            if not np.array_equal(expected,actual):
                raise AssertionError("serialized inference parity failed")
            size_weights=loaded["w1_packed"].nbytes+loaded["w2_packed"].nbytes
            payload=sum(v.nbytes for k,v in loaded.items() if k!="metadata")
            elapsed=timing(loaded,x[te])
            pred_cal=archive_logits(loaded,x[cal])
            calibration_reference=logits(model,x[cal])
            error=float(np.mean(np.square(actual.astype(np.float64)-reference)))
            weight_error=float(np.mean(np.concatenate([
                (q1.reconstruct().astype(np.float64)-model["w1"]).reshape(-1)**2,
                (q2.reconstruct().astype(np.float64)-model["w2"]).reshape(-1)**2])))
            acc=accuracy(actual,y[te])
            record={"seed":metadata["seed"],"method":method,"bits":bits,"alpha1":q1.alpha,"alpha2":q2.alpha,
                    "accuracy":acc,"accuracy_delta_pp":100*(acc-records[0]["accuracy"]),
                    "test_logit_mse":error,
                    "calibration_logit_mse":float(np.mean(np.square(pred_cal.astype(np.float64)-calibration_reference))),
                    "weight_mse":weight_error,"packed_weight_bytes":size_weights,"payload_bytes":payload,
                    "npz_file_bytes":(out/name).stat().st_size,
                    "weight_compression":float_weights/size_weights,"payload_compression":float_payload/payload,
                    "file_compression":(out/"float32.npz").stat().st_size/(out/name).stat().st_size,
                    "median_ms":elapsed["median_ms"],"latency_ratio_vs_float":elapsed["median_ms"]/baseline_timing["median_ms"],
                    "timing":elapsed,"model_path":name}
            records.append(record)
            trace[f"{method}-{bits}bit"]=selection
    json_write(out/"selection-trace.json",trace)
    json_write(out/"metrics.json",records)
    return records


def aggregate(records):
    result=[]
    for method,bits in [("float32",32)]+[(m,b) for b in (8,4,3) for m in ("weight-mse","calibration-logits")]:
        group=[r for r in records if r["method"]==method and r["bits"]==bits]
        row={"method":method,"bits":bits,"seeds":[r["seed"] for r in group]}
        for key in ("accuracy","accuracy_delta_pp","test_logit_mse","calibration_logit_mse","weight_mse","packed_weight_bytes",
                    "payload_bytes","npz_file_bytes","weight_compression","payload_compression","file_compression",
                    "median_ms","latency_ratio_vs_float"):
            values=[r[key] for r in group]
            row[key+"_mean"]=float(np.mean(values))
            row[key+"_std"]=float(np.std(values,ddof=1)) if len(values)>1 else 0.
        result.append(row)
    return result


def write_report(out, records, metadata):
    out=Path(out)
    for item in metadata:
        item.setdefault("training_seconds_scope", "MLP training and baseline NPZ serialization")
        json_write(out/f"seed-{item['seed']}"/"metadata.json",item)
    aggregated=aggregate(records)
    try:
        from threadpoolctl import threadpool_info
        pools = threadpool_info()
    except ImportError:
        pools = []
    payload={"tool":"TinyQuant Lab","version":VERSION,"created_at_utc":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),
             "runtime":{"python":sys.version.split()[0],"numpy":np.__version__,"platform":platform.platform(),
                        "processor":platform.processor(),"blas_threads":{k:os.environ.get(k) for k in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS")},
                        "loaded_threadpools":pools},
             "method":{"architecture":[64,48,10],"quantization":"row-wise symmetric weights; biases/scales float32; input/activations/matmul float32",
                       "packing":"offset-binary codes, least-significant-bit first, packed into uint8; qmax=2^(bits-1)-1; reserved all-ones code",
                       "clipping_candidates":ALPHAS,"weight_objective":"per-layer weight MSE over 12 candidates",
                       "calibration_objective":"final output-logit MSE over 12x12 two-layer clipping grid",
                       "test_policy":"No test samples or labels participate in training or clipping selection",
                       "timing":"five warmups, 41 complete forwards, median and IQR; no file IO; packed unpack/dequantize included; "+records[0]["timing"].get("scheme","sequential per-model measurements"),
                       "storage":"packed_weight_bytes: actual two packed arrays; payload_bytes: packed weights+float32 scales+biases; npz_file_bytes: actual uncompressed NPZ file including headers/metadata",
                       "reference":REFERENCE_URL},
             "seed_metadata":metadata,"records":records,"aggregate":aggregated}
    json_write(out/"results.json",payload)
    keys=[k for k in records[0] if k not in ("timing",)]
    with (out/"metrics.csv").open("w",encoding="utf-8",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader()
        writer.writerows({k:r[k] for k in keys} for r in records)
    rows=""
    for r in aggregated:
        rows+=f'<tr><td>{html.escape(r["method"])}</td><td>{r["bits"]}</td><td>{100*r["accuracy_mean"]:.2f}% <small>±{100*r["accuracy_std"]:.2f}</small></td><td>{r["test_logit_mse_mean"]:.4f}</td><td>{r["packed_weight_bytes_mean"]:,.0f}</td><td>{r["payload_bytes_mean"]:,.0f}</td><td>{r["npz_file_bytes_mean"]:,.0f}</td><td>{r["payload_compression_mean"]:.2f}×</td><td>{r["median_ms_mean"]:.3f}</td><td>{r["latency_ratio_vs_float_mean"]:.2f}×</td></tr>'
    seed_label=" · ".join(str(m["seed"]) for m in metadata)
    cards=f'<div class="card"><b>{len(metadata)} seeds</b><span>{seed_label}</span></div><div class="card"><b>1,797 digits</b><span>64 input pixels · 10 classes</span></div><div class="card"><b>3,552 weights</b><span>64 → 48 → 10 · NumPy training</span></div>'
    doc=f"""<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>TinyQuant Lab — CPU quantization results</title><style>
:root{{--bg:#0c1220;--panel:#142034;--line:#2c3b52;--ink:#e7edf6;--muted:#9eafc7;--accent:#7dd3fc}}*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:16px/1.65 system-ui,sans-serif}}main{{max-width:1240px;margin:0 auto;padding:64px 30px}}.eyebrow{{color:var(--accent);letter-spacing:.14em;font-size:12px;text-transform:uppercase}}h1{{font-size:clamp(32px,5vw,58px);line-height:1.12;margin:15px 0 20px}}h2{{font-size:24px;margin-top:46px}}p{{max-width:850px;color:var(--muted)}}.cards{{display:flex;gap:14px;margin:30px 0}}.card{{background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:22px;flex:1}}.card b{{display:block;font-size:26px}}.card span,small{{color:var(--muted);font-size:13px}}.scroll{{overflow:auto;border:1px solid var(--line);border-radius:14px}}table{{border-collapse:collapse;width:100%;font-size:14px;white-space:nowrap}}th,td{{padding:16px 12px;text-align:right;border-bottom:1px solid var(--line)}}th{{background:var(--panel);color:var(--muted);font-size:12px}}th:first-child,td:first-child{{text-align:left}}td:first-child{{color:var(--accent)}}a{{color:var(--accent)}}.note{{padding:22px;border-left:3px solid var(--accent);background:var(--panel)}}.split{{display:grid;grid-template-columns:1fr 1fr;gap:28px}}footer{{margin-top:45px;padding-top:24px;border-top:1px solid var(--line);font-size:13px;color:var(--muted)}}@media(max-width:700px){{main{{padding:32px 18px}}.cards{{flex-direction:column}}.split{{grid-template-columns:1fr}}}}</style>
<main><div class="eyebrow">Experiment / October 2026 / CPU</div><h1>Small weights.<br>Measured tradeoffs.</h1><p>TinyQuant Lab trains a digit classifier, packs its weights into 8-, 4- and 3-bit arrays, then measures accuracy, file size and the cost of unpacking before inference.</p><div class="cards">{cards}</div>
<h2>{len(metadata)}-seed results</h2><p>Accuracy shows the mean ± sample standard deviation across seeds. Latency is the mean of {len(metadata)} per-seed medians for a complete test batch. A latency ratio above 1 means a slower forward pass.</p>
<div class="scroll"><table><thead><tr><th>Selection objective</th><th>Bits</th><th>Test accuracy</th><th>Logit MSE</th><th>Weight bytes</th><th>Payload bytes</th><th>NPZ bytes</th><th>Payload compression</th><th>Batch ms</th><th>Latency / float</th></tr></thead><tbody>{rows}</tbody></table></div>
<div class="split"><section><h2>Two clipping objectives</h2><p><b>Weight MSE</b> selects one clipping factor per layer by reconstruction error. <b>Calibration logits</b> selects both clipping factors by final output error on a separate calibration partition. Both use the same 12-factor grid from 0.45 to 1.00.</p></section><section><h2>Separated data</h2><p>Each seed uses a stratified train / calibration / test split. Training uses a fixed {metadata[0]["epochs"]}-epoch schedule. Test samples enter only the final evaluation. Split indices and data fingerprints accompany every model.</p></section></div>
<h2>What the byte counts measure</h2><p>Weight bytes count the actual packed arrays. Payload bytes also include float32 scales and biases. NPZ bytes measure the saved file with headers and metadata. The runtime expands packed weights to float32 for NumPy matrix multiplication, so stored compression and forward-pass speed are separate measurements.</p>
<div class="note">Batch inference uses the complete test partition: {metadata[0]["split_sizes"]["test"]} samples per call. Timing includes unpacking and dequantization, excludes disk loading, and records 41 measured forwards after five warmups. Measurement order: {html.escape(records[0]["timing"].get("scheme","sequential per-model measurements"))}. BLAS thread budget: 1.</div>
<h2>Reproduce the experiment</h2><p>Run <code>python tinyquant.py experiment --output results --seeds 17 43 91</code>. The directory contains packed NPZ models, split identities, selection traces, JSON metrics and CSV records.</p>
<footer>Dataset: <a href="{DATASET_URL}">scikit-learn load_digits</a> · Concept reference: <a href="{REFERENCE_URL}">GuidedQuant, ICML 2025</a>. This independent small-model experiment compares two clipping objectives; it uses float32 NumPy inference. <a href="results.json">Full numerical record</a> · <a href="metrics.csv">CSV</a></footer></main></html>"""
    (out/"report.html").write_text(doc,encoding="utf-8")
    return payload


def main(argv=None):
    parser=argparse.ArgumentParser(description="Train, pack and measure a small NumPy classifier.")
    parser.add_argument("--version",action="version",version=VERSION)
    sub=parser.add_subparsers(dest="command",required=True)
    tr=sub.add_parser("train",help="train one seed with fixed epochs")
    tr.add_argument("--output",type=Path,required=True);tr.add_argument("--seed",type=int,default=17);tr.add_argument("--epochs",type=int,default=350)
    ex=sub.add_parser("experiment",help="train three seeds, compare objectives, write measured report")
    ex.add_argument("--output",type=Path,required=True);ex.add_argument("--seeds",type=int,nargs="+",default=SEEDS);ex.add_argument("--epochs",type=int,default=350)
    run=sub.add_parser("run",help="run a stored model on a .npy matrix")
    run.add_argument("--model",type=Path,required=True);run.add_argument("--input",type=Path,required=True);run.add_argument("--output",type=Path,required=True)
    args=parser.parse_args(argv)
    if args.command=="train":
        if args.epochs<=0:parser.error("--epochs must be positive")
        print(json.dumps(prepare_seed(args.output,args.seed,args.epochs)["split_sizes"]))
    elif args.command=="experiment":
        if args.epochs<=0 or len(set(args.seeds))!=len(args.seeds):parser.error("positive epochs and unique seeds required")
        args.output.mkdir(parents=True,exist_ok=True)
        records=[];metadata=[]
        for seed in args.seeds:
            folder=args.output/f"seed-{seed}"
            meta=prepare_seed(folder,seed,args.epochs);metadata.append(meta)
            records.extend(experiment_seed(folder))
            print(f"seed {seed}: baseline test accuracy {meta['baseline_test_accuracy']:.4f}",flush=True)
        payload=write_report(args.output,records,metadata)
        print(json.dumps(payload["aggregate"],indent=2))
    else:
        if args.output.resolve() in (args.model.resolve(),args.input.resolve()) or any(args.output.exists() and args.output.samefile(p) for p in (args.model,args.input)):
            parser.error("output cannot overwrite input or model")
        x=np.load(args.input,allow_pickle=False)
        pred=archive_logits(load_model(args.model),x)
        args.output.parent.mkdir(parents=True,exist_ok=True)
        with args.output.open("wb") as output_file:
            np.save(output_file,pred)
        print(json.dumps({"samples":len(pred),"classes":pred.shape[1]}))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
