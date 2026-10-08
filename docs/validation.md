# Recorded experiment and verification

The 2026-10-08 run trains a 64 → 48 → 10 NumPy classifier on the bundled scikit-learn digits data. Seeds 17, 43 and 91 each use 1,164 training, 266 calibration and 367 test samples. Split indices, dataset fingerprints and clipping searches accompany the models.

| Variant | Mean test accuracy | Sample standard deviation | Stored model payload |
|---|---:|---:|---:|
| Float32 | 97.64% | 0.83 percentage points | 14,440 bytes |
| 4-bit calibration-logits | 97.46% | 1.03 percentage points | 2,240 bytes |
| 3-bit calibration-logits | 97.37% | 0.88 percentage points | 1,796 bytes |

The 4-bit payload includes 1,776 bytes of packed weights, 232 bytes of row scales and 232 bytes of biases. Its payload compression ratio is 6.45× relative to the float32 baseline. The mean saved 4-bit calibration NPZ file size is 4,106 bytes, versus 15,693 bytes for the baseline NPZ. These are model-storage measurements; inference expands the packed weights for float32 matrix multiplication.

## Checks

- 25 unit tests: gradient finite differences, split isolation, calibration-only selection, packing bounds and independently expected byte values, tiny/invalid values, serialization parity and input protection.
- 21 stored models: independent Python-integer decoding and matrix forward, exact classification accuracy, actual array/file bytes, recorded MSE within documented float32 tolerances, split identity and selected calibration-grid minimum.
- A separate implementation also checked all 21 models, seven aggregates, sample standard deviations and all raw timing medians/IQRs against the recorded results.

CI runs the unit suite and stored-model verifier on Ubuntu with Python 3.12 and NumPy 2.5.3. It reuses the bundled dataset and saved models, so it performs no training or dataset download.

## Read the timing record

Recorded timings come from the original Windows experiment. Each model uses five warmups and 41 complete forwards of 367 test samples. Measurement order is randomized round robin across all 21 models with RNG seed 718. Packed timings include bit unpacking, dequantization and float32 matrix multiplication; disk loading is excluded. Raw samples and host/BLAS details are in results/results.json. The published runtime record keeps library filenames and removes host directory paths.

## Sources and reports

- [Full numerical record](../results/results.json)
- [CSV](../results/metrics.csv)
- [Read-only HTML report](../results/report.html)
- [Korean experiment report covering TinyQuant Lab and EvalExecution Lab](Model_Quantization_and_Evaluation_Report_KO.pdf)
- [Official digits data documentation](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html)
- [GuidedQuant concept reference](https://proceedings.mlr.press/v267/kim25d.html)

The clipping grid and final-logit objective belong to this independent small-classifier experiment. The saved models, source and data support a local repeat of its measured results.
