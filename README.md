# TinyQuant Lab

[Recorded experiment and validation](docs/validation.md) · [Korean experiment report](docs/Model_Quantization_and_Evaluation_Report_KO.pdf) · [Citation metadata](CITATION.cff)

Train a small digit classifier and measure what changes when its weights are packed into 8, 4 or 3 bits. Compare clipping selected by weight reconstruction error with clipping selected by final output error on a separate calibration partition.

NumPy implements the model, training gradients, quantization, bit packing and inference. The dataset is the official bundled digits file from scikit-learn 1.9.1, cached and parsed directly with NumPy. All work runs on a CPU.

## Install and run

Python 3.12 was used for the recorded experiment.

~~~powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe cache_dataset.py
.\.venv\Scripts\python.exe -m unittest -v test_tinyquant
.\.venv\Scripts\python.exe tinyquant.py experiment --output reproduced-results --seeds 17 43 91
.\.venv\Scripts\python.exe rebenchmark.py reproduced-results
.\.venv\Scripts\python.exe verify_results.py reproduced-results
~~~

On Linux or macOS use .venv/bin/python. The default schedule trains 350 epochs per seed.

The reproduced-results directory contains results.json with runtime, method, provenance and numerical records; metrics.csv; a read-only standalone report.html; and one seed directory with a float32 NPZ model, six packed NPZ models, exact split indices, metadata, training history, candidate search traces and seed metrics. The bundled results directory keeps the recorded experiment.

Train one baseline:

~~~powershell
.\.venv\Scripts\python.exe tinyquant.py train --output baseline --seed 17
~~~

Run a saved model on an existing input matrix:

~~~powershell
.\.venv\Scripts\python.exe tinyquant.py run --model results\seed-17\calibration-logits-4bit.npz --input input.npy --output logits.npy
~~~

Input is a sample-by-64 NumPy matrix with pixel values divided by 16. Output is sample-by-10 logits. The CLI writes to the exact output path and protects the input matrix and model from path and hard-link aliases.

## Model and data

The network uses 64 inputs, 48 ReLU hidden units and 10 classes: 3,552 weights and 58 biases. NumPy backpropagation trains float32 weights with Adam and cross entropy plus half-L2 regularization. The fixed training schedule uses only training samples.

The [scikit-learn digits dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html) contains 1,797 samples and 64 pixel features. Its upstream source is [UCI Optical Recognition of Handwritten Digits](https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits). The experiment uses per-class random partitions of the complete bundled data rather than the official UCI train/test partition.

Within each class, 65% is allocated to training, 15% to calibration and the remainder to testing; rounding happens per class. Only train samples update weights. Calibration inputs select clipping for the calibration-logits objective. Test samples enter only final evaluation. Each seed saves partition indices and a dataset SHA-256.

## Two clipping objectives

For each row, the clipping limit is alpha times the maximum absolute weight. Scale is the limit divided by 2^(bits-1)-1. Quantization uses nearest-integer rounding and the symmetric signed range. Zero rows have scale 1. Tiny positive scales are clamped to the smallest float32 subnormal.

Both methods search 12 clipping factors from 0.45 to 1.00:

| Method | Selection |
|---|---|
| weight-mse | Select each layer's factor independently by weight reconstruction MSE. |
| calibration-logits | Search all 144 factor pairs and minimize final output-logit MSE on calibration inputs. |

The logit objective preserves the float32 model's output and does not use calibration labels. The calibration search executes the model for 144 candidates; weight MSE examines weight arrays directly.

## Packed storage and timing

Signed values shift by qmax into offset-binary codes, then write least-significant-bit first into uint8 arrays. Actual 4-bit nibble packing and 3-bit codes spanning byte boundaries are included. Reserved all-ones codes and nonzero padding bits are rejected.

| Field | Measurement |
|---|---|
| packed_weight_bytes | Actual two packed weight arrays, or two float32 arrays for baseline. |
| payload_bytes | Weights, float32 row scales where needed, and float32 biases. |
| npz_file_bytes | Actual uncompressed NPZ file, including archive headers and metadata. |

Compression ratios use the matching float32 baseline as numerator. Byte counts describe stored model arrays and files. During inference, packed weights expand to float32 for NumPy matrix multiplication. Inputs, activations and biases remain float32.

Timing covers a complete forward on the test partition. Packed timing includes unpacking and dequantization on every call; disk reading is excluded. Five warmups precede 41 measured calls. The final measurement interleaves all 21 models in randomized round-robin order with RNG seed 718 to reduce host-load drift. Raw samples, median and 25th/75th percentile milliseconds are recorded. BLAS thread variables are set to 1 before NumPy import; actual loaded threadpools are also recorded. A latency ratio above 1 means packed inference took longer than float32. Initial sequential measurements remain in timing-initial.json.

## Verification

Unit tests cover finite-difference gradients, split completeness and overlap rejection, guarded calibration-only selection, bounds, zero rows, overflow, tiny subnormal scales, signed/odd-count packing, independently expected bytes, malformed packed codes, exact serialization parity and CLI input protection.

The independent verify_results.py decodes packed models using Python integer shifts, recomputes logits and accuracy, checks file and payload sizes, verifies split identities and audits the selected calibration grid minimum.

## Concept reference

[GuidedQuant: Large Language Model Quantization via Exploiting End Loss Guidance](https://proceedings.mlr.press/v267/kim25d.html), ICML 2025, motivates examining an end-output objective. TinyQuant Lab implements an independent two-layer classifier experiment with a clipping grid and final-logit MSE. Its result is a local NumPy experiment with trained models and measured small-model tradeoffs.

## License

MIT for this project source. Dataset attribution and distribution terms follow the linked scikit-learn and UCI sources.
