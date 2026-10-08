"""Cache the official scikit-learn data file without installing SciPy."""
from pathlib import Path
import hashlib
import json
import urllib.request

url = "https://raw.githubusercontent.com/scikit-learn/scikit-learn/1.9.1/sklearn/datasets/data/digits.csv.gz"
path = Path(__file__).parent / "dataset" / "digits.csv.gz"
path.parent.mkdir(parents=True, exist_ok=True)
payload = urllib.request.urlopen(url, timeout=60).read()
path.write_bytes(payload)
record = {"source_url": url, "source_release": "scikit-learn 1.9.1", "sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload)}
(path.parent / "provenance.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
print(json.dumps(record))
