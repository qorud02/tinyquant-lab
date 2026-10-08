"""Audit stored experiment with independent bit decoding and matrix arithmetic."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import tinyquant as tq


def decode_reference(payload, shape, bits):
    blob=bytes(payload)
    qmax=2**(bits-1)-1
    numbers=[]
    for index in range(int(np.prod(shape))):
        start=index*bits
        code=0
        for bit in range(bits):
            code|=((blob[(start+bit)//8] >> ((start+bit)%8)) & 1) << bit
        if code>2*qmax:
            raise AssertionError("reserved code")
        numbers.append(code-qmax)
    return np.array(numbers,dtype=np.float32).reshape(shape)


def reference_logits(arrays,x):
    info=json.loads(arrays["metadata"].tobytes().decode())
    if info["kind"]=="float32":
        w1,w2=arrays["w1"],arrays["w2"]
    else:
        w1=decode_reference(arrays["w1_packed"],info["w1_shape"],info["bits"])*arrays["w1_scales"][:,None]
        w2=decode_reference(arrays["w2_packed"],info["w2_shape"],info["bits"])*arrays["w2_scales"][:,None]
    hidden=np.maximum(np.matmul(x,w1.T)+arrays["b1"],0)
    return np.matmul(hidden,w2.T)+arrays["b2"]


def verify(root):
    root=Path(root)
    results=json.loads((root/"results.json").read_text(encoding="utf-8"))
    x,y,prov=tq.load_data()
    checks=[]
    if len(results["records"])!=7*len(results["seed_metadata"]):
        raise AssertionError("missing model records")
    for meta in results["seed_metadata"]:
        folder=root/f"seed-{meta['seed']}"
        with np.load(folder/"split.npz",allow_pickle=False) as split:
            tr,cal,te=split["train"],split["calibration"],split["test"]
        joined=np.concatenate((tr,cal,te))
        assert sorted(joined.tolist())==list(range(len(y))),"split leakage/coverage"
        assert len(set(tr)&set(cal))==len(set(cal)&set(te))==len(set(tr)&set(te))==0
        assert hashlib.sha256(tr.tobytes()+cal.tobytes()+te.tobytes()).hexdigest()==meta["split_sha256"]
        assert meta["data"]["normalized_data_sha256"]==prov["normalized_data_sha256"]
        baseline=tq.load_model(folder/"float32.npz")
        float_pred=reference_logits(baseline,x[te])
        assert float(np.mean(float_pred.argmax(axis=1)==y[te]))>=.93,"insufficient trained baseline"
        traces=json.loads((folder/"selection-trace.json").read_text(encoding="utf-8"))
        for record in [r for r in results["records"] if r["seed"]==meta["seed"]]:
            path=folder/record["model_path"]
            model=tq.load_model(path)
            ref=reference_logits(model,x[te])
            actual=tq.archive_logits(model,x[te])
            assert np.array_equal(actual,ref),"independent decode/forward parity"
            assert abs(float(np.mean(ref.argmax(axis=1)==y[te]))-record["accuracy"])<1e-15
            assert path.stat().st_size==record["npz_file_bytes"]
            assert sum(v.nbytes for k,v in model.items() if k!="metadata")==record["payload_bytes"]
            recomputed_mse=float(np.mean((ref.astype(np.float64)-float_pred)**2))
            if abs(recomputed_mse-record["test_logit_mse"])>=1e-12:
                print(json.dumps({"seed":record["seed"],"method":record["method"],"bits":record["bits"],"recomputed_mse":recomputed_mse,"recorded_mse":record["test_logit_mse"],"difference":recomputed_mse-record["test_logit_mse"]}),flush=True)
            assert np.isclose(recomputed_mse,record["test_logit_mse"],rtol=1e-5,atol=1e-8),"recorded test MSE differs beyond float32 tolerance"
            if record["method"]!="float32":
                assert model["w1_packed"].nbytes+model["w2_packed"].nbytes==record["packed_weight_bytes"]
                info=json.loads(model["metadata"].tobytes().decode())
                for key in ("w1","w2"):
                    assert model[key+"_packed"].nbytes==(int(np.prod(info[key+"_shape"]))*record["bits"]+7)//8
            if record["method"]=="calibration-logits":
                grid=traces[f"calibration-logits-{record['bits']}bit"]
                selected=[r for r in grid if r["alpha1"]==record["alpha1"] and r["alpha2"]==record["alpha2"]][0]
                assert selected["calibration_logit_mse"]==min(r["calibration_logit_mse"] for r in grid)
                ref_cal=reference_logits(model,x[cal]);float_cal=reference_logits(baseline,x[cal])
                assert np.isclose(float(np.mean((ref_cal.astype(np.float64)-float_cal)**2)),selected["calibration_logit_mse"],rtol=1e-5,atol=1e-8),"recorded calibration MSE differs beyond float32 tolerance"
            checks.append({"seed":meta["seed"],"method":record["method"],"bits":record["bits"],"status":"PASS"})
    result={"status":"PASS","models_checked":len(checks),"checks":checks,
            "checks_per_model":["independent Python-integer decoding","independent matrix forward","exact prediction parity",
                                "accuracy recomputation","actual NPZ and payload bytes","test logit MSE"],
            "checks_per_seed":["three partitions disjoint/full coverage","split identity SHA","dataset SHA","float baseline >=93%"],
            "calibration_checks":"stored grid argmin and independently recomputed calibration output MSE",
            "mse_comparison_tolerance":{"relative":1e-5,"absolute":1e-8,"basis":"float32 matmul reconstruction may differ in last-bit reduction; label equality remains exact"}}
    (root/"VERIFICATION.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps(result,indent=2))
    return result


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("results",type=Path)
    args=parser.parse_args()
    verify(args.results)
