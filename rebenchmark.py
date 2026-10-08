"""Re-measure all stored models in interleaved order to reduce host-load drift."""
import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import time
import numpy as np
import tinyquant as tq


def run(root,repeats=41):
    root=Path(root)
    data=json.loads((root/"results.json").read_text(encoding="utf-8"))
    records=data["records"]
    tq.json_write(root/"timing-initial.json",[{"seed":r["seed"],"method":r["method"],"bits":r["bits"],"timing":r["timing"]} for r in records])
    x,_,_=tq.load_data()
    tasks=[]
    for r in records:
        folder=root/f"seed-{r['seed']}"
        with np.load(folder/"split.npz",allow_pickle=False) as split:
            test=split["test"]
        tasks.append((tq.load_model(folder/r["model_path"]),x[test]))
    try:
        from threadpoolctl import threadpool_limits
        context=threadpool_limits(limits=1)
    except ImportError:
        context=nullcontext()
    rng=np.random.default_rng(718)
    durations=[[] for _ in tasks]
    with context:
        for _ in range(5):
            for i in rng.permutation(len(tasks)):
                tq.archive_logits(*tasks[i])
        for _ in range(repeats):
            for i in rng.permutation(len(tasks)):
                start=time.perf_counter_ns()
                tq.archive_logits(*tasks[i])
                durations[i].append((time.perf_counter_ns()-start)/1e6)
        for r,elapsed,(_,inputs) in zip(records,durations,tasks):
            r["median_ms"]=float(np.median(elapsed))
            r["timing"]={"median_ms":r["median_ms"],"p25_ms":float(np.percentile(elapsed,25)),
                         "p75_ms":float(np.percentile(elapsed,75)),"repeats":repeats,
                         "batch_samples":len(inputs),"scheme":f"randomized round-robin across all {len(tasks)} models; RNG 718",
                         "scope":"complete in-memory batch forward; packed unpack+dequantize+float32 matmul included; disk loading excluded",
                         "raw_ms":elapsed}
        for r in records:
            base=next(a for a in records if a["seed"]==r["seed"] and a["method"]=="float32")
            r["latency_ratio_vs_float"]=r["median_ms"]/base["median_ms"]
        final=tq.write_report(root,records,data["seed_metadata"])
    for meta in data["seed_metadata"]:
        tq.json_write(root/f"seed-{meta['seed']}"/"metrics.json",[r for r in records if r["seed"]==meta["seed"]])
    print(json.dumps([{"method":r["method"],"bits":r["bits"],"accuracy_mean":r["accuracy_mean"],
                      "median_ms_mean":r["median_ms_mean"],"latency_ratio_mean":r["latency_ratio_vs_float_mean"]}
                     for r in final["aggregate"]],indent=2))
    return final


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("results",type=Path)
    args=parser.parse_args()
    run(args.results)
