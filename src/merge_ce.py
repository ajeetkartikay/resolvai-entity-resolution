"""Average the cross-encoder scores of several stack.py runs (same rows) into ens_train_ce / ens_test_ce."""
import sys, glob, numpy as np, pandas as pd
root = sys.argv[1] if len(sys.argv) > 1 else "."
out = sys.argv[2] if len(sys.argv) > 2 else "ens"
import os; os.makedirs(out, exist_ok=True)
for kind in ["train_ce", "test_ce"]:
    files = sorted(f for f in glob.glob(f"{root}/**/{kind}.parquet", recursive=True) if "smoke" not in f)
    print(kind, files)
    base = None
    for f in files:
        d = pd.read_parquet(f).sort_values("row").reset_index(drop=True)
        if base is None:
            base, acc = d.row.values, d.ce.values.astype(np.float64)
        else:
            assert len(d) == len(base) and (d.row.values == base).all(), f"row mismatch in {f}"
            acc += d.ce.values
    pd.DataFrame({"row": base, "ce": (acc / len(files)).astype(np.float32)}).to_parquet(f"{out}/ens_{kind}.parquet", index=False)
    print("wrote", f"{out}/ens_{kind}.parquet", len(base), "rows from", len(files), "models")
