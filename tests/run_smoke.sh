#!/usr/bin/env bash
# End-to-end smoke test on synthetic data (CPU, < 1 minute). The cross-encoder is replaced by a
# fuzzy-match stand-in (--mock) so no GPU or model download is needed.
set -euo pipefail
cd "$(dirname "$0")/.."
D=$(mktemp -d)
python tests/make_synthetic.py "$D/data"
python src/baseline_pipeline.py --data_root "$D/data" --split test  --out_dir "$D/data/blocking_test"
python src/baseline_pipeline.py --data_root "$D/data" --split train --s1_frac 0.5 --out_dir "$D/data/blocking_train"
python src/route_b.py --data_root "$D/data" --mode train --out_dir "$D/data/lgb" --fit_frac 0.3 --es_frac 0.1 --chunk 50000
python src/route_b.py --data_root "$D/data" --mode test  --out_dir "$D/data/lgb_test" --chunk 50000
python src/stack.py   --data_root "$D/data" --out_dir "$D/final" --mock
echo "OK: final outputs in $D/final"
