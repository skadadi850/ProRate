#!/usr/bin/env bash
# Evaluates the seven ProRate checkpoints and writes the seed 1 figures to outputs/. Logs go to logs/.
#   bash run.sh        (PYMOL=/path/to/pymol if pymol is not on PATH)
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"
PY="${PY:-python}"
PYMOL="${PYMOL:-$(command -v pymol || true)}"
DEVICE="${DEVICE:-cuda}"
export PYTHONPATH=$PWD OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TQDM_DISABLE=1
R=release
O=outputs
mkdir -p logs "$O/eval"

[ -d "$R" ] || { echo "release/ is missing: run bash download.sh"; exit 1; }
[ -n "$PYMOL" ] || { echo "pymol not found: set PYMOL to its path"; exit 1; }
! grep -rn --include='*.py' foldingdiff scripts prorate || { echo "foldingdiff reference left in the sources"; exit 1; }

ARMS=(protrig_s8k_rw0p4_s0 protrig_s8k_rw0p2_s1 protrig_s8k_rw0p1_s0 protrig_s8k_rw0p06_s1
      protrig_s8k_rw0p03_s1 protrig_s8k_rw0p015_s0 protrig_s8k_rw0p06_link3_s1)
FREE=protrig_s8k_rw0p015_s0
VALID=protrig_s8k_rw0p06_link3_s1

step() {
  local name=$1; shift
  echo "$(date +%T) start $name"
  "$@" > "logs/$name.log" 2>&1 || { echo "FAILED $name: see logs/$name.log"; exit 1; }
}

EVALS=()
for arm in "${ARMS[@]}"; do
  EVALS+=("$O/eval/$arm/eval.json")
  [ -s "$O/eval/$arm/eval.json" ] && continue
  rw=$($PY -c "import sys, torch; print(torch.load(sys.argv[1], map_location='cpu', weights_only=False)['args']['rate_weight'])" "$R/checkpoints/$arm.pt")
  step "eval_$arm" $PY scripts/figures.py eval --ckpt "$R/checkpoints/$arm.pt" --arm "$arm" --rate-weight "$rw" \
      --rate-chains 1500 --train-dir "$R/data/train" --test cameo="$R/data/cameo" casp14="$R/data/casp14" \
      --batch-size 64 --device "$DEVICE" --out "$O/eval/$arm/eval.json"
done

step panel $PY scripts/figures.py panel --protrig "${EVALS[@]}" --geobpe "$R/cache/geobpe_paper_points.json" \
    --length-free "$O/eval/$FREE/eval_per_chain.json" --length-valid "$O/eval/$VALID/eval_per_chain.json" \
    --length-geobpe "$R/cache/geobpe_quality/geobpe_K600.json" \
    --length-protoken "$R/cache/baselines/protoken.json" --length-aminoaseed "$R/cache/baselines/aminoaseed.json" \
    --baseline-dir "$R/cache/baselines" --speed-dir "$R/cache/speed" --geobpe-dir "$R/cache/speed/geobpe" \
    --out "$O/fig_panel_seed1"

step structures $PY scripts/fig_structures_pymol.py --pymol "$PYMOL" --device "$DEVICE" \
    --test cameo="$R/data/cameo" casp14="$R/data/casp14" \
    --ckpt "$R/checkpoints/$FREE.pt" --valid-ckpt "$R/checkpoints/$VALID.pt" \
    --geobpe-decoded "$R/cache/geobpe_decoded" --baseline-dir "$R/cache/baselines" \
    --protrig-bpr 39.5 --valid-bpr 36.2 --geobpe-bpr 36.1 --protoken-bpr 133.0 --aminoaseed-bpr 398.0 \
    --select-worst-closeup --protrig-max-rmsd 0.5 --geobpe-min-link-error 2.0 --num 6 --seed 0 \
    --out "$O/fig_structures_pymol_seed1"

# the close-up rewrites the structures manifest and re-renders the structures figure with the red squares
step closeup $PY scripts/fig_boundary_closeup.py --pymol "$PYMOL" --device "$DEVICE" \
    --manifest "$O/fig_structures_pymol_seed1.manifest.json" --baseline-dir "$R/cache/baselines" \
    --num 6 --links 1 --scatter-chains 100 --out "$O/fig_boundary_closeup_seed1"

step token_efficiency $PY scripts/fig_token_efficiency.py --device "$DEVICE" --out "$O/fig_token_efficiency"

for f in fig_panel_seed1 fig_structures_pymol_seed1 fig_boundary_closeup_seed1 fig_boundary_closeup_seed1_scatter \
         fig_boundary_closeup_seed1_zoom fig_token_efficiency; do
  [ -s "$O/$f.png" ] || { echo "MISSING $O/$f.png"; exit 1; }
done
echo "$(date +%T) wrote $O/*.png"