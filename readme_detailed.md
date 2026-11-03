# ProRate: detailed guide

This document describes everything in the repository and the release archive. The short version is in [readme.md](readme.md).

Naming note: the method is called ProRate in the paper. In the code, checkpoints and file names it is called ProtRig (the class is
`ProtRig`, arms are named `protrig_...`). They are the same thing.

## 1. Method in brief

- A chain is split into pieces: maximal runs of resolved residues joined by valid peptide bonds. A residue is resolved if N, CA and C are
  present. A link is a valid peptide bond if its C-N distance is 1.0 to 2.0 angstroms and the angles theta_CACN, theta_CNCA and theta_NCAC
  lie between 0.5 and 2.9 radians.
- Pieces of at least 3 residues are tokenized. Shorter pieces are stored raw as 9m-6 float32 internal coordinates. Later pieces are placed
  relative to the first by a six-value rigid transform fitted in closed loop. All of this is charged in the rate.
- Each tokenized piece is split into fragments of 1 to 16 residues (`l_max = 16`) by an exact dynamic program over the cost
  `3L * min_q RMSD(span, q)^2 + eta * L^gamma + lambda * (log2|V| + boundary bits)`. The codebooks are k-medoids over observed fragments
  under Kabsch RMSD, one codebook per length.
- Adjacent fragments are joined by a six-coordinate boundary cell (theta_CACN, l_CN, psi, theta_CNCA, omega, phi) on a uniform grid.
  Boundaries are fitted in a closed loop against the chain as the decoder will rebuild it, so rounding errors are corrected by the next
  boundary instead of accumulating.
- Decoding is forward kinematics from the integers: no network. Rate is bits per residue (BPR): `L(T)/N_D + sum(chain bits)/sum(residues)`
  with `L(T) = 32 * (sum_L K_L (9L-6) + 18)` bits amortized over the residues N_D of the fitting set.
- "free" arms leave the boundary cells unconstrained. The "valid" arm (`link3`) restricts the boundary ranges to those of natural links and,
  in the encoder, keeps CA-C-N, C-N-CA and omega within 2 sigma of the true link (sigma 2.0, 1.8 and 5.0 degrees) wherever the grid permits.

## 2. Repository layout

```
prorate/                  the codec, no trained network
  decoder.py              forward kinematics, rigid fitting, Engh and Huber defaults
  quantizer.py            per-length fragment codebooks (k-medoids), six-axis boundary quantizer, closed-form RMSD
  model.py                ProtRig: exact DP segmentation, closed-loop boundary quantization, token stream, rebuild from integers
  data.py                 PDB parsing, pieces, break transforms, rate accounting, datasets
  metrics.py              Kabsch RMSD, lDDT, peptide-link and omega reports
  checkpoint.py           load_protrig(): loads released or freshly trained checkpoints
scripts/
  figures.py              eval (one arm), panel (the main figure), table, geobpe-induce scoring
  fig_structures_pymol.py structures figure (PyMOL ray tracing)
  fig_boundary_closeup.py link close-ups, link-error distributions, zoom figure
  fig_token_efficiency.py utilization rate and normalized perplexity
  train_protrig.py        fits one codec (alternating segmentation and k-medoids)
download.sh               downloads, verifies (sha256) and unpacks the Zenodo archive into release/
run.sh                    evaluates the seven checkpoints and writes all figures
train.sh                  retrains one arm
requirements.txt          pinned Python packages
```

## 3. Requirements

- Python 3.10, with the packages in `requirements.txt`: `torch==2.6.0`, `numpy==1.26.4`, `matplotlib==3.10.1`, `esm==3.2.0`,
  `pillow==11.2.1`. `esm` is used only to parse PDB files and to compute the backbone RMSD and CA-lDDT in the form of the published
  baselines.
- PyMOL (open source) for the structures, close-up and zoom figures: `conda install -c conda-forge pymol-open-source`. Pass its path with
  `PYMOL=/path/to/pymol` if it is not on `PATH`.
- A CUDA GPU. `DEVICE=cuda` is the default. `DEVICE=cpu` is accepted by the scripts but has not been tested and will be slow.
- TM-score is not computed. The `tm` field of `eval.json` is NaN unless a `TMscore` binary is on `PATH`; no figure uses it.
- Disk: the archive is 296 MB; the 7,980 training chains take 1.2 GB once unpacked.
- Reference machine for the timings below: Intel Core i9-14900KF, 62 GiB RAM, NVIDIA RTX 4090 (24 GB).

## 4. The release archive

`bash download.sh` fetches `prorate_release.tar.gz` from Zenodo, checks its sha256 and unpacks it into `release/` (about the same sha256
is printed on the Zenodo record). Contents:

```
release/
  checkpoints/            7 ProRate checkpoints (see section 7)
  data/
    cameo/                194 test chains
    casp14/               37 test chains
    train/                7,980 training chains (a fixed 8,000-chain subset of the 40,349-chain set used by GeoBPE, minus unparsable files)
    train_list.txt        the training list used by the training commands
  cache/                  results of the other methods and the speed measurements (section 11)
    baselines/            ProToken and AminoAseed: result JSONs and decoded backbones (<name>_decoded/<chain stem>.npy)
    geobpe_decoded/       GeoBPE K=600 rebuilt backbones, one .npy per chain (219 chains)
    geobpe_quality/       GeoBPE K=600 per-chain results
    geobpe_paper_points.json   reference points of GeoBPE, ESM3 and VQ-VAE
    speed/                throughput and latency JSONs of ProRate and GeoBPE
  train_commands/         one file per arm with the arguments of train_protrig.py, one token per line
  reference/              the six figures as first rendered
```

Note: chain 7w6b_A is in both CAMEO and CASP14 (the files are identical), so there are 230 distinct stems. Cached per-chain files are named by
stem. Coverage of the cached decodings: GeoBPE 219 chains (its featurization drops 11), ProToken 229, AminoAseed 230.

Checkpoints are plain `torch.save` files with keys `model` (state dict), `cfg` (the ProtRigConfig as a dict), `args` (rate_weight,
min_piece, seed), `train_files` (file names) and `train_residues`. Load one with:

```python
from prorate.checkpoint import load_protrig
model = load_protrig("release/checkpoints/protrig_s8k_rw0p4_s0.pt", "cuda")
```

## 5. Running everything

```bash
bash run.sh
```

`run.sh` must be run from the repository root. It stops at the first failing stage and names the log (`FAILED <stage>: see logs/<stage>.log`).

| Stage | Script | Reads | Writes | Time on the reference machine |
|---|---|---|---|---|
| `eval_<arm>` (7) | `figures.py eval` | checkpoint, CAMEO, CASP14, 1,500 training chains | `outputs/eval/<arm>/eval.json`, `eval_per_chain.json` | 3 to 5 min each |
| `panel` | `figures.py panel` | the seven evals, `cache/` | `outputs/fig_panel_seed1.{pdf,png}` | seconds |
| `structures` | `fig_structures_pymol.py` | two checkpoints, test sets, `cache/` | `outputs/fig_structures_pymol_seed1.*` and manifest | about 2 min |
| `closeup` | `fig_boundary_closeup.py` | structures manifest, two checkpoints, `cache/` | close-up, scatter and zoom figures, manifest, scatter JSON | about 2 min |
| `token_efficiency` | `fig_token_efficiency.py` | the 8.6 BPR checkpoint, test sets | `outputs/fig_token_efficiency.*` | under 1 min |

The total is about 35 minutes. Stages are resumable at the evaluation level: an arm whose `outputs/eval/<arm>/eval.json` exists is skipped.
Delete that file to redo the arm. The figure stages always rerun.

Environment variables read by `run.sh`: `PY` (interpreter, default `python`), `PYMOL` (PyMOL executable), `DEVICE` (default `cuda`).

To replot only the panel after the evals exist:

```bash
E=$(ls outputs/eval/*/eval.json | tr '\n' ' ')
python scripts/figures.py panel --protrig $E --geobpe release/cache/geobpe_paper_points.json \
  --length-free outputs/eval/protrig_s8k_rw0p015_s0/eval_per_chain.json --length-valid outputs/eval/protrig_s8k_rw0p06_link3_s1/eval_per_chain.json \
  --length-geobpe release/cache/geobpe_quality/geobpe_K600.json --length-protoken release/cache/baselines/protoken.json \
  --length-aminoaseed release/cache/baselines/aminoaseed.json --baseline-dir release/cache/baselines \
  --speed-dir release/cache/speed --geobpe-dir release/cache/speed/geobpe --out outputs/fig_panel_seed1
```

To replot the PyMOL figures without redoing the decoding: `python scripts/fig_structures_pymol.py --figure-only --out outputs/fig_structures_pymol_seed1`
reassembles from the saved renders and `--render-only --pymol <path>` reruns PyMOL on the saved PDBs. The close-up script takes the same flags.

## 6. Evaluation protocol

`figures.py eval --ckpt <stage1.pt> --arm <name> --rate-weight <lambda> --rate-chains 1500 --train-dir release/data/train --test cameo=<dir> casp14=<dir>`

- **Rate (BPR).** `L(T)` is amortized over the residues of the arm's fitting set D (the chains recorded in the checkpoint). The stream rate
  of D is estimated on 1,500 chains drawn with `RandomState(0)` and scaled to the full residue count of D. The same 1,500 chains are used for
  every arm. The uniform per-token code of GeoBPE's `BPE.capacity` is used: `log2|V|` bits per fragment id and the sum of log2(levels) bits per
  boundary cell (the exact boundary bits are recorded in `eval.json` under `description_length` and `train`).
- **Distortion.** Every test chain is encoded, then decoded from the integer stream alone (codebooks, boundary cells, raw pieces and break
  transforms; no coordinates). Backbone RMSD is ESM's `ProteinChain.rmsd(only_compute_backbone_rmsd=True)`: the per-atom RMSD divided by
  sqrt(3), the form used by GeoBPE and the published baselines. `bb_rmsd_std` is the standard per-atom form. CA-lDDT is ESM's `lddt_ca`.
  Means are over chains.
- **Test chains are not filtered.** Chain breaks are crossed by coded break transforms, pieces under 3 residues are sent raw, and all of
  this is charged in the rate.
- **Train sample.** 100 training chains drawn with `RandomState(1)` are evaluated the same way; `test_over_train_bb_rmsd` in `eval.json` is the
  ratio of the test RMSD to this value.
- **Link geometry.** `peptide_validity` and `omega_planarity` report, for decoded and reference links, the fraction beyond 2 and 3 sigma of
  the Engh and Huber ideals (C-N 1.329 +- 0.014 A, CA-C-N 116.2 +- 2.0, C-N-CA 121.7 +- 1.8 degrees) and beyond 10 and 30 degrees of omega
  from trans, split into links between fragments and links inside fragments.

Main fields of `eval.json`: `arm`, `rate_weight`, `train.bpr_per_res_L_V` (the BPR used in the figures), `train.bpr_per_res_tokens` (stream
only), `description_length`, `train_sample`, and for each of `cameo` and `casp14`: `bb_rmsd`, `bb_rmsd_std`, `ca_rmsd`, `lddt_ca`, `evaluated`,
`listed`, `breaks`, `raw_residues`, `peptide_validity`, `omega_planarity`, `link_geometry_decoded_minus_reference`, `encode_residues_per_s`.
`eval_per_chain.json` holds the per-chain rows (RMSD, lDDT, residues, bits) behind the length plots.

## 7. The seven checkpoints

All were fitted on the same 7,980 chains (`--rounds 6 --prune-from 6`, codebook sizes up to 512 for lengths 1-2, 2048 for 3-6 and 1024 for
7-16 before pruning, `--boundary-search --forced-calibration`). The arm name encodes lambda (`rw0p06` is 0.06) and the seed.

| Arm | Role | lambda | Seed | Codes | BPR | RMSD CAMEO | RMSD CASP14 | CA-lDDT CAMEO | CA-lDDT CASP14 |
|---|---|---|---|---|---|---|---|---|---|
| `protrig_s8k_rw0p4_s0` | free | 0.4 | 0 | 1577 | 8.58 | 0.3325 | 0.3355 | 0.9277 | 0.9249 |
| `protrig_s8k_rw0p2_s1` | free | 0.2 | 1 | 881 | 11.25 | 0.2015 | 0.2100 | 0.9798 | 0.9773 |
| `protrig_s8k_rw0p1_s0` | free | 0.1 | 0 | 930 | 14.30 | 0.1513 | 0.1551 | 0.9931 | 0.9928 |
| `protrig_s8k_rw0p06_s1` | free | 0.06 | 1 | 907 | 20.22 | 0.1156 | 0.1170 | 0.9979 | 0.9976 |
| `protrig_s8k_rw0p03_s1` | free | 0.03 | 1 | 1144 | 20.87 | 0.1049 | 0.1066 | 0.9988 | 0.9987 |
| `protrig_s8k_rw0p015_s0` | free | 0.015 | 0 | 1133 | 39.54 | 0.0615 | 0.0647 | 0.9996 | 0.9996 |
| `protrig_s8k_rw0p06_link3_s1` | valid | 0.06 | 1 | 1681 | 36.22 | 0.3548 | 0.3627 | 0.9437 | 0.9325 |

These are the values written by `run.sh` on the reference machine (`outputs/eval/*/eval.json`). Rate columns match exactly; distortion can
differ in the last digits on other hardware because the boundary search runs in float64 on GPU-dependent kernels. The first-render
`eval.json` files of the original runs differ from these by at most 0.06 A on a single chain (0.002 to 0.006 A on the means): those were
produced by an earlier version of the decoding code, and the numbers in this table are the reference.

The "figure set" is called `seed1` after the file names of the original run. It combines the arms above (mixed seeds) and is not a single seed.

## 8. The figures

- **`fig_panel_seed1`** (main figure). A, B: RMSD and CA-lDDT against BPR (filled CAMEO, open CASP14). C, D: RMSD and CA-lDDT against chain
  length for the common chains of all methods (five length bins). E: throughput; F: latency per chain; G: CPU seconds per 1,000 residues.
  ProRate (free) points are the six free arms, the diamond is the valid arm. ProToken and AminoAseed points are our measurements; GeoBPE,
  ESM3 and VQ-VAE points come from `geobpe_paper_points.json`. E to G read the shipped speed JSONs (section 11).
- **`fig_structures_pymol_seed1`.** Ground truth, ProRate (free, `rw0p015_s0`), ProRate (valid, `link3_s1`), GeoBPE, ProToken and AminoAseed
  for six chains, superposed on the ground truth over all N, CA, C atoms and rendered from one camera per chain. The chains are not random:
  with `--select-worst-closeup`, the pool is every test chain of 80 to 250 residues with no unresolved residue that all methods decoded
  (117 chains); a chain is eligible if it has a link between fragments where both ProRate models are within 0.5 A of the ground truth over the
  2 x 3 residues around the link and the GeoBPE link error is at least 2 sigma (75 chains); the six eligible chains where GeoBPE is farthest
  from the ground truth around the link are shown, worst first. After the close-up stage has run, the script rewrites this figure with red
  squares marking where each close-up comes from.
- **`fig_boundary_closeup_seed1`.** For each of the six chains, the link where GeoBPE is farthest from the ground truth: C-N length,
  CA-C-N, C-N-CA and omega deviation in units of sigma, with the RMSD and LDDT over the residues shown. Red marks values beyond 2 sigma
  (omega: 10 degrees).
- **`fig_boundary_closeup_seed1_scatter`.** Cumulative distributions of the error against the ground truth for about 14,000 links (six
  drawn chains plus 100 more, `--scatter-chains 100`): RMSD and 1 - CA-lDDT over 16 residues around the link, C-N length, C-N-CA, CA-C-N and
  omega. The saved `*_scatter.json` holds the underlying numbers.
- **`fig_boundary_closeup_seed1_zoom`.** The structures figure joined to the close-ups, with the red box and connecting lines.
- **`fig_token_efficiency`.** Utilization rate and max-normalized perplexity of the codebook on CAMEO and CASP14. The ProRate row is computed by
  `fig_token_efficiency.py` from `protrig_s8k_rw0p4_s0` (1577 codes; the 8.6 BPR model): UR 59.35% / 38.55% and normalized perplexity 0.3632 /
  0.2834 (CAMEO / CASP14). The other rows are constants from Table 10 of the GeoBPE paper. Note that the GeoBPE bar is K=2500 at about 41 BPR
  and the ProRate bar is the 8.6 BPR model, so the two are at different rates. The models plotted elsewhere have lower values (the
  39.5 BPR free model has UR 24.2% / 12.4%).

## 9. Reproducibility

- On one machine, `run.sh` is deterministic: two runs give bit-identical panel, scatter and token-efficiency figures. The PyMOL figures
  agree to about 1e-7 in mean pixel difference (ray-tracing noise).
- Seeds: evaluation uses `RandomState(0)` for the rate sample and `RandomState(1)` for the training sample; the structures selection uses
  `--seed 0` (the selection itself is deterministic); training uses `--seed` from the training command.
- `release/reference/` holds the figures as first rendered. The regenerated panel and scatter differ from them by about 5e-4 in mean pixel
  difference and the token-efficiency figure by 5e-5. Rows 5 and 6 of the structures figure differ (the first render shows 7ejg_C and
  7f4s_D, `run.sh` shows 7f4s_D and 7pln_A), so the structures, close-up and zoom figures have a different height from the references. The
  four chains above them, the link counts of the scatter (13,991, 13,470 and 13,997) and the BPR values are the same.
- Speed panels (E to G) are read from shipped JSONs and are not remeasured; they depend on the hardware of section 3.

## 10. Training

```bash
bash train.sh protrig_s8k_rw0p06_link3_s1
```

reads `release/train_commands/<arm>.txt` (one argument per line) and runs `scripts/train_protrig.py`. The output is
`outputs/train/<arm>/` with `stage1.pt` (the final codec), `ckpt.pt` (saved every round), `config.json` and `metrics.jsonl` (per-round
objective, timings and GPU memory). Logs go to `logs/train_<arm>.log`.

The training commands share: `--file-list release/data/train_list.txt --max-chains 0 --fit-chains 0 --val-chains 0 --rounds 6
--prune-from 6 --tol 0 --batch-size 32 --boundary-batch 128 --max-items 16000 --boundary-search --forced-calibration
--codebook-sizes 1-2:512,3-6:2048,7-16:1024`, plus `--rate-weight` and `--seed`. The valid arm adds `--natural-boundary --natural-q 0.5
--link-weight 0.01 --link-tol 2.0 --fsq-levels 8,8,64,8,16,64`.

Each round segments every fitting piece exactly under the current codebooks, refits the codebooks by k-medoids on the chosen segments,
refits the boundary grid and rate table, and recalibrates the boundary cost E(L) = a * L^p; from round 6 it prunes every code whose error
saving does not pay its `32(9L-6)` bits of L(T) at the arm's lambda. There is no gradient step.

To resume an interrupted run, append the two lines `--resume` and `outputs/train/<arm>/ckpt.pt` to the arm's file in
`release/train_commands/` (the trainer accepts `--resume`; `train.sh` does not add it itself).

To evaluate a retrained arm, pass its checkpoint to `figures.py eval`, for example
`python scripts/figures.py eval --ckpt outputs/train/<arm>/stage1.pt --arm <arm> --rate-weight <lambda> --rate-chains 1500 --train-dir release/data/train --test cameo=release/data/cameo casp14=release/data/casp14 --out outputs/eval_retrained/<arm>/eval.json`.

A retrain is seeded but is not guaranteed to match the released checkpoint bit for bit: k-medoids and the float64 SVDs are not
deterministic across hardware. Evaluating the released checkpoints, not a retrain, is what reproduces the figures.

## 11. Other methods and where their numbers come from

| Method | What is used | Source |
|---|---|---|
| ProToken | RMSD and lDDT per chain, decoded backbones, timings; BPR 133 | measured by us on the CAMEO and CASP14 chains (229 decoded) |
| AminoAseed | same; BPR 398 | measured by us (230 decoded) |
| GeoBPE K=600 | per-chain RMSD and lDDT, decoded backbones, CPU timings | measured by us; fitted on 184 chains (41,959 residues); chains its featurization accepts (185 CAMEO, 35 CASP14) |
| GeoBPE K=2500, 6000, 21000 | points in `geobpe_paper_points.json` | logged and published results of GeoBPE |
| ESM3, VQ-VAE | points in `geobpe_paper_points.json` | published results |
| All baselines | token efficiency rows | Table 10 of the GeoBPE paper |

`geobpe_paper_points.json` also holds `_approx` entries for ProToken and AminoAseed; the figures use our measured points for those two
methods instead. ProRate speed (throughput and latency) comes from `cache/speed/*.json`, measured with an RTX 4090 and an i9-14900KF.
GeoBPE timing comes from `cache/speed/geobpe/geobpe_K600_released.json` (one CPU thread).

To verify any of these numbers, run the original repositories on the same chains in `release/data/`.

## 12. Why only CAMEO and CASP14

They are the held-out, post-cutoff test sets on which all compared tokenizers report reconstruction, so the points are comparable. They are
small (194 and 37 chains), so differences on CASP14 should be read with that in mind. A random sample of 300 chains from the benchmark's
validation split is not a clean held-out set: 196 of the 300 share at least 90% sequence identity with a chain of the 40,349-chain training
set, so it is not used.

## 13. Troubleshooting

- `release/ is missing`: run `bash download.sh`. It needs the Zenodo record to be published.
- `pymol not found`: install PyMOL and set `PYMOL=/path/to/pymol`.
- `foldingdiff reference left in the sources`: `run.sh` refuses to start if any source refers to the original GeoBPE codebase; this should not
  happen in a clean clone.
- CUDA out of memory in `eval`: lower `--batch-size` in `run.sh` (the default is 64).
- A figure stage failed: open `logs/<stage>.log`. A failed eval leaves no `eval.json`, so rerunning `run.sh` redoes only the failed arm.
- `Angles is None` warnings for some CASP14 chains come from the GeoBPE featurization and are harmless for ProRate.
- Different numbers in the last digits on another GPU are expected (section 7).

## 14. Citation and license

Citation: see `readme.md`; the data and checkpoints have their own DOI (https://doi.org/ZENODO_DOI).

Code: MIT (see `LICENSE`). Data: PDB entries, CAMEO and CASP14 targets keep the terms of their sources. The decoded structures of ProToken and
AminoAseed in `cache/` are outputs of those models, which have their own licenses. The archive is distributed under CC BY 4.0.
