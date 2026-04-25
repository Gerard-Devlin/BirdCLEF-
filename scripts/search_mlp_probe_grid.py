import argparse
import csv
import itertools
import os
import re
import subprocess
import sys
import time
from pathlib import Path


RAW_OOF_RE = re.compile(r"\[raw Perch\] honest OOF macro-AUC:\s*([0-9]*\.?[0-9]+)")
FULL_OOF_RE = re.compile(r"Full pipeline OOF AUC:\s*([0-9]*\.?[0-9]+)")


def parse_int_list(raw: str) -> list[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def parse_float_list(raw: str) -> list[float]:
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def parse_kv(items: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Invalid --set item (expect KEY=VALUE): {item}")
        k, v = item.split("=", 1)
        k = k.strip()
        if not k:
            raise ValueError(f"Invalid --set key: {item}")
        out[k] = v
    return out


def extract_metric(log_path: Path, metric_mode: str) -> float | None:
    text = log_path.read_text(encoding="utf-8", errors="ignore")
    m = (RAW_OOF_RE if metric_mode == "raw" else FULL_OOF_RE).search(text)
    if not m:
        return None
    return float(m.group(1))


def main() -> None:
    p = argparse.ArgumentParser(
        description="Grid-search MLP probe hyperparams for two_pass_ssm_pipeline_v2.py."
    )
    p.add_argument("--root", type=Path, required=True, help="Project root, e.g. /home/wuzhijian/birdclef+")
    p.add_argument("--adapter-ckpt", type=Path, required=True, help="Path to perch adapter ckpt (.pth)")
    p.add_argument("--run-tag-prefix", default="grid_mlp", help="Prefix for each run tag")
    p.add_argument("--python-bin", default=sys.executable, help="Python executable")
    p.add_argument("--pipeline-script", type=Path, default=Path("scripts/two_pass_ssm_pipeline_v2.py"))
    p.add_argument("--cuda-visible-devices", default="1")
    p.add_argument("--mode", default="train", choices=["train", "submit"])
    p.add_argument("--metric-mode", default="raw", choices=["raw", "full"])
    p.add_argument(
        "--stop-after-raw",
        action="store_true",
        help="Set BC26_STOP_AFTER_RAW_OOF=1 (recommended when --metric-mode raw).",
    )
    p.add_argument("--seed", default="42")
    p.add_argument("--adapter-weight", default="0.22")
    p.add_argument("--results-csv", type=Path, default=None)
    p.add_argument("--max-runs", type=int, default=0, help="0 means run all combinations")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument(
        "--set",
        action="append",
        default=[],
        help="Extra env override KEY=VALUE (repeatable), e.g. --set BC26_PROTO_EPOCHS=140",
    )

    p.add_argument("--grid-hidden1", default="128,256,384")
    p.add_argument("--grid-hidden2", default="64,128,192")
    p.add_argument("--grid-max-iter", default="400,700,1000")
    p.add_argument("--grid-niter-no-change", default="25,40")
    p.add_argument("--grid-alpha", default="0.0005,0.001,0.003")
    p.add_argument("--grid-lr-init", default="0.0002,0.0003,0.0005")
    p.add_argument("--grid-blend", default="0.35,0.5,0.65")
    p.add_argument("--grid-proto-cross-attn-heads", default="2")
    p.add_argument(
        "--grid-adapter-weight",
        default="",
        help="Comma-separated adapter weights for raw/full search, e.g. 0,0.05,0.1,0.15,0.2",
    )

    args = p.parse_args()

    root = args.root.resolve()
    if not root.exists():
        raise FileNotFoundError(f"root not found: {root}")
    if not args.adapter_ckpt.exists():
        raise FileNotFoundError(f"adapter ckpt not found: {args.adapter_ckpt}")

    pipeline_script = args.pipeline_script
    if not pipeline_script.is_absolute():
        pipeline_script = (root / pipeline_script).resolve()
    if not pipeline_script.exists():
        raise FileNotFoundError(f"pipeline script not found: {pipeline_script}")

    logs_dir = root / "logs"
    ckpt_dir = root / "work" / "ckpt"
    cache_dir = root / "work" / "cache"
    logs_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    results_csv = args.results_csv or (logs_dir / f"{args.run_tag_prefix}_results.csv")
    extra_env = parse_kv(args.set)

    h1_list = parse_int_list(args.grid_hidden1)
    h2_list = parse_int_list(args.grid_hidden2)
    max_iter_list = parse_int_list(args.grid_max_iter)
    niter_no_change_list = parse_int_list(args.grid_niter_no_change)
    alpha_list = parse_float_list(args.grid_alpha)
    lr_list = parse_float_list(args.grid_lr_init)
    blend_list = parse_float_list(args.grid_blend)
    proto_heads_list = parse_int_list(args.grid_proto_cross_attn_heads)
    adapter_weight_list = (
        parse_float_list(args.grid_adapter_weight)
        if args.grid_adapter_weight.strip()
        else [float(args.adapter_weight)]
    )

    combos = list(
        itertools.product(
            h1_list,
            h2_list,
            max_iter_list,
            niter_no_change_list,
            alpha_list,
            lr_list,
            blend_list,
            proto_heads_list,
            adapter_weight_list,
        )
    )
    if args.max_runs > 0:
        combos = combos[: args.max_runs]

    print(f"[INFO] total combos: {len(combos)}")
    print(f"[INFO] results csv: {results_csv}")
    if args.metric_mode == "raw":
        print(
            "[WARN] raw metric is computed before Proto/MLP/residual. "
            "If only MLP probe params change, raw score will likely stay identical."
        )

    base_env = os.environ.copy()
    base_env.update(
        {
            "PYTHONNOUSERSITE": "1",
            "ORT_DISABLE_CPU_AFFINITY": "1",
            "CUDA_VISIBLE_DEVICES": str(args.cuda_visible_devices),
            "BC26_USE_GPU": "1",
            "BC26_MODE": args.mode,
            "BC26_BASE": str(root / "dataset"),
            "BC26_MODEL_DIR": str(root / "models" / "bird-vocalization-classifier-tensorflow2-perch_v2_cpu-v1"),
            "BC26_WORK_DIR": str(cache_dir),
            "BC26_ONNX_PATH": str(root / "source" / "Perch-onnx-for-birdclef+2026" / "perch_v2_ir9.onnx"),
            "BC26_EXTRA_CACHE_DIRS": str(root / "source" / "Perch_meta"),
            "BC26_LOAD_CKPT_IN_TRAIN": "0",
            "BC26_DISABLE_GENUS_PROXY": "1",
            "BC26_SEED": str(args.seed),
            "BC26_PERCH_ADAPTER_CKPT": str(args.adapter_ckpt),
            "BC26_PERCH_ADAPTER_WEIGHT": str(args.adapter_weight),
        }
    )
    if args.stop_after_raw or args.metric_mode == "raw":
        base_env["BC26_STOP_AFTER_RAW_OOF"] = "1"
    base_env.update(extra_env)

    header = [
        "run_tag",
        "hidden1",
        "hidden2",
        "max_iter",
        "n_iter_no_change",
        "alpha",
        "lr_init",
        "blend",
        "proto_cross_attn_heads",
        "adapter_weight",
        "metric",
        "metric_value",
        "status",
        "elapsed_sec",
        "log_path",
        "ckpt_path",
    ]
    rows: list[dict[str, str | int | float]] = []

    for idx, (h1, h2, mx, nit, a, lr, blend, pheads, aw) in enumerate(combos, start=1):
        run_tag = (
            f"{args.run_tag_prefix}_{idx:03d}_h{h1}-{h2}_it{mx}_n{nit}_"
            f"a{a:g}_lr{lr:g}_b{blend:g}_ph{pheads}_aw{aw:g}"
        )
        log_path = logs_dir / f"train_ckpt_{run_tag}.log"
        ckpt_path = ckpt_dir / f"two_pass_pipeline_ckpt_{run_tag}.pth"
        sub_path = root / "work" / f"submission_local_{run_tag}.csv"

        env = base_env.copy()
        env.update(
            {
                "BC26_CKPT_PATH": str(ckpt_path),
                "BC26_SUBMISSION_PATH": str(sub_path),
                "BC26_MLP_PROBE_HIDDEN1": str(h1),
                "BC26_MLP_PROBE_HIDDEN2": str(h2),
                "BC26_MLP_PROBE_MAX_ITER": str(mx),
                "BC26_MLP_PROBE_N_ITER_NO_CHANGE": str(nit),
                "BC26_MLP_PROBE_ALPHA": f"{a}",
                "BC26_MLP_PROBE_LR_INIT": f"{lr}",
                "BC26_MLP_ALPHA_BLEND": f"{blend}",
                "BC26_PROTO_CROSS_ATTN_HEADS": str(pheads),
                "BC26_PERCH_ADAPTER_WEIGHT": f"{aw}",
            }
        )

        print(f"\n[RUN {idx}/{len(combos)}] {run_tag}")
        print(
            "  "
            f"h=({h1},{h2}) max_iter={mx} n_iter_no_change={nit} "
            f"alpha={a} lr={lr} blend={blend} proto_heads={pheads} adapter_weight={aw}"
        )

        if args.dry_run:
            rows.append(
                {
                    "run_tag": run_tag,
                    "hidden1": h1,
                    "hidden2": h2,
                    "max_iter": mx,
                    "n_iter_no_change": nit,
                    "alpha": a,
                    "lr_init": lr,
                    "blend": blend,
                    "proto_cross_attn_heads": pheads,
                    "adapter_weight": aw,
                    "metric": args.metric_mode,
                    "metric_value": "",
                    "status": "DRY_RUN",
                    "elapsed_sec": 0.0,
                    "log_path": str(log_path),
                    "ckpt_path": str(ckpt_path),
                }
            )
            continue

        start = time.time()
        with log_path.open("w", encoding="utf-8") as f:
            proc = subprocess.run(
                [args.python_bin, str(pipeline_script)],
                cwd=str(root),
                env=env,
                stdout=f,
                stderr=subprocess.STDOUT,
                check=False,
            )
        elapsed = time.time() - start
        metric_val = extract_metric(log_path, args.metric_mode)
        status = "OK" if proc.returncode == 0 else f"FAIL_{proc.returncode}"

        row = {
            "run_tag": run_tag,
            "hidden1": h1,
            "hidden2": h2,
            "max_iter": mx,
            "n_iter_no_change": nit,
            "alpha": a,
            "lr_init": lr,
            "blend": blend,
            "proto_cross_attn_heads": pheads,
            "adapter_weight": aw,
            "metric": args.metric_mode,
            "metric_value": "" if metric_val is None else metric_val,
            "status": status,
            "elapsed_sec": round(elapsed, 2),
            "log_path": str(log_path),
            "ckpt_path": str(ckpt_path),
        }
        rows.append(row)
        print(f"  status={status} {args.metric_mode}_auc={metric_val} elapsed={elapsed:.1f}s")

        with results_csv.open("w", encoding="utf-8", newline="") as wf:
            writer = csv.DictWriter(wf, fieldnames=header)
            writer.writeheader()
            for r in rows:
                writer.writerow(r)

    ok_rows = [r for r in rows if isinstance(r.get("metric_value"), (float, int))]
    if ok_rows:
        best = max(ok_rows, key=lambda x: float(x["metric_value"]))
        print("\n[BEST]")
        print(
            f"run_tag={best['run_tag']} {args.metric_mode}_auc={best['metric_value']} "
            f"h=({best['hidden1']},{best['hidden2']}) max_iter={best['max_iter']} "
            f"n_iter_no_change={best['n_iter_no_change']} alpha={best['alpha']} "
            f"lr={best['lr_init']} blend={best['blend']} "
            f"proto_heads={best['proto_cross_attn_heads']} "
            f"adapter_weight={best['adapter_weight']}"
        )
    else:
        print("\n[WARN] No successful OOF result found.")

    print(f"[DONE] Results written to: {results_csv}")


if __name__ == "__main__":
    main()
