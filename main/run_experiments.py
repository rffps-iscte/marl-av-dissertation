import os, sys, json, time, argparse, subprocess as sp
from pathlib import Path
from typing import Dict

THIS = Path(__file__).resolve()
THIS_DIR = THIS.parent
if (THIS_DIR / "__init__.py").exists():
    PROJECT_ROOT = THIS_DIR.parent
    PKG_NAME = THIS_DIR.name
    PKG_ROOT = THIS_DIR
else:
    PROJECT_ROOT = THIS_DIR
    PKG_NAME = "main"
    PKG_ROOT = PROJECT_ROOT / PKG_NAME

SUMO_DIR = PKG_ROOT / "sumo-simulation"
NET_FILE = SUMO_DIR / "my_grid5x5.net.xml"
ROUTES_DEV = SUMO_DIR / "my_routes_dev.rou.xml"
ROUTES_TRAIN = SUMO_DIR / "my_routes_train.rou.xml"
RUNS_DIR = PROJECT_ROOT / "runs"

ALGO_CONFIGS = {
    "MAPPO": {"module": f"{PKG_NAME}.MAPPO.main", "base_args": {}},
    "QMIX":  {"module": f"{PKG_NAME}.QMIX.main",  "base_args": {}},
}
TRAIN_SEEDS = [1, 2, 3]
ENV_EXTRA = {"PYTHONPATH": str(PROJECT_ROOT) + os.pathsep + os.environ.get("PYTHONPATH", "")}


def ensure_python(): return sys.executable or "python"

def to_argv(arg_dict):
    out = []
    for k, v in arg_dict.items():
        out.append(str(k))
        if v != "": out.append(str(v))
    return out

def write_sumocfg(out_path, net_file, route_file, step_length="1.0", seed="42"):
    xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<configuration>
  <input><net-file value="{net_file.as_posix()}"/><route-files value="{route_file.as_posix()}"/></input>
  <time><begin value="0"/><end value="10000"/><step-length value="{step_length}"/><seed value="{seed}"/></time>
  <processing><time-to-teleport value="60"/><default.action-step-length value="{step_length}"/><no-step-log value="true"/></processing>
  <report><verbose value="false"/></report>
</configuration>"""
    out_path.write_text(xml, encoding="utf-8")
    return out_path

def start_one(algo, seed, base_args, preset, step_length):
    py = ensure_python()
    run_dir = RUNS_DIR / algo / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    route_file = ROUTES_DEV if preset == "dev" else ROUTES_TRAIN
    cfg_path = run_dir / "sim.sumocfg"
    write_sumocfg(cfg_path, NET_FILE, route_file, step_length=step_length, seed=str(seed))

    args = dict(base_args)
    args["--seed"] = str(seed)
    args["--results_dir"] = str(run_dir)
    args["--sumo_config_path"] = str(cfg_path)

    (run_dir / "config.json").write_text(json.dumps({"algo": algo, "seed": seed, "args": args}, indent=2))

    env = os.environ.copy(); env.update(ENV_EXTRA)
    cmd = [py, "-m", ALGO_CONFIGS[algo]["module"]] + to_argv(args)
    log_out = (run_dir / "stdout.log").open("w", encoding="utf-8")
    log_err = (run_dir / "stderr.log").open("w", encoding="utf-8")

    print(f"\n>> {algo} | seed={seed} | preset={preset}")
    print(f"$ {' '.join(cmd)}")
    p = sp.Popen(cmd, cwd=str(PROJECT_ROOT), env=env, stdout=log_out, stderr=log_err)
    return p, log_out, log_err, run_dir

def wait_close(proc_tuple):
    p, lo, le, rd = proc_tuple
    rc = p.wait(); lo.close(); le.close()
    if rc != 0:
        print(f"FAIL {rd} code {rc}")
        try: print("\n".join(Path(le.name).read_text(errors="ignore").splitlines()[-40:]))
        except: pass
    else:
        print(f"OK {rd}")
    return rc

def aggregate(algo):
    import numpy as np
    base = RUNS_DIR / algo
    rewards, lengths, seeds = [], [], []
    for seed in TRAIN_SEEDS:
        p = base / f"seed_{seed}" / "rewards.npy"
        if p.exists():
            r = np.load(p); rewards.append(r); lengths.append(len(r)); seeds.append(seed)
    if not rewards: return {"algo": algo, "seeds": [], "mean": None, "ci95": None}
    T = min(lengths)
    R = np.stack([r[:T] for r in rewards], axis=0)
    mean = R.mean(axis=0); std = R.std(axis=0, ddof=1)
    ci95 = 1.96 * std / np.sqrt(R.shape[0])
    return {"algo": algo, "seeds": seeds, "R": R, "mean": mean, "ci95": ci95}

def aggregate_kpis(algo):
    import numpy as np
    base = RUNS_DIR / algo
    rows = []
    for seed in TRAIN_SEEDS:
        d = base / f"seed_{seed}"
        try:
            spd = np.load(d / "kpi_speed_norm.npy")
            hps = np.load(d / "kpi_halts_per_vehstep.npy")
            arv = np.load(d / "kpi_arrivals_per_100vehsteps.npy")
            try:
                mwt = np.load(d / "kpi_mean_waiting_time.npy")
                mtt = np.load(d / "kpi_mean_travel_time.npy")
                tel = np.load(d / "kpi_teleports.npy")
            except: mwt, mtt, tel = np.zeros_like(spd), np.zeros_like(spd), np.zeros_like(spd)
            rows.append((algo, seed, float(spd.mean()), float(hps.mean()), float(arv.mean()),
                         float(mwt.mean()), float(mtt.mean()), float(tel.sum())))
        except: pass
    return rows

def write_summary_csv(agg_m, agg_q):
    import numpy as np
    outdir = RUNS_DIR / "aggregate"; outdir.mkdir(parents=True, exist_ok=True)
    lines = ["algo,seed,episodes,final_return,last20_mean"]
    for agg in [agg_m, agg_q]:
        if agg.get("R") is None: continue
        R = agg["R"]; T = R.shape[1]; last = max(1, min(20, T))
        for i, seed in enumerate(agg["seeds"]):
            lines.append(f"{agg['algo']},{seed},{T},{float(R[i,-1]):.4f},{float(R[i,-last:].mean()):.4f}")
    (outdir / "summary.csv").write_text("\n".join(lines), encoding="utf-8")

    krows = aggregate_kpis("MAPPO") + aggregate_kpis("QMIX")
    if krows:
        klines = ["algo,seed,speed_norm,halts_per_vehstep,arrivals_per_100,mean_wait,mean_travel,teleports"]
        for row in krows:
            a, s, sp, ha, ar, wt, tt, te = row
            klines.append(f"{a},{s},{sp:.4f},{ha:.6f},{ar:.2f},{wt:.2f},{tt:.2f},{te:.0f}")
        (outdir / "kpi_summary.csv").write_text("\n".join(klines), encoding="utf-8")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parallel", type=int, default=2)
    ap.add_argument("--mode", choices=["fast", "full"], default="full")
    ap.add_argument("--traffic_preset", choices=["dev", "train"], default="train")
    ap.add_argument("--step_length", type=str, default="1.0")
    ap.add_argument("--episodes", type=int)
    ap.add_argument("--max_steps", type=int)
    ap.add_argument("--device", type=str)
    ap.add_argument("--algo", choices=["both", "MAPPO", "QMIX"], default="both")
    ap.add_argument("--no_plots", action="store_true")
    args = ap.parse_args()

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    overrides = {}
    if args.mode == "fast": overrides.update({"--episodes": "40", "--max_steps": "300"})
    if args.episodes: overrides["--episodes"] = str(args.episodes)
    if args.max_steps: overrides["--max_steps"] = str(args.max_steps)
    if args.device: overrides["--device"] = args.device

    algos = ["MAPPO", "QMIX"] if args.algo == "both" else [args.algo]
    procs = []
    for algo in algos:
        cfg = ALGO_CONFIGS[algo]
        base_args = dict(cfg["base_args"]); base_args.update(overrides)
        for seed in TRAIN_SEEDS:
            procs.append(start_one(algo, seed, base_args, args.traffic_preset, args.step_length))
            while len([p for p, *_ in procs if p.poll() is None]) >= args.parallel:
                time.sleep(5)

    for t in procs:
        if t[0].poll() is None: wait_close(t)

    agg_m = aggregate("MAPPO") if "MAPPO" in algos else {"R": None}
    agg_q = aggregate("QMIX") if "QMIX" in algos else {"R": None}

    if not args.no_plots:
        import numpy as np, matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        outdir = RUNS_DIR / "aggregate"; outdir.mkdir(parents=True, exist_ok=True)

        fig = plt.figure(figsize=(9, 5))
        def _plot(agg, label):
            if agg.get("mean") is None: return
            x = np.arange(len(agg["mean"])); m, c = agg["mean"], agg["ci95"]
            plt.plot(x, m, label=label); plt.fill_between(x, m-c, m+c, alpha=0.2)
        _plot(agg_m, "MAPPO"); _plot(agg_q, "QMIX")
        plt.xlabel("Episode"); plt.ylabel("Total Return"); plt.title("Learning Curves")
        plt.legend(); plt.tight_layout()
        fig.savefig(outdir / "learning_curves.png", dpi=160); plt.close(fig)

        for fname, ylabel, _ in [("kpi_speed_norm.npy", "Speed (norm)", True),
                                   ("kpi_halts_per_vehstep.npy", "Halts/Veh-Step", False),
                                   ("kpi_arrivals_per_100vehsteps.npy", "Arrivals/100", True),
                                   ("kpi_mean_waiting_time.npy", "Mean Wait (s)", False),
                                   ("kpi_mean_travel_time.npy", "Mean Travel (s)", False)]:
            fig, ax = plt.subplots(figsize=(9, 5))
            for algo_name in algos:
                arrays = []
                for seed in TRAIN_SEEDS:
                    fp = RUNS_DIR / algo_name / f"seed_{seed}" / fname
                    if fp.exists(): arrays.append(np.load(fp))
                if not arrays: continue
                T = min(len(a) for a in arrays)
                M = np.stack([a[:T] for a in arrays], axis=0)
                x = np.arange(T)
                ax.plot(x, M.mean(0), label=algo_name)
                ax.fill_between(x, M.mean(0) - 1.96*M.std(0, ddof=1)/np.sqrt(len(arrays)),
                                M.mean(0) + 1.96*M.std(0, ddof=1)/np.sqrt(len(arrays)), alpha=0.2)
            ax.set_xlabel("Episode"); ax.set_ylabel(ylabel); ax.legend(); fig.tight_layout()
            fig.savefig(outdir / fname.replace(".npy", ".png"), dpi=160); plt.close(fig)

    write_summary_csv(agg_m, agg_q)
    print("Done.")

if __name__ == "__main__":
    main()
