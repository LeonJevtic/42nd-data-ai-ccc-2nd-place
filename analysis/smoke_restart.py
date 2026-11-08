"""End-to-end smoke test of the next trainer config without touching runs/r1."""
import os, shlex, sys, glob
import os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repository root
sys.path.insert(0, ROOT)
os.chdir(ROOT)
from multiprocessing import Pool
import numpy as np

if __name__ == "__main__":
    import argparse, torch
    import az.train as T
    argv = shlex.split(open("runs/r1/args.txt").read())
    argv += ["--explore-frac", "0.15", "--temp-moves", "12", "--rescue-set", "matches/rescue_set.npz", "--rescue-set-frac", "0.02"]
    sys.argv = ["az.train"] + argv
    # reuse main()'s parser by monkeypatching parse: build it the same way
    p_holder = {}
    orig = argparse.ArgumentParser.parse_args
    def grab(self, *a, **k):
        ns = orig(self, *a, **k); p_holder["a"] = ns; raise SystemExit(0)
    argparse.ArgumentParser.parse_args = grab
    try:
        T.main()
    except SystemExit:
        pass
    argparse.ArgumentParser.parse_args = orig
    a = p_holder["a"]
    a.tourney_file = os.path.abspath(a.tourney_file); a.chunk = 8; a.concurrent = 8
    ck = sorted(glob.glob("runs/r1/net_*.pt"), key=os.path.getmtime)[-1]
    print("args ok:", a.explore_frac, a.temp_moves, a.rescue_set, a.rescue_set_frac, "ckpt", ck, flush=True)
    with Pool(3) as pool:
        data, st = T.gen_games_gpu(pool, os.path.abspath(ck), 24, 12345, a, opponents=[])
    print("selfplay stats:", st, flush=True)
    assert all(len(data[k]) == len(data["v"]) for k in ("obs", "idx", "prob", "kill", "mat", "rsc"))
    dev = "cuda"
    from az.net import Net
    net = Net(64, 6).to(dev); net.load_state_dict(torch.load(ck, map_location=dev, weights_only=False)["net"], strict=False)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-4)
    buf = {k: torch.as_tensor(data[k], device=dev) for k in ("obs", "v", "kill", "mat")}
    buf["idx"] = torch.as_tensor(data["idx"].astype(np.int16), device=dev)
    buf["prob"] = torch.as_tensor(data["prob"].astype(np.float16), device=dev)
    buf["aux_obs"] = torch.as_tensor(data["aux_obs"], device=dev); buf["aux_kill"] = torch.as_tensor(data["aux_kill"], device=dev)
    buf["rsc_idx"] = torch.as_tensor(np.flatnonzero(data["rsc"] == 1), device=dev)
    rs = np.load(a.rescue_set)
    buf["rs_obs"] = torch.as_tensor(rs["obs"], device=dev); buf["rs_idx"] = torch.as_tensor(rs["idx"].astype(np.int16), device=dev)
    buf["rs_prob"] = torch.as_tensor(rs["prob"].astype(np.float16), device=dev)
    import inspect
    print("train_steps signature:", inspect.signature(T.train_steps))
    out = T.train_steps(net, opt, buf, 5, 256, dev, rescue_boost=a.rescue_boost, rescue_set_frac=a.rescue_set_frac)
    print("train ok:", out, flush=True)
