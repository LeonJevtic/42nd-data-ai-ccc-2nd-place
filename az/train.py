"""AlphaZero-style loop: self-play (CPU workers) -> train (GPU) -> export ONNX.

python -m az.train --run runs/r1
"""
import argparse
import json
import os
import shutil
import time
from multiprocessing import Pool

import numpy as np

from az import arena
from az.selfplay import selfplay_game

# anti-diagonal reflection (r, c) -> (7-c, 7-r): a symmetry of every rule
PERM = np.array([(7 - (s % 8)) * 8 + (7 - s // 8) for s in range(64)], dtype=np.int64)


def log(run, msg):
    line = time.strftime("%H:%M:%S ") + msg
    print(line, flush=True)
    with open(os.path.join(run, "log.txt"), "a", encoding="utf-8") as f:
        f.write(line + "\n")


def gen_games(pool, spec, n, seed0, a, opponents=()):
    # a share of the games is played against fixed sparring opponents
    n_opp = int(n * a.opp_frac) if opponents else 0
    tasks = [(spec, seed0 + i, a.sims, a.temp_moves, a.noise_eps, a.q_mix) for i in range(n - n_opp)]
    tasks += [(spec, seed0 + n + i, a.sims, a.temp_moves, a.noise_eps, a.q_mix,
               opponents[i % len(opponents)]) for i in range(n_opp)]
    out = [g for g in pool.imap_unordered(selfplay_game, tasks, chunksize=2) if g is not None]
    data = {k: np.concatenate([g[k] for g in out]) for k in ("obs", "idx", "prob", "v")}
    selfp = [g for g in out if not g["opp"]]
    cyan = sum(1 for g in selfp if g["winner"] == "CYAN")
    stats = {"games": len(out), "cyan_win": round(cyan / max(1, len(selfp)), 3),
             "avg_len": round(float(np.mean([g["len"] for g in out])), 1),
             "samples": int(len(data["v"]))}
    for o in opponents:
        vs = [g for g in out if g["opp"] == o]
        stats[f"vs_{os.path.basename(o)}"] = f"{sum(g['won'] for g in vs)}/{len(vs)}"
    return data, stats


def gen_games_gpu(pool, ckpt, n, seed0, a, opponents=()):
    """Same output as gen_games, but games run concurrently with GPU-batched inference."""
    from az.gpu_selfplay import play_batch
    s = {"sims": a.sims, "noise_eps": a.noise_eps, "temp_moves": a.temp_moves, "q_mix": a.q_mix,
         "late_frac": a.late_frac, "open_sharpen": a.open_sharpen, "aux_rate": a.aux_rate,
         "tourney_file": a.tourney_file, "tourney_frac": a.tourney_frac,
         "rescue_rate": getattr(a, "rescue_rate", 0.0),
         "explore_frac": a.explore_frac, "explore_min": a.explore_min, "explore_max": a.explore_max}
    n_opp = int(n * a.opp_frac) if opponents else 0
    jobs = [(seed0 + i, None) for i in range(n - n_opp)]
    jobs += [(seed0 + n + i, opponents[i % len(opponents)]) for i in range(n_opp)]
    # interleave so every task gets a mix of self-play and opponent games
    jobs = jobs[::2] + jobs[1::2]
    tasks = [(ckpt, jobs[i:i + a.chunk], a.concurrent, s) for i in range(0, len(jobs), a.chunk)]
    out = [g for res in pool.imap_unordered(play_batch, tasks) for g in res]
    data = {k: np.concatenate([g[k] for g in out])
            for k in ("obs", "idx", "prob", "v", "kill", "mat", "aux_obs", "aux_kill")}
    data["rsc"] = np.concatenate([g.get("rsc", np.zeros(len(g["v"]), np.int8)) for g in out])
    selfp = [g for g in out if not g["opp"] and not g.get("late") and not g.get("explore")]
    late = [g for g in out if g.get("late")]
    cyan = sum(1 for g in selfp if g["winner"] == "CYAN")
    stats = {"games": len(out), "cyan_win": round(cyan / max(1, len(selfp)), 3),
             "avg_len": round(float(np.mean([g["len"] for g in selfp])), 1) if selfp else 0,
             "late_games": len(late),
             "late_cyan_win": round(sum(g["winner"] == "CYAN" for g in late) / max(1, len(late)), 3),
             "samples": int(len(data["v"])),
             "kill_labels": f"{int(data['aux_kill'].sum())}/{len(data['aux_kill'])} leaves, "
                            f"{int(data['kill'].sum())}/{len(data['kill'])} roots"}
    if a.explore_frac > 0.0:
        stats["explore_games"] = sum(1 for g in out if g.get("explore"))
    for o in opponents:
        vs = [g for g in out if g["opp"] == o]
        stats[f"vs_{os.path.basename(o)}"] = f"{sum(g['won'] for g in vs)}/{len(vs)}"
    if getattr(a, "rescue_rate", 0.0) > 0.0:
        rsc_b = sum(g.get("rsc_blunders", 0) for g in out)
        rsc_r = sum(g.get("rsc_rescued", 0) for g in out)
        rsc_h = sum(g.get("rsc_hopeless", 0) for g in out)
        rsc_t = sum(g.get("rsc_time", 0.0) for g in out)
        rsc_e = sum(g.get("rsc_errors", 0) for g in out)
        stats["rescue"] = f"blunders={rsc_b} rescued={rsc_r} hopeless={rsc_h} errors={rsc_e} t={rsc_t:.1f}s"
    return data, stats


def save_net(net, path, a):
    import torch
    torch.save({"net": {k: v.detach().cpu() for k, v in net.state_dict().items()},
                "cfg": {"channels": a.channels, "blocks": a.blocks}}, path)


def train_steps(net, opt, buf, steps, batch, device, rescue_boost=0.0, rescue_set_frac=0.0):
    import torch
    import torch.nn.functional as F
    perm = torch.as_tensor(PERM, device=device)
    n = buf["obs"].shape[0]
    n_aux = buf["aux_obs"].shape[0] if "aux_obs" in buf else 0
    rsc_idx = buf.get("rsc_idx") if rescue_boost > 0.0 else None
    # rescue-set auxiliary policy loss (only active when rescue_set_frac > 0)
    rs_obs = buf.get("rs_obs") if rescue_set_frac > 0.0 else None
    rs_idx = buf.get("rs_idx") if rescue_set_frac > 0.0 else None
    rs_prob = buf.get("rs_prob") if rescue_set_frac > 0.0 else None
    use_rs = rs_obs is not None and len(rs_obs) > 0
    rs_n = len(rs_obs) if use_rs else 0
    reflect = lambda o: o.transpose(2, 3).flip(2).flip(3)
    net.train()
    tot_p = tot_v = tot_k = tot_m = tot_rs = 0.0
    for _ in range(steps):
        sel = torch.randint(0, n, (batch,), device=device)
        if rsc_idx is not None and len(rsc_idx) > 0:
            k = int(batch * rescue_boost)
            if k > 0:
                picks = rsc_idx[torch.randint(0, len(rsc_idx), (k,), device=device)]
                sel[:k] = picks
        obs = buf["obs"][sel].float() / 8.0
        idx = buf["idx"][sel].long()
        prob = buf["prob"][sel].float()
        # soft cap: lost/won positions keep a gradient, so the net can rank them by how long they survive
        vt = buf["v"][sel] * 0.95
        kt = buf["kill"][sel].float()
        mt = buf["mat"][sel]
        # augment half of the batch with the anti-diagonal reflection
        flip = torch.rand(batch, device=device) < 0.5
        obs = torch.where(flip[:, None, None, None], reflect(obs), obs)
        valid = idx >= 0
        safe = idx.clamp(min=0)
        idx_f = perm[safe // 64] * 64 + perm[safe % 64]
        safe = torch.where(flip[:, None], idx_f, safe)

        rs_b = int(batch * rescue_set_frac) if use_rs else 0
        if rs_b > 0:
            rs_sel = torch.randint(0, rs_n, (rs_b,), device=device)
            rs_o = rs_obs[rs_sel].float() / 8.0
            rs_i = rs_idx[rs_sel].long()
            rs_p = rs_prob[rs_sel].float()
            rs_flip = torch.rand(rs_b, device=device) < 0.5
            rs_o = torch.where(rs_flip[:, None, None, None], reflect(rs_o), rs_o)
            rs_valid = rs_i >= 0
            rs_safe = rs_i.clamp(min=0)
            rs_if = perm[rs_safe // 64] * 64 + perm[rs_safe % 64]
            rs_safe = torch.where(rs_flip[:, None], rs_if, rs_safe)
            # one forward pass with the main batch: a separate small batch of these few positions
            # would drag the BatchNorm running statistics (used by the exported net) towards them
            out = net(torch.cat([obs, rs_o]), return_kill=True)
            logits, v, kl, mp = (t[:batch] for t in out)
            rs_logits = out[0][batch:]
        else:
            logits, v, kl, mp = net(obs, return_kill=True)
        g = logits.gather(1, safe).masked_fill(~valid, -1e9)
        lp = -(prob * F.log_softmax(g, dim=1)).sum(1).mean()
        lv = F.mse_loss(v, vt)
        # exact kill-this-turn labels (-1 = unknown, from older data files)
        known = kt >= 0
        lk = (F.binary_cross_entropy_with_logits(kl[known], kt[known]) if known.any()
              else torch.zeros((), device=device))
        # future material balance (-9 = unknown, from older data files)
        mk = mt > -5
        lm = F.mse_loss(mp[mk], mt[mk]) if mk.any() else torch.zeros((), device=device)
        if n_aux:
            sa = torch.randint(0, n_aux, (batch // 2,), device=device)
            ao = buf["aux_obs"][sa].float() / 8.0
            af = torch.rand(ao.shape[0], device=device) < 0.5
            ao = torch.where(af[:, None, None, None], reflect(ao), ao)
            _, _, akl, _ = net(ao, return_kill=True)
            lk = lk + F.binary_cross_entropy_with_logits(akl, buf["aux_kill"][sa].float())
        loss = lp + lv + 0.5 * lk + 0.5 * lm
        # rescue-set policy term (extra cross-entropy on curated blunder positions), weighted per
        # sample like the main batch so a few positions cannot outweigh the whole policy loss
        if rs_b > 0:
            rs_g = rs_logits.gather(1, rs_safe).masked_fill(~rs_valid, -1e9)
            rs_lp = -(rs_p * F.log_softmax(rs_g, dim=1)).sum(1).mean()
            loss = loss + rs_lp * (rs_b / batch)
            tot_rs += rs_lp.item()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        tot_p += lp.item()
        tot_v += lv.item()
        tot_k += lk.item()
        tot_m += lm.item()
    net.eval()
    return (tot_p / steps, tot_v / steps, tot_k / steps, tot_m / steps,
            tot_rs / steps if use_rs else None)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", default="runs/r1")
    p.add_argument("--workers", type=int, default=22)
    p.add_argument("--channels", type=int, default=64)
    p.add_argument("--blocks", type=int, default=6)
    p.add_argument("--gen0-games", type=int, default=4000)
    p.add_argument("--games", type=int, default=1500)
    p.add_argument("--sims", type=int, default=128)
    p.add_argument("--temp-moves", type=int, default=12)
    p.add_argument("--noise-eps", type=float, default=0.25)
    p.add_argument("--q-mix", type=float, default=0.25)
    p.add_argument("--window", type=int, default=6, help="iterations of data kept")
    p.add_argument("--reuse", type=float, default=6.0, help="avg times each new sample is trained on")
    p.add_argument("--batch", type=int, default=1024)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--iters", type=int, default=1000)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--opponents", nargs="*", default=[], help="fixed sparring models (.onnx)")
    p.add_argument("--opp-frac", type=float, default=0.2, help="share of games vs opponents")
    p.add_argument("--gpu-selfplay", action="store_true", help="concurrent games + batched GPU inference")
    p.add_argument("--concurrent", type=int, default=32, help="games per worker in flight (gpu mode)")
    p.add_argument("--chunk", type=int, default=96, help="games per worker task (gpu mode)")
    p.add_argument("--league", default="", help="comma list of generation offsets to spar against, e.g. 2,4,8")
    p.add_argument("--late-frac", type=float, default=0.0, help="share of self-play games from random late positions")
    p.add_argument("--open-sharpen", type=float, default=1.0, help="exponent on opening (actions 1-8) visit targets")
    p.add_argument("--aux-rate", type=float, default=0.0, help="share of search leaves labelled with can_kill")
    p.add_argument("--aux-window", type=int, default=3, help="iterations of kill-label data kept")
    p.add_argument("--tourney-file", default="", help="npz of real tournament positions to start games from")
    p.add_argument("--tourney-frac", type=float, default=0.0, help="share of self-play games from those positions")
    p.add_argument("--synth-file", default="", help="npz of synthetic kill-labelled positions (az/synth.py)")
    p.add_argument("--rescue-rate", type=float, default=0.0, help="probability of rewriting policy on blundered turns")
    p.add_argument("--rescue-boost", type=float, default=0.0, help="fraction of each batch filled with rescued samples")
    p.add_argument("--rescue-set", default="", help="path to rescue_set.npz built by az.rescue_set")
    p.add_argument("--rescue-set-frac", type=float, default=0.0,
                   help="fraction of each batch drawn from rescue-set for an auxiliary policy loss")
    p.add_argument("--explore-frac", type=float, default=0.0,
                   help="share of self-play games started after random legal actions from a real/initial position")
    p.add_argument("--explore-min", type=int, default=1)
    p.add_argument("--explore-max", type=int, default=6)
    a = p.parse_args()
    if a.synth_file:
        a.synth_file = os.path.abspath(a.synth_file)
    if a.tourney_file:
        a.tourney_file = os.path.abspath(a.tourney_file)
    if a.rescue_set:
        a.rescue_set = os.path.abspath(a.rescue_set)
    a.opponents = [os.path.abspath(o) for o in a.opponents]

    import torch
    from az.net import Net, export_onnx

    os.makedirs(a.run, exist_ok=True)
    device = "cuda"
    net = Net(a.channels, a.blocks).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    it0 = 0
    if a.resume and os.path.exists(os.path.join(a.run, "ckpt.pt")):
        ck = torch.load(os.path.join(a.run, "ckpt.pt"), map_location=device)
        missing, _ = net.load_state_dict(ck["net"], strict=False)
        if missing:
            print("new parameters (fresh optimizer state):", missing)
        else:
            opt.load_state_dict(ck["opt"])
        for group in opt.param_groups:   # --lr wins over the checkpointed rate
            group["lr"] = a.lr
        it0 = ck["it"] + 1
    log(a.run, f"start {vars(a)} from it {it0}")

    pool = Pool(a.workers)
    files = sorted(f for f in os.listdir(a.run) if f.startswith("data_"))
    spec = "heuristic"
    if it0 > 0:
        spec = os.path.abspath(os.path.join(a.run, f"gen{it0}.onnx"))
    net_path = lambda i: os.path.abspath(os.path.join(a.run, f"net_{i}.pt"))
    if a.gpu_selfplay:
        save_net(net, net_path(it0), a)
    offsets = [int(o) for o in a.league.split(",") if o]

    for it in range(it0, a.iters):
        t0 = time.time()
        n_games = a.gen0_games if it == 0 else a.games
        opponents = list(a.opponents) if it > 0 else []
        if a.gpu_selfplay and it > 0:
            opponents += [net_path(it - o) for o in offsets if os.path.exists(net_path(it - o))]
            data, st = gen_games_gpu(pool, net_path(it), n_games, seed0=it * 1_000_003, a=a,
                                     opponents=opponents)
        else:
            data, st = gen_games(pool, spec, n_games, seed0=it * 1_000_003, a=a, opponents=opponents)
        fn = f"data_{it:04d}.npz"
        np.savez(os.path.join(a.run, fn), **data)
        files = sorted(f for f in os.listdir(a.run) if f.startswith("data_"))[-a.window:]
        # fresh workers every iteration: their GPU memory (CUDA graphs, caching allocator) grows
        # by ~2.8 GB per iteration and stalls the training step after 2-3 iterations otherwise
        pool.close()
        pool.join()
        pool = Pool(a.workers)
        t1 = time.time()
        log(a.run, f"it {it} selfplay[{os.path.basename(spec)}] {st} in {t1 - t0:.0f}s")

        parts = [np.load(os.path.join(a.run, f)) for f in files]
        rsc_np = np.concatenate(
            [q["rsc"] if "rsc" in q else np.zeros(len(q["v"]), np.int8) for q in parts])
        buf = {
            "obs": torch.as_tensor(np.concatenate([q["obs"] for q in parts]), device=device),
            "idx": torch.as_tensor(np.concatenate([q["idx"] for q in parts]).astype(np.int16), device=device),
            "prob": torch.as_tensor(np.concatenate([q["prob"] for q in parts]).astype(np.float16), device=device),
            "v": torch.as_tensor(np.concatenate([q["v"] for q in parts]), device=device),
            "kill": torch.as_tensor(np.concatenate(
                [q["kill"] if "kill" in q else np.full(len(q["v"]), -1, np.int8) for q in parts]), device=device),
            "mat": torch.as_tensor(np.concatenate(
                [q["mat"] if "mat" in q else np.full(len(q["v"]), -9, np.float32) for q in parts]), device=device),
            "rsc_idx": torch.as_tensor(np.where(rsc_np == 1)[0], device=device),
        }
        aux_parts = [q for q in parts if "aux_obs" in q][-a.aux_window:]
        if a.synth_file:
            aux_parts.append(np.load(a.synth_file))   # fixed exactly-labelled firewall/kill positions
        if aux_parts:
            buf["aux_obs"] = torch.as_tensor(np.concatenate([q["aux_obs"] for q in aux_parts]), device=device)
            buf["aux_kill"] = torch.as_tensor(np.concatenate([q["aux_kill"] for q in aux_parts]), device=device)
        steps = max(200, int(st["samples"] * a.reuse / a.batch))
        if it == 0:
            steps *= 2
        if a.rescue_set and a.rescue_set_frac > 0.0:
            rs = np.load(a.rescue_set)
            buf["rs_obs"] = torch.as_tensor(rs["obs"], device=device)
            buf["rs_idx"] = torch.as_tensor(rs["idx"].astype(np.int16), device=device)
            buf["rs_prob"] = torch.as_tensor(rs["prob"].astype(np.float16), device=device)
        rsc_in_win = int(len(buf["rsc_idx"])) if "rsc_idx" in buf else 0
        lp, lv, lk, lm, lrs = train_steps(net, opt, buf, steps, a.batch, device,
                                           rescue_boost=getattr(a, "rescue_boost", 0.0),
                                           rescue_set_frac=getattr(a, "rescue_set_frac", 0.0))
        del buf
        torch.cuda.empty_cache()   # hand the replay buffer's memory back to the self-play workers
        out = os.path.abspath(os.path.join(a.run, f"gen{it + 1}.onnx"))
        export_onnx(net, out)
        if a.gpu_selfplay:
            save_net(net, net_path(it + 1), a)
        torch.save({"net": net.state_dict(), "opt": opt.state_dict(), "it": it,
                    "cfg": {"channels": a.channels, "blocks": a.blocks}},
                   os.path.join(a.run, "ckpt.pt"))
        t2 = time.time()
        rsc_suffix = f" rsc_in_win={rsc_in_win}" if getattr(a, "rescue_boost", 0.0) > 0.0 else ""
        rs_suffix = f" rs_lp={lrs:.3f}" if lrs is not None else ""
        log(a.run, f"it {it} train {steps} steps on {len(files)} files: policy {lp:.3f} value {lv:.3f} "
                   f"kill {lk:.3f} mat {lm:.3f}{rsc_suffix}{rs_suffix} in {t2 - t1:.0f}s")

        # quick referee-settings check against the previous generator
        r = arena.match(out, spec, games=44, sims=100, pool=pool, seed=it * 7919)
        log(a.run, f"it {it} arena gen{it + 1} vs {os.path.basename(spec)}: {json.dumps(r)} in {time.time() - t2:.0f}s")
        if it % 3 == 2:
            r = arena.match(out, "heuristic", games=44, sims=100, pool=pool, seed=it * 31)
            log(a.run, f"it {it} arena gen{it + 1} vs heuristic: {json.dumps(r)}")
        shutil.copy(out, os.path.join(a.run, "latest.onnx"))
        spec = out


if __name__ == "__main__":
    main()
