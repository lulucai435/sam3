#!/usr/bin/env python3
"""
GPU keepalive: 预占指定显存 + 持续 matmul，避免被 cluster 的
low_mem_util / low_sm_util 策略 kill。

用法：
    python scripts/gpu_keepalive.py                  # 默认 70 GB / 50%
    python scripts/gpu_keepalive.py --mem-gb 60
    python scripts/gpu_keepalive.py --mem-frac 0.5   # 占总显存的 50%
    CUDA_VISIBLE_DEVICES=0 python scripts/gpu_keepalive.py

Ctrl-C 干净退出。
"""
from __future__ import annotations

import argparse
import signal
import sys
import time

import torch


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mem-gb", type=float, default=None,
                    help="预占显存（GB）。和 --mem-frac 二选一。")
    ap.add_argument("--mem-frac", type=float, default=0.5,
                    help="预占显存占总容量的比例（默认 0.5 → H200 上 ~71GB）。")
    ap.add_argument("--matmul-size", type=int, default=8192,
                    help="持续 matmul 的方阵边长（默认 8192）。")
    ap.add_argument("--report-every", type=float, default=30.0,
                    help="多少秒打一次状态（默认 30s）。")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available", file=sys.stderr)
        sys.exit(1)

    dev = torch.device("cuda:0")
    total_gb = torch.cuda.get_device_properties(dev).total_memory / 1024**3
    name = torch.cuda.get_device_name(dev)

    if args.mem_gb is not None:
        target_gb = args.mem_gb
    else:
        target_gb = total_gb * args.mem_frac

    print(f"Device: {name} ({total_gb:.1f} GB total)")
    print(f"Target reserved memory: {target_gb:.1f} GB ({100*target_gb/total_gb:.0f}%)")

    # 预占：fp32 张量，元素数 = bytes/4
    n_elems = int(target_gb * 1024**3 / 4)
    print(f"Allocating fp32 buffer of {n_elems:,} elements ...")
    try:
        _hold = torch.empty(n_elems, dtype=torch.float32, device=dev)
        _hold.fill_(0)  # 触发实际分配
    except torch.cuda.OutOfMemoryError as e:
        print(f"ERROR: OOM while reserving {target_gb:.1f} GB: {e}", file=sys.stderr)
        sys.exit(2)

    # 持续 matmul：占 SM，util 接近 100%
    s = args.matmul_size
    a = torch.randn(s, s, device=dev, dtype=torch.float32)
    b = torch.randn(s, s, device=dev, dtype=torch.float32)
    print(f"Starting {s}x{s} matmul loop. Ctrl-C to exit.")

    stop = {"v": False}
    def _onsig(signum, _frame):
        print(f"\nGot signal {signum}, exiting cleanly ...")
        stop["v"] = True
    signal.signal(signal.SIGINT, _onsig)
    signal.signal(signal.SIGTERM, _onsig)

    t_start = time.time()
    t_last = t_start
    iters = 0
    while not stop["v"]:
        c = a @ b
        # 用一下结果防止被优化掉
        a = c * 1e-8 + a
        torch.cuda.synchronize()
        iters += 1

        now = time.time()
        if now - t_last >= args.report_every:
            alloc = torch.cuda.memory_allocated(dev) / 1024**3
            resv = torch.cuda.memory_reserved(dev) / 1024**3
            elapsed = now - t_start
            ips = iters / elapsed
            print(f"[{elapsed/60:.1f} min] iters={iters} "
                  f"({ips:.1f}/s) alloc={alloc:.1f}GB reserved={resv:.1f}GB",
                  flush=True)
            t_last = now

    print("Exited.")


if __name__ == "__main__":
    main()
