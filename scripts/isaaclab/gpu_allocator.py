#!/usr/bin/env python3
"""Keep CUDA contexts and a small allocation alive on selected GPUs."""

from __future__ import annotations

import os
import signal
import sys
import time
from typing import List


def _parse_gpus(raw: str) -> List[int]:
    raw = raw.replace(",", " ").strip()
    if not raw:
        return []
    if raw.lower() == "all":
        import torch

        return list(range(torch.cuda.device_count()))
    gpus = []
    for token in raw.split():
        try:
            gpu = int(token)
        except ValueError as exc:
            raise ValueError(f"invalid GPU id {token!r} in GPUS={raw!r}") from exc
        if gpu not in gpus:
            gpus.append(gpu)
    return gpus


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name, str(default))
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {value!r}") from exc


def main() -> int:
    try:
        import torch
    except Exception as exc:
        print(f"failed to import torch: {exc}", file=sys.stderr, flush=True)
        return 1

    if not torch.cuda.is_available():
        print("CUDA is not available", file=sys.stderr, flush=True)
        return 1

    try:
        gpus = _parse_gpus(os.environ.get("GPUS", ""))
        if not gpus:
            gpus = list(range(torch.cuda.device_count()))
        memory_mb = _env_int("GPU_ALLOCATOR_MEMORY_MB", 1024)
        safety_margin_mb = _env_int("GPU_ALLOCATOR_SAFETY_MARGIN_MB", 256)
        heartbeat_seconds = max(1, _env_int("GPU_ALLOCATOR_HEARTBEAT_SECONDS", 300))
        strict = os.environ.get("GPU_ALLOCATOR_STRICT", "1") == "1"
    except Exception as exc:
        print(str(exc), file=sys.stderr, flush=True)
        return 1

    if memory_mb <= 0:
        print(f"GPU_ALLOCATOR_MEMORY_MB must be positive, got {memory_mb}", file=sys.stderr, flush=True)
        return 1

    tensors = []
    failures = []
    requested_bytes = memory_mb * 1024 * 1024
    safety_margin_bytes = max(0, safety_margin_mb) * 1024 * 1024

    for gpu in gpus:
        try:
            if gpu < 0 or gpu >= torch.cuda.device_count():
                raise RuntimeError(f"GPU id {gpu} out of range; visible_count={torch.cuda.device_count()}")
            device = torch.device(f"cuda:{gpu}")
            torch.cuda.set_device(device)
            free_bytes, total_bytes = torch.cuda.mem_get_info(device)
            alloc_bytes = min(requested_bytes, max(0, free_bytes - safety_margin_bytes))
            if alloc_bytes <= 0:
                raise RuntimeError(
                    f"not enough free memory on cuda:{gpu}: "
                    f"free={free_bytes // (1024 * 1024)} MiB, safety_margin={safety_margin_mb} MiB"
                )
            tensor = torch.empty((alloc_bytes,), dtype=torch.uint8, device=device)
            tensor.fill_(1)
            tensors.append((gpu, tensor))
            torch.cuda.synchronize(device)
            print(
                f"holding cuda:{gpu}: allocated={alloc_bytes // (1024 * 1024)} MiB "
                f"free_before={free_bytes // (1024 * 1024)} MiB total={total_bytes // (1024 * 1024)} MiB",
                flush=True,
            )
        except Exception as exc:
            failures.append((gpu, str(exc)))
            print(f"failed to allocate on cuda:{gpu}: {exc}", file=sys.stderr, flush=True)

    if failures and strict:
        print("strict mode: exiting because at least one requested GPU failed", file=sys.stderr, flush=True)
        return 1
    if not tensors:
        print("no GPU allocations were created", file=sys.stderr, flush=True)
        return 1

    stop = False

    def _handle_signal(signum, _frame):
        nonlocal stop
        print(f"received signal {signum}; releasing held GPU allocations", flush=True)
        stop = True

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    held = ", ".join(f"cuda:{gpu}" for gpu, _ in tensors)
    print(f"GPU allocator alive on {held}; pid={os.getpid()}", flush=True)
    while not stop:
        time.sleep(heartbeat_seconds)
        print(f"GPU allocator heartbeat: holding {held}; pid={os.getpid()}", flush=True)

    tensors.clear()
    torch.cuda.empty_cache()
    print("GPU allocator stopped", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
