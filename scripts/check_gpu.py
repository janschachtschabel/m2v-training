"""Print whether PyTorch sees CUDA GPUs, their names, memory and compute capability, and test a small matmul on each."""

from __future__ import annotations

import sys


def main() -> int:
    import torch

    print(f"torch {torch.__version__}, CUDA build {torch.version.cuda}")
    if not torch.cuda.is_available():
        print("No CUDA GPU visible. Training runs on the CPU (slow). Check the NVIDIA driver and the torch build.")
        return 1
    failed = 0
    count = torch.cuda.device_count()
    for index in range(count):
        props = torch.cuda.get_device_properties(index)
        major, minor = torch.cuda.get_device_capability(index)
        print(f"cuda:{index}: {props.name}, {props.total_memory / 2**30:.1f} GiB, compute capability {major}.{minor}")
        arch = f"sm_{major}{minor}"
        if arch not in torch.cuda.get_arch_list():
            print(f"  this torch build has no kernels for {arch}; install a build for a newer CUDA (RTX 50xx: cu128+).")
            failed += 1
            continue
        x = torch.randn(1024, 1024, device=f"cuda:{index}", dtype=torch.bfloat16)
        print("  bfloat16 matmul OK:", float((x @ x).float().abs().mean()) > 0)
    if count >= 2:
        print(f"{count} GPUs: start-compare runs one queue per GPU; teacher.devices uses several GPUs in one run.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
