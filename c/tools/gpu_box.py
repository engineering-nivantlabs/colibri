#!/usr/bin/env python3
"""Conservative Linux Vulkan launcher for a shared CPU/discrete-GPU box.

Uses Colibri's existing exclusive expert tier, not a new memory allocator.
Prints a reviewable launch plan unless --run is supplied.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

GIB = 1024 ** 3


def memory_available(path=Path('/proc/meminfo')):
    for line in path.read_text().splitlines():
        if line.startswith('MemAvailable:'):
            return int(line.split()[1]) * 1024
    raise ValueError('MemAvailable is unavailable')


def make_plan(model, available, reserve_gib, threads, cap, ctx, tokens, cpu=False):
    if available < (reserve_gib + 4) * GIB:
        raise ValueError('Less than 4 GiB remains after the host reserve; free memory first')
    # RAM_GB uses decimal GB in the engine. It is a planning target, not an OS limit.
    ram_gb = min(12.0, (available - reserve_gib * GIB) / 1e9)
    env = {
        'SNAP': str(model), 'OMP_NUM_THREADS': str(threads),
        'COLI_NO_OMP_TUNE': '1', 'RAM_GB': f'{ram_gb:.3f}',
        'Q36_MAXT': str(ctx), 'N_NEW': str(tokens),
        'COLI_CUDA': '0', 'COLI_VULKAN': '0' if cpu else '1',
        'COLI_VK_TIER': '0' if cpu else '1',
        'COLI_VK_TIER_EXCLUSIVE': '1',
        'COLI_VK_TIER_RESERVE_GB': '3',
        'COLI_VK_TIER_HOST_GB': '2',
    }
    return {'env': env, 'cap': cap, 'available_gib': round(available / GIB, 2),
            'host_reserve_gib': reserve_gib,
            'memory_model': 'VRAM hot experts, RAM warm experts, disk cold experts; PCIe transfers',
            'note': 'RAM_GB is an estimate, not an RSS limit. GPU budget is measured by Vulkan at startup.'}


def positive(value):
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError('must be positive')
    return result


def capacity(value):
    result = int(value)
    if result < 0:
        raise argparse.ArgumentTypeError('must be nonnegative')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True, help='Qwen GGUF supported by qwen36')
    parser.add_argument('--prompt', type=Path, required=True, help='text prompt file')
    parser.add_argument('--threads', type=positive, default=12)
    parser.add_argument('--cap', type=capacity, default=0, help='RAM expert slots per layer (0: engine sizes from RAM_GB; not GiB)')
    parser.add_argument('--ctx', type=positive, default=2048)
    parser.add_argument('--tokens', type=positive, default=32)
    parser.add_argument('--host-reserve-gib', type=positive, default=6)
    parser.add_argument('--cpu', action='store_true', help='CPU baseline with the same cache/context')
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    engine = root / 'c' / 'qwen36'
    try:
        model = args.model.expanduser().resolve(strict=True)
        prompt = args.prompt.expanduser().resolve(strict=True)
        if not model.is_file() or not prompt.is_file():
            raise ValueError('model and prompt must be files')
        plan = make_plan(model, memory_available(), args.host_reserve_gib,
                         args.threads, args.cap, args.ctx, args.tokens, args.cpu)
        plan['command'] = [str(engine), str(args.cap), '4', str(prompt)]
        print(json.dumps(plan, indent=2), flush=True)
        if not args.run:
            return 0
        if not engine.is_file():
            raise ValueError('Build first: make -C c qwen36 VK=1 -j4')
        # Do not inherit stale device indices, extra GPUs, or benchmark fault controls.
        env = {k: v for k, v in os.environ.items()
               if not k.startswith(('COLI_VK_', 'COLI_CUDA', 'Q36_', 'QWEN_', 'CUDA_', 'K3_'))
               and k not in ('SERVE', 'GGUF_CHECK', 'TOK_ONLY', 'TOK', 'PPL', 'TF', 'REPLAY', 'CONSIST')}
        env.update(plan['env'])
        return subprocess.call(plan['command'], cwd=root / 'c', env=env)
    except (OSError, ValueError) as error:
        print(f'gpu-box: {error}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    sys.exit(main())
