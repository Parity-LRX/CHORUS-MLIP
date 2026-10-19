"""Remote-only frozen radial value+VJP tile microbenchmark; full model follows."""
import argparse
import gc
import json
import statistics
import time
from pathlib import Path
import torch
from chorus.hp.radial_trunk import _Trunk, _native


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(20261017)
    weights = [torch.randn(64, 8, device='cuda')*0.1,
               torch.randn(64, 64, device='cuda')*0.1,
               torch.randn(64, 64, device='cuda')*0.1]
    scales = (1.6791767923989418,)*3
    rows = []
    for tile in (0, 16, 32, 64, 128):
        small = torch.randn(4096, 8, device='cuda', requires_grad=True)
        probe = torch.randn(4096, 64, device='cuda')
        expected = _native(small, *weights, scales)
        got = _Trunk.apply(small, *weights, (*scales, tile)) if tile else expected
        torch.testing.assert_close(got, expected, atol=3e-6, rtol=1e-4)
        dx, = torch.autograd.grad((got*probe).sum(), small, retain_graph=True)
        ref, = torch.autograd.grad((expected*probe).sum(), small)
        torch.testing.assert_close(dx, ref, atol=4e-6, rtol=1e-4)
        del small, probe, expected, got, dx, ref
        x = torch.randn(500000, 8, device='cuda', requires_grad=True)
        grad = torch.randn(500000, 64, device='cuda')
        def run():
            value = (_Trunk.apply(x, *weights, (*scales, tile)) if tile else _native(x, *weights, scales))
            return value, torch.autograd.grad(value, x, grad)[0]
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):run()
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        cg = torch.cuda.CUDAGraph()
        with torch.cuda.graph(cg):result = run()
        samples = []
        for _ in range(15):
            torch.cuda.synchronize()
            start = time.perf_counter()
            cg.replay()
            torch.cuda.synchronize()
            samples.append((time.perf_counter()-start)*1000)
        row = {'tile': tile, 'median_ms': statistics.median(samples), 'samples_ms': samples}
        rows.append(row)
        print(json.dumps(row), flush=True)
        del cg, result, x, grad
        gc.collect()
        torch.cuda.empty_cache()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({'rows': rows}, indent=2)+'\n')


if __name__ == '__main__':main()
