"""Long-sequence CUDA/FP16 regression gate for the linear-attention retry."""

import argparse
import json
from pathlib import Path

import torch
from ppcbind.models.geometry import LinearAttentionContext


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-json", type=Path, required=True)
    args = p.parse_args()
    torch.manual_seed(49)
    torch.set_num_threads(4)
    assert torch.cuda.is_available()
    q = torch.ones(1, 8, 4096, 32, device="cuda")
    with torch.autocast("cuda", dtype=torch.float16):
        legacy_denom = torch.einsum("bhld,bhd->bhl", q.float(), q.sum(2).float())
    legacy_inf = int(torch.isinf(legacy_denom).sum())
    assert legacy_inf > 0, "Legacy CUDA overflow not reproduced; inspect before attributing the failed run"
    block = LinearAttentionContext(256, heads=8, dropout=0.).cuda().eval()
    with torch.no_grad():
        block.qkv.weight.zero_()
        block.qkv.bias.fill_(1.)
        block.output.weight.normal_(std=.02)
    h = torch.randn(1, 4096, 256, device="cuda", requires_grad=True)
    mask = torch.ones(1, 4096, device="cuda", dtype=torch.bool)
    with torch.no_grad():
        reference = block(h, mask)
    with torch.autocast("cuda", dtype=torch.float16):
        out = block(h, mask)
    assert torch.isfinite(out).all()
    torch.testing.assert_close(out, reference, rtol=.01, atol=.01)
    out.square().mean().backward()
    assert torch.isfinite(h.grad).all()
    assert all(torch.isfinite(param.grad).all() for param in block.parameters() if param.grad is not None)
    record = dict(torch=torch.__version__, GPU=torch.cuda.get_device_name(0), length=4096,
                  legacy_autocast_dtype=str(legacy_denom.dtype), legacy_infinite_denominators=legacy_inf,
                  fixed_output_finite=True, gradients_finite=True,
                  max_abs_difference_from_FP32=float((out - reference).abs().max()),
                  no_data_or_labels_read=True)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    if args.output_json.exists():
        raise FileExistsError(args.output_json)
    args.output_json.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
