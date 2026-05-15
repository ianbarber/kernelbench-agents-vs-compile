import torch
import candidate

seed = 0xC0FFEE
g = torch.Generator(device='cuda')
g.manual_seed(seed)
x = torch.randn((1, 512, 2048), device='cuda', dtype=torch.bfloat16, generator=g)
res = torch.randn((512, 2048), device='cuda', dtype=torch.bfloat16, generator=g)
w = (torch.randn((2048,), device='cuda', dtype=torch.float32, generator=g) * 0.1 + 1.0).to(torch.bfloat16)

cand = candidate.run(x, res, w, 1e-6)

s = x.to(torch.float32) + res.to(torch.float32)
var = s.pow(2).mean(dim=-1, keepdim=True)
inv = torch.rsqrt(var + 1e-6)
ref = (s * inv * w.to(torch.float32)).to(torch.bfloat16)

row = 100
diff = (cand[0, row].to(torch.float32) - ref[0, row].to(torch.float32)).abs()
print('max diff in row 100:', diff.max().item(), 'at col', diff.argmax().item())
for col in [0, 512, 1023, 1024, 1025, 1491, 2047]:
    print(f'col {col}: cand={cand[0, row, col].item()}, ref={ref[0, row, col].item()}, diff={diff[col].item()}')

# Check if the issue is always at the tile boundary
for row in range(512):
    row_diff = (cand[0, row].to(torch.float32) - ref[0, row].to(torch.float32)).abs()
    if row_diff.max() > 0.1:
        max_col = row_diff.argmax().item()
        print(f'row {row} has large diff at col {max_col}: {row_diff[max_col].item()}')
        if max_col >= 1024:
            print('  -> second tile')
        else:
            print('  -> first tile')
        break
