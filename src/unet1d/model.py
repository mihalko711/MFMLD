import torch
from torch import nn

import math
from typing import Optional, Tuple, Union, List


class ResidualBlock1D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, num_groups: int = 16,
                 dropout: float = 0.1, act_func: callable = nn.ReLU(), 
                 kernel_size: int = 3, padding: int = 1, dilation: int = 1,
                ):
        super().__init__()

        self.norm1 = nn.GroupNorm(num_groups, in_channels)
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation)
        self.act1 = act_func

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.norm2 = nn.GroupNorm(num_groups, out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding, dilation=dilation)
        self.act2 = act_func

        if in_channels != out_channels:
            self.shortcut = nn.Conv1d(in_channels, out_channels, 1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor):

        h = self.conv1(self.act1(self.norm1(x)))
        h = self.conv2(self.dropout(self.act2(self.norm2(h))))

        return h + self.shortcut(x)

class AttentionBlock1D(nn.Module):
    def __init__(self, n_channels: int = 1, n_heads: int = 1, d_k: int = None, num_groups: int = 16):
        super().__init__()

        if d_k is None:
            d_k = n_channels

        self.norm = nn.GroupNorm(num_groups, n_channels)
        self.projection = nn.Linear(n_channels, n_heads * d_k * 3)

        self.output = nn.Linear(n_heads * d_k, n_channels)

        self.scale = d_k ** -0.5

        self.n_heads = n_heads
        self.d_k = d_k

    def forward(self, x: torch.Tensor):

        batch_size, n_channels, width = x.shape

        x = self.norm(x)

        x = x.permute(0, 2, 1)

        qkv = self.projection(x).view(batch_size, -1, self.n_heads, self.d_k * 3)
        q, k, v  = torch.chunk(qkv, 3, dim=-1)

        attn = torch.einsum("bihd,bjhd->bijh", q, k) * self.scale
        attn = attn.softmax(dim=2)

        res = torch.einsum("bijh,bjhd->bihd", attn, v)

        res = res.view(batch_size, -1, self.n_heads * self.d_k)
        res = self.output(res)
        res = res.permute(0, 2, 1).view(batch_size, n_channels, width)

        return res
        
class DownBlock1D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, has_attn: bool):
        super().__init__()

        self.res = ResidualBlock1D(in_channels, out_channels)

        if has_attn:
            self.attn = AttentionBlock1D(out_channels)
        else:
            self.attn = nn.Identity()


    def forward(self, x: torch.Tensor):
        x = self.res(x)
        x = self.attn(x)
        return x

class UpBlock1D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, has_attn: bool):
        super().__init__()

        self.res = ResidualBlock1D(in_channels + out_channels, out_channels)

        if has_attn:
            self.attn = AttentionBlock1D(out_channels)
        else:
            self.attn = nn.Identity()


    def forward(self, x: torch.Tensor):
        x = self.res(x)
        x = self.attn(x)
        return x

class MiddleBlock1D(nn.Module):
    def __init__(self, n_channels: int):
        super().__init__()

        self.res1 = ResidualBlock1D(n_channels, n_channels)
        self.res2 = ResidualBlock1D(n_channels, n_channels)
        self.attn = AttentionBlock1D(n_channels)


    def forward(self, x: torch.Tensor):
        x = self.res1(x)
        x = self.attn(x)
        x = self.res2(x)
        return x

class Downsample1D(nn.Module):
    def __init__(self, n_channels: int):
        super().__init__()
        self.conv = nn.Conv1d(n_channels, n_channels, 3, 2, 1)

    def forward(self, x: torch.Tensor):
        return self.conv(x)

class Upsample1D(nn.Module):
    def __init__(self, n_channels):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='linear', align_corners=False)
        self.conv = nn.Conv1d(n_channels, n_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor):
        x = self.upsample(x)
        return self.conv(x)

class UNet1D(nn.Module):
    def __init__(self, entry_channels: int = 1, n_channels: int = 16,
                ch_mults: Union[Tuple[int, ...], List[int]] = (1, 2, 4),
                is_attn: Union[Tuple[bool, ...], List[bool]] = (False, False, False),
                n_blocks: int = 2, act: callable = nn.ReLU()):
        super().__init__()

        self.entry_channels = entry_channels
        self.n_channels = n_channels
        self.ch_mults = ch_mults
        self.is_attn = is_attn
        self.n_blocks = n_blocks
        self.act = act
        
        n_resolutions = len(ch_mults)

        self.signal_proj = nn.Conv1d(entry_channels, n_channels, 3, padding=1)

        down = []
        out_channels = in_channels = n_channels

        for i in range(n_resolutions):
            out_channels = in_channels * ch_mults[i]
            for _ in range(n_blocks):
                down.append(DownBlock1D(in_channels, out_channels, has_attn = is_attn[i]))
                in_channels = out_channels
                
            if i < n_resolutions - 1:
                down.append(Downsample1D(in_channels))

        self.down = nn.ModuleList(down)

        self.middle = MiddleBlock1D(out_channels)

        
        up = []
        in_channels = out_channels

        for i in reversed(range(n_resolutions)):
            out_channels = in_channels

            for _ in range(n_blocks):
                up.append(UpBlock1D(in_channels, out_channels, has_attn = is_attn[i]))

            out_channels = in_channels // ch_mults[i]
            up.append(UpBlock1D(in_channels, out_channels, has_attn = is_attn[i]))
            in_channels = out_channels

            if i > 0:
                up.append(Upsample1D(in_channels))

        self.up = nn.ModuleList(up)
        self.norm = nn.GroupNorm(8, n_channels)
        self.act = act
        self.final = nn.Conv1d(in_channels, entry_channels, 3, padding=1)

    def forward(self, x: torch.Tensor):
        x = self.signal_proj(x)
        h = [x]

        for m in self.down:
            x = m(x)
            h.append(x)

        x = self.middle(x)

        for m in self.up:
            if isinstance(m, Upsample1D):
                x = m(x)
            else:
                
                s = h.pop()
                x = torch.cat((x, s), dim=1)
                x = m(x)

        return self.final(self.act(self.norm(x)))
