"""EchoCardMAE-style single-frame masked autoencoder components."""

from __future__ import annotations

import torch
from torch import nn

from .patch import PatchEmbed2D, get_2d_sincos_pos_embed, patchify, roi_to_patch_mask
from .vit_blocks import Block


def _init_vit_weights(module: nn.Module) -> None:
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.LayerNorm):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)


class FrameMAEEncoder2D(nn.Module):
    def __init__(
        self,
        img_size: int = 112,
        patch_size: int = 8,
        in_chans: int = 1,
        embed_dim: int = 384,
        depth: int = 12,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        drop_path_rate: float = 0.0,
    ):
        super().__init__()
        self.patch_embed = PatchEmbed2D(img_size, patch_size, in_chans, embed_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.background_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        pos = get_2d_sincos_pos_embed(embed_dim, self.patch_embed.grid_size)
        self.register_buffer("pos_embed", pos, persistent=False)
        dpr = torch.linspace(0, drop_path_rate, depth).tolist()
        self.blocks = nn.ModuleList([Block(embed_dim, num_heads, mlp_ratio=mlp_ratio, drop_path=dpr[i]) for i in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)
        self.apply(_init_vit_weights)
        nn.init.normal_(self.mask_token, std=0.02)
        nn.init.normal_(self.background_token, std=0.02)

    @property
    def num_patches(self) -> int:
        return self.patch_embed.num_patches

    def embed(self, x: torch.Tensor, mask: torch.Tensor | None = None, valid_mask: torch.Tensor | None = None) -> torch.Tensor:
        tokens = self.patch_embed(x)
        if valid_mask is not None:
            bg = self.background_token.to(device=x.device, dtype=tokens.dtype).expand_as(tokens)
            tokens = torch.where(valid_mask.to(device=x.device).unsqueeze(-1), tokens, bg)
        if mask is not None:
            mask_tok = self.mask_token.to(device=x.device, dtype=tokens.dtype).expand_as(tokens)
            tokens = torch.where(mask.to(device=x.device).unsqueeze(-1), mask_tok, tokens)
        return tokens + self.pos_embed.to(device=x.device, dtype=x.dtype)

    def forward_tokens(self, tokens: torch.Tensor, capture_layers: tuple[int, ...] = ()) -> tuple[torch.Tensor, dict[int, torch.Tensor]]:
        hidden: dict[int, torch.Tensor] = {}
        for idx, blk in enumerate(self.blocks, start=1):
            tokens = blk(tokens)
            if idx in capture_layers:
                hidden[idx] = self.norm(tokens)
        tokens = self.norm(tokens)
        return tokens, hidden


class FrameMAEDecoder2D(nn.Module):
    def __init__(
        self,
        num_patches: int = 196,
        patch_size: int = 8,
        in_chans: int = 1,
        encoder_dim: int = 384,
        decoder_dim: int = 192,
        depth: int = 4,
        num_heads: int = 3,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        grid = int(num_patches**0.5)
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.encoder_to_decoder = nn.Linear(encoder_dim, decoder_dim, bias=False)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        self.background_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        pos = get_2d_sincos_pos_embed(decoder_dim, grid)
        self.register_buffer("pos_embed", pos, persistent=False)
        self.blocks = nn.ModuleList([Block(decoder_dim, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)])
        self.norm = nn.LayerNorm(decoder_dim)
        self.head = nn.Linear(decoder_dim, patch_size * patch_size * in_chans)
        self.apply(_init_vit_weights)
        nn.init.normal_(self.mask_token, std=0.02)
        nn.init.normal_(self.background_token, std=0.02)

    def forward(self, state_tokens: torch.Tensor, mask: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        b, n, _ = state_tokens.shape
        x = self.encoder_to_decoder(state_tokens)
        full = self.background_token.to(dtype=x.dtype).expand(b, n, -1).clone()
        mask_tok = self.mask_token.to(dtype=x.dtype).expand(b, n, -1)
        full = torch.where(valid_mask.unsqueeze(-1), x, full)
        full = torch.where(mask.unsqueeze(-1), mask_tok, full)
        full = full + self.pos_embed.to(device=x.device, dtype=x.dtype)
        for blk in self.blocks:
            full = blk(full)
        return self.head(self.norm(full))


def random_mask_in_roi(valid_mask: torch.Tensor, mask_ratio: float) -> torch.Tensor:
    """Randomly mask valid ROI patches, leaving invalid background unmasked."""

    b, n = valid_mask.shape
    noise = torch.rand(b, n, device=valid_mask.device)
    noise = noise.masked_fill(~valid_mask, -1.0)
    valid_counts = valid_mask.sum(dim=1)
    mask = torch.zeros(b, n, dtype=torch.bool, device=valid_mask.device)
    for i in range(b):
        count = int(valid_counts[i].item())
        if count <= 0:
            continue
        num_mask = max(1, int(round(count * mask_ratio)))
        ids = torch.topk(noise[i], k=num_mask, largest=True).indices
        mask[i, ids] = True
    return mask


class EchoFrameMAE(nn.Module):
    """Single-frame MAE body shared by RVM and TTT recurrent wrappers."""

    def __init__(self, **kwargs):
        super().__init__()
        img_size = int(kwargs.get("img_size", 112))
        patch_size = int(kwargs.get("patch_size", 8))
        in_chans = int(kwargs.get("in_chans", 1))
        embed_dim = int(kwargs.get("embed_dim", 384))
        depth = int(kwargs.get("depth", 12))
        num_heads = int(kwargs.get("num_heads", 6))
        decoder_embed_dim = int(kwargs.get("decoder_embed_dim", 192))
        decoder_depth = int(kwargs.get("decoder_depth", 4))
        decoder_num_heads = int(kwargs.get("decoder_num_heads", 3))
        mlp_ratio = float(kwargs.get("mlp_ratio", 4.0))
        drop_path_rate = float(kwargs.get("drop_path_rate", 0.0))
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.encoder = FrameMAEEncoder2D(img_size, patch_size, in_chans, embed_dim, depth, num_heads, mlp_ratio, drop_path_rate)
        self.decoder = FrameMAEDecoder2D(
            self.encoder.num_patches,
            patch_size,
            in_chans,
            embed_dim,
            decoder_embed_dim,
            decoder_depth,
            decoder_num_heads,
            mlp_ratio,
        )

    @property
    def num_patches(self) -> int:
        return self.encoder.num_patches

    def make_masks(self, x: torch.Tensor, mask_ratio: float, roi_mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        patch_roi = roi_to_patch_mask(roi_mask, self.img_size, self.patch_size)
        if patch_roi is None:
            patch_roi = torch.ones(x.shape[0], self.num_patches, dtype=torch.bool, device=x.device)
        else:
            patch_roi = patch_roi.to(device=x.device)
        return random_mask_in_roi(patch_roi, mask_ratio), patch_roi

    def encode_frames(
        self,
        frames: torch.Tensor,
        mask: torch.Tensor | None = None,
        valid_mask: torch.Tensor | None = None,
        capture_layers: tuple[int, ...] = (),
    ) -> tuple[torch.Tensor, dict[int, torch.Tensor]]:
        tokens = self.encoder.embed(frames, mask=mask, valid_mask=valid_mask)
        return self.encoder.forward_tokens(tokens, capture_layers=capture_layers)

    def decode_frames(self, state_tokens: torch.Tensor, mask: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        return self.decoder(state_tokens, mask, valid_mask)

    def target_patches(self, frames: torch.Tensor) -> torch.Tensor:
        return patchify(frames, self.patch_size)
