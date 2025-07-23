
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from einops import rearrange
# from timm.models.layers import trunc_normal_
# from timm.models.layers import to_2tuple
# from timm.models.layers import DropPath



# class TransformerBottleneck(nn.Module):
#     def __init__(self, input_resolution, dim, drop_path=0.):
#         super().__init__()
#         # 使用 PatchMerging 进行下采样，通道数保持不变（已修改 reduction）
#         self.patch_merging = PatchMerging(input_resolution, dim, norm_layer=nn.LayerNorm)
#         new_resolution = (input_resolution[0] // 2, input_resolution[1] // 2)
#         # 使用 BasicLayer 进行 Transformer 特征提取
#         self.transformer_layer = BasicLayer(
#             dim=dim, 
#             input_resolution=new_resolution,
#             depth=2, 
#             num_heads=8, 
#             window_size=7, 
#             mlp_ratio=4., 
#             qkv_bias=True, 
#             qk_scale=None, 
#             drop=0., 
#             attn_drop=0., 
#             drop_path=drop_path,
#             norm_layer=nn.LayerNorm,
#             downsample=None,
#             use_checkpoint=False)
#         # 使用 PatchExpand 将下采样后的特征上采样回原分辨率
#         self.patch_expand = PatchExpand(new_resolution, dim, dim_scale=2, norm_layer=nn.LayerNorm)
#     def forward(self, x):
#         # x: (B, H*W, dim) 其中 H,W 为原始分辨率（例如32x32）
#         x_merged = self.patch_merging(x)         # (B, (H/2)*(W/2), dim)
#         x_trans = self.transformer_layer(x_merged) # (B, (H/2)*(W/2), dim)
#         x_expanded = self.patch_expand(x_trans)    # (B, H*W, dim)
#         return x_expanded


# class MoEFFNGating(nn.Module):
#     def __init__(self, dim, hidden_dim, num_experts):
#         super(MoEFFNGating, self).__init__()
#         self.gating_network = nn.Linear(dim, dim)
#         self.experts = nn.ModuleList([nn.Sequential(
#             nn.Linear(dim, hidden_dim),
#             nn.GELU(),
#             nn.Linear(hidden_dim, dim)) for _ in range(num_experts)])
#     def forward(self, x):
#         weights = self.gating_network(x)
#         weights = torch.nn.functional.softmax(weights, dim=-1)
#         outputs = [expert(x) for expert in self.experts]
#         outputs = torch.stack(outputs, dim=0)
#         outputs = (weights.unsqueeze(0) * outputs).sum(dim=0)
#         return outputs

# class Mlp(nn.Module):
#     def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
#         super().__init__()
#         out_features = out_features or in_features
#         hidden_features = hidden_features or in_features
#         self.fc1 = nn.Linear(in_features, hidden_features)
#         self.act = act_layer()
#         self.fc2 = nn.Linear(hidden_features, out_features)
#         self.drop = nn.Dropout(drop)
#     def forward(self, x):
#         x = self.fc1(x)
#         x = self.act(x)
#         x = self.drop(x)
#         x = self.fc2(x)
#         x = self.drop(x)
#         return x

# def window_partition(x, window_size):
#     """
#     将输入特征图划分为窗口
#     Args:
#         x: (B, H, W, C)
#         window_size (int): 窗口大小
#     Returns:
#         windows: (num_windows*B, window_size, window_size, C)
#     """
#     B, H, W, C = x.shape
#     x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
#     windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
#     return windows

# def window_reverse(windows, window_size, H, W):
#     """
#     将窗口还原为原始特征图
#     Args:
#         windows: (num_windows*B, window_size, window_size, C)
#         window_size (int): 窗口大小
#         H (int): 图像高度
#         W (int): 图像宽度
#     Returns:
#         x: (B, H, W, C)
#     """
#     B = int(windows.shape[0] / (H * W / window_size / window_size))
#     x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
#     x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
#     return x

# class WindowAttention(nn.Module):
#     r""" 基于窗口的多头自注意力模块，支持相对位置偏差
#     """
#     def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
#         super().__init__()
#         self.dim = dim
#         self.window_size = window_size  # (Wh, Ww)
#         self.num_heads = num_heads
#         head_dim = dim // num_heads
#         self.scale = qk_scale or head_dim ** -0.5

#         # 定义相对位置偏差参数表
#         self.relative_position_bias_table = nn.Parameter(
#             torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))
        
#         # 计算窗口内每对 token 的相对位置索引
#         coords_h = torch.arange(self.window_size[0])
#         coords_w = torch.arange(self.window_size[1])
#         coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))
#         coords_flatten = torch.flatten(coords, 1)  # (2, Wh*Ww)
#         relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # (2, Wh*Ww, Wh*Ww)
#         relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # (Wh*Ww, Wh*Ww, 2)
#         relative_coords[:, :, 0] += self.window_size[0] - 1
#         relative_coords[:, :, 1] += self.window_size[1] - 1
#         relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
#         relative_position_index = relative_coords.sum(-1)  # (Wh*Ww, Wh*Ww)
#         self.register_buffer("relative_position_index", relative_position_index)

#         self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
#         self.attn_drop = nn.Dropout(attn_drop)
#         self.proj = nn.Linear(dim, dim)
#         self.proj_drop = nn.Dropout(proj_drop)
#         trunc_normal_(self.relative_position_bias_table, std=.02)
#         self.softmax = nn.Softmax(dim=-1)
#     def forward(self, x, mask=None):
#         B_, N, C = x.shape
#         qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
#         q, k, v = qkv[0], qkv[1], qkv[2]
#         q = q * self.scale
#         attn = (q @ k.transpose(-2, -1))
#         relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
#             self.window_size[0] * self.window_size[1],
#             self.window_size[0] * self.window_size[1], -1)
#         relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
#         attn = attn + relative_position_bias.unsqueeze(0)
#         if mask is not None:
#             nW = mask.shape[0]
#             attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
#             attn = attn.view(-1, self.num_heads, N, N)
#             attn = self.softmax(attn)
#         else:
#             attn = self.softmax(attn)
#         attn = self.attn_drop(attn)
#         x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
#         x = self.proj(x)
#         x = self.proj_drop(x)
#         return x
#     def extra_repr(self) -> str:
#         return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'
#     def flops(self, N):
#         flops = 0
#         flops += N * self.dim * 3 * self.dim
#         flops += self.num_heads * N * (self.dim // self.num_heads) * N
#         flops += self.num_heads * N * N * (self.dim // self.num_heads)
#         flops += N * self.dim * self.dim
#         return flops

# class SwinTransformerBlock(nn.Module):
#     r""" Swin Transformer Block, 包含窗口自注意力和 MLP 两部分
#     """
#     def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
#                  mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
#                  act_layer=nn.GELU, norm_layer=nn.LayerNorm):
#         super().__init__()
#         self.dim = dim
#         self.input_resolution = input_resolution
#         self.num_heads = num_heads
#         self.window_size = window_size
#         self.shift_size = shift_size
#         self.mlp_ratio = mlp_ratio
#         # 如果 H 或 W 不是 window_size 的整数倍，则调整 window_size
#         H, W = self.input_resolution
#         if H % window_size != 0 or W % window_size != 0:
#             new_window_size = min(H, W)
#             print(f"Warning: 调整 window_size={window_size} -> {new_window_size} 以匹配输入尺寸 ({H}, {W})")
#             self.window_size = new_window_size

#         self.norm1 = norm_layer(dim)
#         self.attn = WindowAttention(
#             dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
#             qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
#         self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
#         self.norm2 = norm_layer(dim)
#         mlp_hidden_dim = int(dim * mlp_ratio)
#         self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
#         if self.shift_size > 0:
#             H, W = self.input_resolution
#             img_mask = torch.zeros((1, H, W, 1))
#             h_slices = (slice(0, -self.window_size),
#                         slice(-self.window_size, -self.shift_size),
#                         slice(-self.shift_size, None))
#             w_slices = (slice(0, -self.window_size),
#                         slice(-self.window_size, -self.shift_size),
#                         slice(-self.shift_size, None))
#             cnt = 0
#             for h in h_slices:
#                 for w in w_slices:
#                     img_mask[:, h, w, :] = cnt
#                     cnt += 1
#             mask_windows = window_partition(img_mask, self.window_size)
#             mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
#             attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
#             attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
#         else:
#             attn_mask = None
#         self.register_buffer("attn_mask", attn_mask)
#     def forward(self, x):
#         H, W = self.input_resolution
#         B, L, C = x.shape
#         assert L == H * W, "input feature has wrong size"
#         shortcut = x
#         x = self.norm1(x)
#         x = x.view(B, H, W, C)
#         if self.shift_size > 0:
#             shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
#         else:
#             shifted_x = x
#         x_windows = window_partition(shifted_x, self.window_size)
#         x_windows = x_windows.view(-1, self.window_size * self.window_size, C)
#         attn_windows = self.attn(x_windows, mask=self.attn_mask)
#         attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
#         shifted_x = window_reverse(attn_windows, self.window_size, H, W)
#         if self.shift_size > 0:
#             x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
#         else:
#             x = shifted_x
#         x = x.view(B, H * W, C)
#         x = shortcut + self.drop_path(x)
#         x = x + self.drop_path(self.mlp(self.norm2(x)))
#         return x
#     def extra_repr(self) -> str:
#         return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
#                f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"
#     def flops(self):
#         flops = 0
#         H, W = self.input_resolution
#         flops += self.dim * H * W
#         nW = H * W / self.window_size / self.window_size
#         flops += nW * self.attn.flops(self.window_size * self.window_size)
#         flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
#         flops += self.dim * H * W
#         return flops

# class PatchMerging(nn.Module):
#     r""" Patch Merging 层，用于下采样特征
#     """
#     def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
#         super().__init__()
#         self.input_resolution = input_resolution
#         self.dim = dim
#         self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
#         self.norm = norm_layer(4 * dim)
#     def forward(self, x):
#         """
#         x: (B, H*W, C)
#         """
#         H, W = self.input_resolution
#         B, L, C = x.shape
#         assert L == H * W, "input feature has wrong size"
#         assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."
#         x = x.view(B, H, W, C)
#         x0 = x[:, 0::2, 0::2, :]
#         x1 = x[:, 1::2, 0::2, :]
#         x2 = x[:, 0::2, 1::2, :]
#         x3 = x[:, 1::2, 1::2, :]
#         x = torch.cat([x0, x1, x2, x3], -1)
#         x = x.view(B, -1, 4 * C)
#         x = self.norm(x)
#         x = self.reduction(x)
#         return x
#     def extra_repr(self) -> str:
#         return f"input_resolution={self.input_resolution}, dim={self.dim}"
#     def flops(self):
#         H, W = self.input_resolution
#         flops = H * W * self.dim
#         flops += (H // 2) * (W // 2) * 4 * self.dim * 2 * self.dim
#         return flops

# class PatchExpand(nn.Module):
#     r""" Patch Expand 层，用于上采样特征
#     """
#     def __init__(self, input_resolution, dim, dim_scale=2, norm_layer=nn.LayerNorm):
#         super().__init__()
#         self.input_resolution = input_resolution
#         self.dim = dim
#         self.expand = nn.Linear(dim, 2 * dim, bias=False) if dim_scale == 2 else nn.Identity()
#         self.norm = norm_layer(dim // dim_scale)
#     def forward(self, x):
#         """
#         x: (B, H*W, C)
#         """
#         H, W = self.input_resolution
#         x = self.expand(x)
#         B, L, C = x.shape
#         assert L == H * W, "input feature has wrong size"
#         x = x.view(B, H, W, C)
#         x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=2, p2=2, c=C // 4)
#         x = x.view(B, -1, C // 4)
#         x = self.norm(x)
#         return x

# # class FinalPatchExpand_X4(nn.Module):
# #     r""" 最终上采样层，将特征图上采样4倍
# #     """
# #     def __init__(self, input_resolution, dim, dim_scale=4, norm_layer=nn.LayerNorm):
# #         super().__init__()
# #         self.input_resolution = input_resolution
# #         self.dim = dim
# #         self.dim_scale = dim_scale
# #         self.expand = nn.Linear(dim, 16 * dim, bias=False)
# #         self.output_dim = dim
# #         self.norm = norm_layer(self.output_dim)
# #     def forward(self, x):
# #         """
# #         x: (B, H*W, C)
# #         """
# #         H, W = self.input_resolution
# #         x = self.expand(x)
# #         B, L, C = x.shape
# #         assert L == H * W, "input feature has wrong size"
# #         x = x.view(B, H, W, C)
# #         x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=self.dim_scale, p2=self.dim_scale,
# #                       c=C // (self.dim_scale ** 2))
# #         x = x.view(B, -1, self.output_dim)
# #         x = self.norm(x)
# #         return x

# class BasicLayer(nn.Module):
#     """ 一个阶段的基础 Swin Transformer 层，由多个 SwinTransformerBlock 组成，可选下采样
#     """
#     def __init__(self, dim, input_resolution, depth, num_heads, window_size,
#                  mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
#                  drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False):
#         super().__init__()
#         self.dim = dim
#         self.input_resolution = input_resolution
#         self.depth = depth
#         self.use_checkpoint = use_checkpoint
#         self.blocks = nn.ModuleList([
#             SwinTransformerBlock(dim=1024, input_resolution=input_resolution,
#                                  num_heads=num_heads, window_size=window_size,
#                                  shift_size=0 if (i % 2 == 0) else window_size // 2,
#                                  mlp_ratio=mlp_ratio,
#                                  qkv_bias=qkv_bias, qk_scale=qk_scale,
#                                  drop=drop, attn_drop=attn_drop,
#                                  drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
#                                  norm_layer=norm_layer)
#             for i in range(depth)])
#         if downsample is not None:
#             self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
#         else:
#             self.downsample = None
#     def forward(self, x):
#         for blk in self.blocks:
#             if self.use_checkpoint:
#                 x = torch.utils.checkpoint.checkpoint(blk, x)
#             else:
#                 x = blk(x)
#         if self.downsample is not None:
#             x = self.downsample(x)
#         return x
#     def extra_repr(self) -> str:
#         return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"
#     def flops(self):
#         flops = 0
#         for blk in self.blocks:
#             flops += blk.flops()
#         if self.downsample is not None:
#             flops += self.downsample.flops()
#         return flops





# # class BasicLayer_up(nn.Module):
# #     """ 一个阶段的上采样层，由多个 SwinTransformerBlock 组成，最后进行上采样
# #     """
# #     def __init__(self, dim, input_resolution, depth, num_heads, window_size,
# #                  mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
# #                  drop_path=0., norm_layer=nn.LayerNorm, upsample=None, use_checkpoint=False):
# #         super().__init__()
# #         self.dim = dim
# #         self.input_resolution = input_resolution
# #         self.depth = depth
# #         self.use_checkpoint = use_checkpoint
# #         self.blocks = nn.ModuleList([
# #             SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
# #                                  num_heads=num_heads, window_size=window_size,
# #                                  shift_size=0 if (i % 2 == 0) else window_size // 2,
# #                                  mlp_ratio=mlp_ratio,
# #                                  qkv_bias=qkv_bias, qk_scale=qk_scale,
# #                                  drop=drop, attn_drop=attn_drop,
# #                                  drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
# #                                  norm_layer=norm_layer)
# #             for i in range(depth)])
# #         if upsample is not None:
# #             self.upsample = PatchExpand(input_resolution, dim=dim, dim_scale=2, norm_layer=norm_layer)
# #         else:
# #             self.upsample = None
# #     def forward(self, x):
# #         for blk in self.blocks:
# #             if self.use_checkpoint:
# #                 x = torch.utils.checkpoint.checkpoint(blk, x)
# #             else:
# #                 x = blk(x)
# #         if self.upsample is not None:
# #             x = self.upsample(x)
# #         return x

# # class PatchEmbed(nn.Module):
# #     r""" 图像块嵌入，将输入图像转换为一系列嵌入向量

# #     Args:
# #         img_size (int): 图像尺寸。默认：256.
# #         patch_size (int): 图像块尺寸。默认：4.
# #         in_chans (int): 输入图像通道数。默认：3.
# #         embed_dim (int): 线性投影输出通道数。默认：96.
# #         norm_layer (nn.Module, optional): 归一化层。默认：None.
# #     """
# #     def __init__(self, img_size=256, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
# #         super().__init__()
# #         img_size = to_2tuple(img_size)
# #         patch_size = to_2tuple(patch_size)
# #         patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
# #         self.img_size = img_size
# #         self.patch_size = patch_size
# #         self.patches_resolution = patches_resolution
# #         self.num_patches = patches_resolution[0] * patches_resolution[1]
# #         self.in_chans = in_chans
# #         self.embed_dim = embed_dim
# #         self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
# #         if norm_layer is not None:
# #             self.norm = norm_layer(embed_dim)
# #         else:
# #             self.norm = None
# #     def forward(self, x):
# #         B, C, H, W = x.shape
# #         assert H == self.img_size[0] and W == self.img_size[1], \
# #             f"输入图像尺寸 ({H}*{W}) 与模型要求 ({self.img_size[0]}*{self.img_size[1]}) 不匹配。"
# #         x = self.proj(x).flatten(2).transpose(1, 2)  # (B, Ph*Pw, C)
# #         if self.norm is not None:
# #             x = self.norm(x)
# #         return x
# #     def flops(self):
# #         Ho, Wo = self.patches_resolution
# #         flops = Ho * Wo * self.embed_dim * self.in_chans * (self.patch_size[0] * self.patch_size[1])
# #         if self.norm is not None:
# #             flops += Ho * Wo * self.embed_dim
# #         return flops

# # class PatchExpand(nn.Module):
# #     r""" Patch Expand 层，用于上采样特征
# #     """
# #     def __init__(self, input_resolution, dim, dim_scale=2, norm_layer=nn.LayerNorm):
# #         super().__init__()
# #         self.input_resolution = input_resolution
# #         self.dim = dim
# #         self.expand = nn.Linear(dim, 2 * dim, bias=False) if dim_scale == 2 else nn.Identity()
# #         self.norm = norm_layer(dim // dim_scale)
# #     def forward(self, x):
# #         """
# #         x: (B, H*W, C)
# #         """
# #         H, W = self.input_resolution
# #         x = self.expand(x)
# #         B, L, C = x.shape
# #         assert L == H * W, "输入特征尺寸不匹配"
# #         x = x.view(B, H, W, C)
# #         x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=2, p2=2, c=C // 4)
# #         x = x.view(B, -1, C // 4)
# #         x = self.norm(x)
# #         return x

# # class FinalPatchExpand_X4(nn.Module):
# #     r""" 最终上采样层，将特征图上采样4倍
# #     """
# #     def __init__(self, input_resolution, dim, dim_scale=4, norm_layer=nn.LayerNorm):
# #         super().__init__()
# #         self.input_resolution = input_resolution
# #         self.dim = dim
# #         self.dim_scale = dim_scale
# #         self.expand = nn.Linear(dim, 16 * dim, bias=False)
# #         self.output_dim = dim
# #         self.norm = norm_layer(self.output_dim)
# #     def forward(self, x):
# #         """
# #         x: (B, H*W, C)
# #         """
# #         H, W = self.input_resolution
# #         x = self.expand(x)
# #         B, L, C = x.shape
# #         assert L == H * W, "输入特征尺寸不匹配"
# #         x = x.view(B, H, W, C)
# #         x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=self.dim_scale, p2=self.dim_scale,
# #                       c=C // (self.dim_scale ** 2))
# #         x = x.view(B, -1, self.output_dim)
# #         x = self.norm(x)
# #         return x