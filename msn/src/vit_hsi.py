
import torch
import torch.nn as nn
import torch.nn.functional as F


from itertools import product
import random

from src.pos_embed import get_3d_sincos_pos_embed, get_1d_spectral_pos_embed_from_grid, get_2d_sincos_pos_embed_for_vanilla


# buscar as funcoes
from einops import rearrange


class BlockwisePatchEmbed(nn.Module):
    """
    Blockwise Hyperspectral Image (HSI) to Patch Embedding Layer.

    This module implements a "blockwise" patching strategy inspired by the idea
    of learning specialized embeddings for different parts of the spectrum.

    Core Logic:
    1.  The input HSI cube's spectral dimension is split into several contiguous
        "blocks".
    2.  A separate 3D Convolutional layer is assigned to each block.
    3.  Each convolution processes its corresponding spectral block, creating
        patches and embedding them in a single, efficient step. This allows the
        model to learn different features for different spectral regions.
    4.  The patch embeddings from all blocks are then concatenated to form the
        final sequence of tokens for a Transformer.
    5.  A final Layer Normalization is applied to the full sequence.
    """
    def __init__(self,
                 channels = 100,
                 patch_size=   (5, 5, 10),
                 in_chans: int = 1,
                 embed_dim: int = 768):
        """
        Args:
            img_size (Tuple[int, int, int]): The size of the input HSI cube in
                (Height, Width, Bands).
            patch_size (Tuple[int, int, int]): The size of each patch in
                (Height, Width, Bands). The number of bands in the image must
                be divisible by the number of bands in the patch.
            in_chans (int): The number of input channels for the HSI.
                Typically 1 for a single HSI cube.
            embed_dim (int): The dimensionality of the output patch embeddings.
        """
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.in_chans = in_chans

        img_b = channels
        patch_h, patch_w, patch_b = self.patch_size

        assert img_b % patch_b == 0, \
            f"Image bands ({img_b}) must be divisible by patch bands ({patch_b})."

        self.num_blocks = img_b // patch_b

        # The convolution kernel size is defined in (Bands, Height, Width)
        # to match the PyTorch Conv3d input format (B, C, D, H, W).
        kernel_size = (patch_b, patch_h, patch_w)

        # Create a separate 3D convolution projection for each spectral block.
        # Each convolution will process a chunk of the HSI with `patch_b` bands.
        self.proj= nn.ModuleList([
            nn.Conv3d(
                in_chans,
                embed_dim,
                kernel_size=kernel_size,
                stride=kernel_size
            ) for _ in range(self.num_blocks)
        ])

        # A single Layer Normalization applied after concatenating all embeddings.
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Processes the input HSI data block by block.

        Args:
            x (torch.Tensor): Input HSI tensor.
                Shape: (B, C, T, H, W), where T is the total number of bands.

        Returns:
            torch.Tensor: The final concatenated and normalized patch embeddings.
                          Shape: (B, N, E), where:
                          B is the batch size.
                          N is the total number of patches from all blocks.
                          E is the embedding dimension.
        """
        # Split the input tensor into blocks along the spectral dimension (dim=2).
        # Each chunk will have the size (B, C, patch_bands, H, W).
        chunks = torch.chunk(x, self.num_blocks, dim=2)

        block_embeddings = []
        for i, chunk in enumerate(chunks):
            # Apply the specific projection for this block.
            # Input:  (B, C, patch_b, H, W)
            # Output: (B, E, 1, H/patch_h, W/patch_w)
            embedded_chunk = self.proj[i](chunk)

            # Reshape to (B, N_spatial, E) for this block
            # N_spatial = (H/patch_h) * (W/patch_w)
            embedded_chunk = embedded_chunk.flatten(3)
            embedded_chunk = torch.einsum('bcts->btsc', embedded_chunk)
            block_embeddings.append(embedded_chunk)

        # Concatenate the embeddings from all blocks along the sequence dimension.
        embeddings = torch.cat(block_embeddings, dim=1)

        # Apply final layer normalization.
        embeddings = self.norm(embeddings)


        self.output_size = embeddings.shape

        return embeddings


class PatchEmbed(nn.Module):
    """ HSI to Patch Embedding
    """
    def __init__(self, 
                 patch_size = (5,5,10),
                 in_chans = 1,
                 embed_dim = 768):
        
        
        super().__init__()

        # Define the kernel size for the 3D convolution, which matches the patch size
        kernel_size = (patch_size[2], patch_size[0], patch_size[1])

        # Create the 3D convolution layer to project patches into a higher-dimensional space
        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=kernel_size, stride=kernel_size)

    def forward(self, x):
        """
        Processes the input HSI data.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, T, H, W).
                B: batch size
                C: channels (typically 1 for HSI)
                T: number of bands (time)
                H: image height
                W: image width
        
        Returns:
            torch.Tensor: The output tensor of patch embeddings,
                          ready for a Transformer model.
        """



        # Apply the 3D convolution to convert HSI patches into embeddings
        # Output shape: (B, embed_dim, T', H', W')
        x = self.proj(x)
        
        # Flatten the spatial and spectral dimensions into a single sequence
        x = x.flatten(3)
        
        # Rearrange the dimensions to get the desired output shape for a Transformer
        x = torch.einsum('bcts->btsc', x) # [B, T', H'* W',embed_dim]

        # Store the output size for debugging/validation
        self.output_size = x.shape
        return x



class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
       #assert attn_drop == 0.0  # do not use
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def adjust_drop_rate(self, drop_rate=0.):
        self.proj_drop = nn.Dropout(drop_rate)

    def forward(self, x, attn_bias=None):
        B, N, C = x.shape
        q = (
            self.q(x)
            .reshape(B, N, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )
        k = (
            self.k(x)
            .reshape(B, N, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )
        v = (
            self.v(x)
            .reshape(B, N, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )

        attn = (q @ k.transpose(-2, -1)) * self.scale
        if attn_bias is not None:
            attn += attn_bias
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        x = x.view(B, -1, C)
        return x


class SwiGLU(nn.Module):
    def __init__(self, dim, hidden_dim, multiple_of=4, dropout=0.):
        super().__init__()
        hidden_dim = int(multiple_of * ((2 * hidden_dim // 3 + multiple_of - 1) // multiple_of))
        self.w1 = nn.Linear(dim, hidden_dim, bias=True)
        self.w2 = nn.Linear(hidden_dim, dim, bias=True)
        self.w3 = nn.Linear(dim, hidden_dim, bias=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        return f'drop_prob={round(self.drop_prob,3):0.3f}'


class Block(nn.Module):
    """
    Transformer Block with specified Attention function
    """

    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        attn_func=Attention,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = attn_func(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp =  SwiGLU(dim, mlp_hidden_dim, mlp_ratio, drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x



class VisionTransformerHSI(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self,rand_size = (60,60,150) , focal_size = (30,30,150),patch_size=(5,5,10), in_chans=1, embed_dim=768, depth=12,
                 num_heads=12 ,mlp_ratio=4., qkv_bias=True, drop_rate=0., attn_drop_rate=0., drop_path_rate=0., trunc_init=True, bwpe=False,
                 norm_layer=nn.LayerNorm, **kwargs):
        super().__init__()
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.bwpe = bwpe

        self.rand_size = rand_size
        self.focal_size = focal_size
        self.patch_size = patch_size

        assert rand_size[0] % patch_size[0] == 0 and rand_size[1] % patch_size[1] == 0 and rand_size[2] % patch_size[2] == 0, \
            f"Image dimensions must be divisible by the patch size. Got image size {rand_size} and patch size {patch_size}."
        assert focal_size[0] % patch_size[0] == 0 and focal_size[1] % patch_size[1] == 0 and focal_size[2] % patch_size[2] == 0, \
            f"Image dimensions must be divisible by the patch size. Got image size {focal_size} and patch size {patch_size}."

        


        if self.bwpe:
            self.patch_embed = BlockwisePatchEmbed(channels = rand_size[2],
                                                   patch_size = patch_size, 
                                          in_chans   = in_chans, 
                                          embed_dim  = embed_dim)
            
        else:
        
            self.patch_embed = PatchEmbed(patch_size = patch_size,
                                          in_chans = in_chans,
                                          embed_dim = embed_dim)
            

        num_rand_patches = (rand_size[0] // patch_size[0]) * (rand_size[1] // patch_size[1]) * (rand_size[2] // patch_size[2])
        num_focal_patches = focal_size[0] // patch_size[0] * focal_size[1] // patch_size[1] * focal_size[2] // patch_size[2]

        self.rand_pos_embed = nn.Parameter(torch.zeros(1, num_rand_patches , embed_dim))
        self.foc_pos_embed = nn.Parameter(torch.zeros(1, num_focal_patches, embed_dim))

        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))


        self.vit_spatial = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=drop_path_rate,
                norm_layer=norm_layer)
            for i in range(depth)])

        self.vit_fusion = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=drop_path_rate,
                norm_layer=norm_layer)
            for i in range(3)])

        self.vit_spectral = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=drop_path_rate,
                norm_layer=norm_layer)
            for i in range(depth)])

        self.trunc_init = trunc_init

        self.norm = norm_layer(embed_dim)

        # get all x and return on

        self.initialize_weights()


    def initialize_weights(self):
        # rand positional embedding
        rand_pos_embed = get_3d_sincos_pos_embed(self.embed_dim,
                                                 self.rand_size[2] // self.patch_size[2],
                                                 self.rand_size[0] // self.patch_size[0], 
                                                 self.rand_size[1] // self.patch_size[1])


        self.rand_pos_embed.data.copy_(rand_pos_embed)
        self.rand_pos_embed.requires_grad = False

        # focal positional embedding
        focal_pos_embed = get_3d_sincos_pos_embed(self.embed_dim,
                                                   self.focal_size[2] // self.patch_size[2],
                                                   self.focal_size[0] // self.patch_size[0],
                                                   self.focal_size[1] // self.patch_size[1])
        self.foc_pos_embed.data.copy_(focal_pos_embed)
        self.foc_pos_embed.requires_grad = False

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        if self.bwpe:
            for proj in self.patch_embed.proj:
                nn.init.xavier_uniform_(proj.weight)
                nn.init.constant_(proj.bias, 0)
        else:

            w = self.patch_embed.proj.weight.data

            if self.trunc_init:
                torch.nn.init.trunc_normal_(w)
                torch.nn.init.trunc_normal_(self.mask_token, std=0.02)
            else:
                torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
                torch.nn.init.normal_(self.mask_token, std=0.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            if self.trunc_init:
                nn.init.trunc_normal_(m.weight, std=0.02)
            else:
                torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)



    def patchify(self, imgs):
        N, _, T, H, W = imgs.shape
        p = self.patch_embed.patch_size[0]
        u = self.patch_embed.patch_size[2]
        assert H == W and H % p == 0 and T % u == 0
        h = w = H // p
        t = T // u

        x = imgs.reshape(shape=(N, 1, t, u, h, p, w, p))
        x = torch.einsum("nctuhpwq->nthwupqc", x)
        x = x.reshape(shape=(N, t * h * w, u * p**2 * 1))
        self.patch_info = (N, T, H, W, p, u, t, h, w)
        return x
    

    def unpatchify(self, x):
        N, T, H, W, p, u, t, h, w = self.patch_info

        x = x.reshape(shape=(N, t, h, w, u, p, p, 1))

        x = torch.einsum("nthwupqc->nctuhpwq", x)
        imgs = x.reshape(shape=(N, 1, T, H, W))
        return imgs
    
    def get_dim_patches(self, T, L, mask_ratio):
        # Handle edge cases where T or L are < 2 (no valid pairs)
        if T < 2 or L < 2:
            len_t = 1 if T < 2 else T
            len_l = 1 if L < 2 else L
            return len_t, len_l

        # Build explicit list of (t, l) pairs
        pairs = [(i, j) for i in range(2, T + 1) for j in range(2, L + 1)]
        len_all = torch.tensor(pairs, dtype=torch.long, device='cpu')

        len_keep = float((1 - mask_ratio) * T * L)

        # Compute product for each candidate and find closest to len_keep
        lens = (len_all[:, 0].float() * len_all[:, 1].float())
        lens_diff = torch.abs(lens - len_keep)

        # indices where difference is minimal
        ind = torch.where(lens_diff == torch.min(lens_diff))[0]
        # choose randomly among ties
        r = torch.randint(0, len(ind), (1,)).item()
        index = len_all[ind[r]]

        len_t = int(index[0].item())
        len_l = int(index[1].item())
        return len_t, len_l
    
    def spatial_spectral_masking(self, x, T, L, mask_ratio):
        N, _, D = x.shape

        mask_1 = torch.ones([N, T * L], device=x.device)
        mask_2 = torch.ones([N, T * L], device=x.device)

        self.len_t, self.len_l = self.get_dim_patches(T, L, mask_ratio)
        len_keep_1 = self.len_t * L
        len_keep_2 = self.len_l * T
        len_keep = self.len_t * self.len_l

        noise_1 = torch.rand(N, T, device=x.device)
        noise_1 = noise_1.repeat_interleave(L, 1)
        ids_shuffle = torch.argsort(noise_1, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        mask_1[:, :len_keep_1] = 0
        mask_1 = torch.gather(mask_1, dim=1, index=ids_restore)

        noise_2 = torch.rand(N, L, device=x.device)
        noise_2 = noise_2.repeat(1, T)
        ids_shuffle = torch.argsort(noise_2, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        mask_2[:, :len_keep_2] = 0
        mask_2 = torch.gather(mask_2, dim=1, index=ids_restore)

        mask_all = mask_1 + mask_2 + torch.linspace(0, 0.5, T * L, device=x.device).unsqueeze(0).repeat(N, 1)

        # sort noise for each sample
        ids_shuffle = torch.argsort(mask_all, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, T * L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)
        return x_masked, mask, ids_restore, ids_keep
    

    def forward(self, x, mask_ratio = 0.0):
        #embed patches
        x = self.patch_embed(x) 

        N, T, L, C = x.shape  # T: num_patches_t, L: num_patches_l


        x = x.reshape(N, T * L, C)

        
        x, mask, ids_restore, ids_keep = self.spatial_spectral_masking(x, T, L, mask_ratio)

        x = x.view(N, -1, C)



        if  (T*L ) == self.rand_pos_embed.shape[1]:
            pos_embed = self.rand_pos_embed[:, :, :].expand(N, -1, -1)

        else:
            pos_embed = self.foc_pos_embed[:, :, :].expand(N, -1, -1)  


        # add pos embedding nos patches
        pos_embed = torch.gather(
            pos_embed,
            dim=1, 
            index=ids_keep.unsqueeze(-1).repeat(1, 1, pos_embed.shape[2])
        )
        x = x + pos_embed


        # ---- separa para cada ramo ----
        # rearrange patches
        x_spatial = rearrange(x, 'b (t l) c -> (b t) l c', t=self.len_t, l=self.len_l)
        x_spectral = rearrange(x, 'b (t l) c -> (b l) t c', t=self.len_t, l=self.len_l)



        # passa pelos blocos
        for blk in self.vit_spatial:
            x_spatial = blk(x_spatial)

        for blk in self.vit_spectral:
            x_spectral = blk(x_spectral)



        x_spatial = rearrange(x_spatial, '(b t) l c -> b (t l) c', b=N, t=self.len_t)
        x_spectral = rearrange(x_spectral, '(b l) t c -> b (t l) c', b=N, l=self.len_l)

        x = x_spectral + x_spatial


        for blk in self.vit_fusion:
            x = blk(x)



        x = self.norm(x)

        # get cls token
        x = x.mean(dim=1)

        return x
    

class VisionTransformerSpec(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self,rand_size = (60,60,150), focal_size = (30,30,150), patch_size=(5,5,10), in_chans=1, embed_dim=768, depth=12,
                 num_heads=12 ,mlp_ratio=4., qkv_bias=True, drop_rate=0., attn_drop_rate=0., drop_path_rate=0., trunc_init=True, bwpe=False,
                 norm_layer=nn.LayerNorm, **kwargs):
        super().__init__()
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.bwpe = bwpe

        self.rand_size = rand_size
        self.focal_size = focal_size # Adicionado para checagem
        self.patch_size = patch_size


        assert rand_size[0] % patch_size[0] == 0 and rand_size[1] % patch_size[1] == 0 and rand_size[2] % patch_size[2] == 0, \
            f"Image dimensions must be divisible by the patch size. Got image size {rand_size} and patch size {patch_size}."
        
        # Adicionada checagem para focal_size também
        assert focal_size[0] % patch_size[0] == 0 and focal_size[1] % patch_size[1] == 0 and focal_size[2] % patch_size[2] == 0, \
            f"Image dimensions must be divisible by the patch size. Got image size {focal_size} and patch size {patch_size}."
        


        
        if self.bwpe:
            self.patch_embed = BlockwisePatchEmbed(channels = rand_size[2],
                                                patch_size = patch_size, 
                                                    in_chans   = in_chans, 
                                                    embed_dim  = embed_dim)
        else:
            self.patch_embed = PatchEmbed(patch_size = patch_size,
                                        in_chans = in_chans,
                                        embed_dim = embed_dim)
        
        num_rand_patches = (rand_size[0] // patch_size[0]) * (rand_size[1] // patch_size[1]) * (rand_size[2] // patch_size[2])
        num_focal_patches = focal_size[0] // patch_size[0] * focal_size[1] // patch_size[1] * focal_size[2] // patch_size[2]

        self.rand_pos_embed = nn.Parameter(torch.zeros(1, num_rand_patches , embed_dim))
        self.foc_pos_embed = nn.Parameter(torch.zeros(1, num_focal_patches, embed_dim))

        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.spec_transformer = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=drop_path_rate,
                norm_layer=norm_layer)
            for i in range(depth)])
        

        self.trunc_init = trunc_init

        self.norm = norm_layer(embed_dim)

        # get all x and return on

        self.initialize_weights()


    def initialize_weights(self):
        # rand positional embedding
        rand_pos_embed = get_1d_spectral_pos_embed_from_grid(self.embed_dim,
                                                 self.rand_size[2] // self.patch_size[2],
                                                 self.rand_size[0] // self.patch_size[0], 
                                                 self.rand_size[1] // self.patch_size[1])


        self.rand_pos_embed.data.copy_(rand_pos_embed)
        self.rand_pos_embed.requires_grad = False

        # focal positional embedding
        focal_pos_embed = get_1d_spectral_pos_embed_from_grid(self.embed_dim,
                                                   self.focal_size[2] // self.patch_size[2],
                                                   self.focal_size[0] // self.patch_size[0],
                                                   self.focal_size[1] // self.patch_size[1])
        self.foc_pos_embed.data.copy_(focal_pos_embed)
        self.foc_pos_embed.requires_grad = False

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        
        if self.bwpe:
            for proj in self.patch_embed.proj:
                nn.init.xavier_uniform_(proj.weight)
                nn.init.constant_(proj.bias, 0)
        else:
            w = self.patch_embed.proj.weight.data

            if self.trunc_init:
                torch.nn.init.trunc_normal_(w)
                torch.nn.init.trunc_normal_(self.mask_token, std=0.02)
            else:
                torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
                torch.nn.init.normal_(self.mask_token, std=0.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            if self.trunc_init:
                nn.init.trunc_normal_(m.weight, std=0.02)
            else:
                torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)



    def patchify(self, imgs):
        N, _, T, H, W = imgs.shape
        p = self.patch_embed.patch_size[0]
        u = self.patch_embed.patch_size[2]
        assert H == W and H % p == 0 and T % u == 0
        h = w = H // p
        t = T // u

        x = imgs.reshape(shape=(N, 1, t, u, h, p, w, p))
        x = torch.einsum("nctuhpwq->nthwupqc", x)
        x = x.reshape(shape=(N, t * h * w, u * p**2 * 1))
        self.patch_info = (N, T, H, W, p, u, t, h, w)
        return x
    

    def unpatchify(self, x):
        N, T, H, W, p, u, t, h, w = self.patch_info

        x = x.reshape(shape=(N, t, h, w, u, p, p, 1))

        x = torch.einsum("nthwupqc->nctuhpwq", x)
        imgs = x.reshape(shape=(N, 1, T, H, W))
        return imgs
    
    def get_dim_patches(self, T, L, mask_ratio):
        # Handle edge cases where T or L are < 2 (no valid pairs)
        if T < 2 or L < 2:
            len_t = 1 if T < 2 else T
            len_l = 1 if L < 2 else L
            return len_t, len_l

        # Build explicit list of (t, l) pairs
        pairs = [(i, j) for i in range(2, T + 1) for j in range(2, L + 1)]
        len_all = torch.tensor(pairs, dtype=torch.long, device='cpu')

        len_keep = float((1 - mask_ratio) * T * L)

        # Compute product for each candidate and find closest to len_keep
        lens = (len_all[:, 0].float() * len_all[:, 1].float())
        lens_diff = torch.abs(lens - len_keep)

        # indices where difference is minimal
        ind = torch.where(lens_diff == torch.min(lens_diff))[0]
        # choose randomly among ties
        r = torch.randint(0, len(ind), (1,)).item()
        index = len_all[ind[r]]

        len_t = int(index[0].item())
        len_l = int(index[1].item())
        return len_t, len_l



    def spectral_only_masking(self, x, T, L, mask_ratio):
        """
        Mascaramento Espectral (Bandas):
        T = Dimensão Espectral (número de bandas, ex: 4)
        L = Dimensão Espacial (número de patches)
        
        Remove bandas inteiras. Se a Banda 2 for removida, ela some de todos os L patches.
        """
        N, _, D = x.shape

        self.len_t, self.len_l = self.get_dim_patches(T, L, mask_ratio)

        # 1. Calcular quantas BANDAS manter
        # Ex: se T=4 e ratio=0.25, len_keep_t = 3 (mantém 3 bandas)
        len_keep_t = int(T * (1 - mask_ratio))
        
        # O total de tokens a manter é (Bandas mantidas * Total de Patches)
        len_keep = len_keep_t * self.len_l

        # 2. Gerar ruído apenas para as bandas (N, T)
        # Cada banda recebe um "score" de importância aleatório
        noise = torch.rand(N, T, device=x.device)

        # 3. Expandir o ruído para cobrir a dimensão espacial (L)
        # IMPORTANTE: Depende da ordem dos seus dados em 'x'.
        # Se x está ordenado como [B1_P1, B1_P2... B2_P1...], usamos repeat_interleave.
        # Isso cria blocos: [ruido_b1, ruido_b1... ruido_b2, ruido_b2...]
        noise = noise.repeat_interleave(L, dim=1)
        
        # 4. Ordenar (Lógica padrão de MAE)
        # Ao ordenar o ruído expandido, todos os patches da mesma banda 
        # ficarão juntos porque têm o mesmo valor de ruído.
        ids_shuffle = torch.argsort(noise, dim=1) 
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # 5. Selecionar os tokens para manter
        ids_keep = ids_shuffle[:, :len_keep]
        
        # Extrair os dados visíveis
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # 6. Gerar a máscara binária (0 = keep, 1 = remove)
        mask = torch.ones([N, T * L], device=x.device)
        mask[:, :len_keep] = 0
        
        # Desembaralhar a máscara para a ordem original
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore, ids_keep


    def forward(self, x, mask_ratio = 0.0):
        # 1. Gerar embeddings dos patches
        # Saída: (B, T, L, C) onde T=patches espectrais, L=patches espaciais
        x = self.patch_embed(x) 
        N, T, L, C = x.shape

        # 2. Achatar para a sequência de tokens
        # Forma: (B, T*L, C)
        total_patches = T * L
        x = x.reshape(N, total_patches, C)

        # 3. Aplicar mascaramento de tokens
        # x agora tem forma [N, len_keep, C]
        x, mask, ids_restore, ids_keep = self.spectral_only_masking(x, T, L, mask_ratio)

        x = x.view(N, -1, C)

        # 4. Selecionar o embedding posicional correto
        if total_patches == self.rand_pos_embed.shape[1]:
            pos_embed = self.rand_pos_embed.expand(N, -1, -1)
        elif total_patches == self.foc_pos_embed.shape[1]:
            pos_embed = self.foc_pos_embed.expand(N, -1, -1)
        else:
            raise ValueError("O número total de patches não corresponde a 'rand_size' nem 'focal_size'.")
        
        # 5. Adicionar embedding posicional (apenas aos tokens mantidos)
        pos_embed = torch.gather(
            pos_embed,
            dim=1, 
            index=ids_keep.unsqueeze(-1).repeat(1, 1, pos_embed.shape[2])
        )
        x = x + pos_embed


        # rearanjar para calcular as atenções espectrais
        x =  rearrange(x, 'b (t l) c -> (b l) t c', t= self.len_t , l= self.len_l)


        # 6. Aplicar os blocos Transformer
        for blk in self.spec_transformer:
            x = blk(x)

        x = rearrange(x, '(b l) t c -> b (t l) c', b=N, l=self.len_l)

        # 7. Normalização final
        x = self.norm(x)

        # 8. Pooling (média) para obter a representação final
        # Forma: (B, C)
        x = x.mean(dim=1)

        return x


class VisionTransformerVanilla(nn.Module):
    """ Vision Transformer com mascaramento global (sem separação espacial/espectral)
    """
    def __init__(self,rand_size = (60,60,150), focal_size = (30,30,150), patch_size=(5,5,150), in_chans=1, embed_dim=768, depth=12,
                 num_heads=12 ,mlp_ratio=4., qkv_bias=True, drop_rate=0., attn_drop_rate=0., drop_path_rate=0., trunc_init=True, bwpe=False,
                 norm_layer=nn.LayerNorm, **kwargs):
        super().__init__()
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.bwpe = bwpe

        self.rand_size = rand_size
        self.focal_size = focal_size # Adicionado para checagem
        self.patch_size = patch_size

        # the patch size[2] has to be the same as the number of bands in the image, otherwise the patch embedding will not work
        assert rand_size[2] == patch_size[2], \
            f"Number of bands in the image must be equal to the patch size in the spectral dimension. Got image bands {rand_size[2]} and patch bands {patch_size[2]}."


        assert rand_size[0] % patch_size[0] == 0 and rand_size[1] % patch_size[1] == 0 and rand_size[2] % patch_size[2] == 0, \
            f"Image dimensions must be divisible by the patch size. Got image size {rand_size} and patch size {patch_size}."
        
        assert focal_size[0] % patch_size[0] == 0 and focal_size[1] % patch_size[1] == 0 and focal_size[2] % patch_size[2] == 0, \
            f"Image dimensions must be divisible by the patch size. Got image size {focal_size} and patch size {patch_size}."
        
        self.patch_embed = PatchEmbed(patch_size = patch_size,
                                        in_chans = in_chans,
                                        embed_dim = embed_dim)
        
        num_rand_patches = (rand_size[0] // patch_size[0]) * (rand_size[1] // patch_size[1]) 
        num_focal_patches = focal_size[0] // patch_size[0] * (focal_size[1] // patch_size[1])  


        self.rand_pos_embed = nn.Parameter(torch.zeros(1, num_rand_patches , embed_dim))
        self.foc_pos_embed = nn.Parameter(torch.zeros(1, num_focal_patches, embed_dim))


        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.transformer_blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=drop_path_rate,
                norm_layer=norm_layer)
            for i in range(depth)])
        
        self.trunc_init = trunc_init

        self.norm = norm_layer(embed_dim)

        self.initialize_weights()


    def initialize_weights(self):
        # now i will use the 2D positional embedding for both rand and focal, since the patch size is (5,5,150) and we are treating the spectral dimension as a whole
        rand_pos_embed = get_2d_sincos_pos_embed_for_vanilla(self.embed_dim,
                                                 self.rand_size[0] // self.patch_size[0],
                                                 self.rand_size[1] // self.patch_size[1])   
        
        self.rand_pos_embed.data.copy_(torch.from_numpy(rand_pos_embed).float())
        self.rand_pos_embed.requires_grad = False

        focal_pos_embed = get_2d_sincos_pos_embed_for_vanilla(self.embed_dim,
                                                   self.focal_size[0] // self.patch_size[0],
                                                   self.focal_size[1] // self.patch_size[1])
        self.foc_pos_embed.data.copy_(torch.from_numpy(focal_pos_embed).float())
        self.foc_pos_embed.requires_grad = False

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        w = self.patch_embed.proj.weight.data
        if self.trunc_init:
            torch.nn.init.trunc_normal_(w)
            torch.nn.init.trunc_normal_(self.mask_token, std=0.02)
        else:
            torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
            torch.nn.init.normal_(self.mask_token, std=0.02)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            if self.trunc_init:
                nn.init.trunc_normal_(m.weight, std=0.02)
            else:
                torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        #  only 2d patchify, since the patch size is (5,5,150) and we are treating the spectral dimension as a whole
        N, _, T, H, W = imgs.shape
        p = self.patch_embed.patch_size[0]
        assert H == W and H % p == 0
        h = w = H // p  

        x = imgs.reshape(shape=(N, 1, T, h, p, w, p))
        x = torch.einsum("nctuhpwq->nthwupqc", x)
        x = x.reshape(shape=(N, T, h * w, p**2 * 1))
        self.patch_info = (N, T, H, W, p, h, w)
        return x
    
    def unpatchify(self, x):
        N, T, H, W, p, h, w = self.patch_info

        x = x.reshape(shape=(N, T, h, w, p, p, 1))

        x = torch.einsum("nthwupqc->nctuhpwq", x)
        imgs = x.reshape(shape=(N, 1, T, H, W))
        return imgs
    
    def random_masking(self, x, mask_ratio):
        """
        Mascaramento aleatório padrão do ViT/MAE.
        Remove uma fração aleatória de patches sem distinção espacial/espectral.
        
        Args:
            x: tensor de entrada [N, L, D] onde L = número total de patches
            mask_ratio: fração de patches a mascarar (0.0 a 1.0)
        
        Returns:
            x_masked: patches visíveis [N, len_keep, D]
            mask: máscara binária [N, L] (0 = visível, 1 = mascarado)
            ids_restore: índices para restaurar ordem original
            ids_keep: índices dos patches mantidos
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        
        # Gerar ruído aleatório para cada patch
        noise = torch.rand(N, L, device=x.device)
        
        # Ordenar patches por ruído (menor ruído = mantido)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        
        # Manter apenas os primeiros len_keep patches
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        
        # Criar máscara binária (0 = keep, 1 = remove)
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)
        
        return x_masked, mask, ids_restore, ids_keep

    def forward(self, x, mask_ratio=0.0):
        """
        Forward pass do ViT vanilla com mascaramento opcional.
        
        Args:
            x: tensor de entrada [B, C, T, H, W]
            mask_ratio: razão de mascaramento (0.0 = sem mascaramento)
        
        Returns:
            x: representação final [B, embed_dim]
        """
        # 1. Patch embedding
        # Saída: (B, T, L, C) onde T=patches espectrais, L=patches espaciais
        x = self.patch_embed(x)
        N, T, L, C = x.shape
        
        # 2. Achatar para sequência única de tokens (tratamento 2D)
        # Como patch_size[2] = 150 (todas as bandas), temos T=1
        # Então a sequência é simplesmente os patches espaciais
        total_patches = T * L
        x = x.reshape(N, total_patches, C)
        
        # 3. Aplicar mascaramento aleatório (se mask_ratio > 0)
        if mask_ratio > 0:
            x, mask, ids_restore, ids_keep = self.random_masking(x, mask_ratio)
        else:
            # Sem mascaramento, todos os patches são mantidos
            ids_keep = torch.arange(total_patches, device=x.device).unsqueeze(0).repeat(N, 1)
        
        # 4. Selecionar positional embedding correto
        if total_patches == self.rand_pos_embed.shape[1]:
            pos_embed = self.rand_pos_embed.expand(N, -1, -1)
        elif total_patches == self.foc_pos_embed.shape[1]:
            pos_embed = self.foc_pos_embed.expand(N, -1, -1)
        else:
            raise ValueError(
                f"Número total de patches ({total_patches}) não corresponde a "
                f"rand_size ({self.rand_pos_embed.shape[1]}) nem focal_size ({self.foc_pos_embed.shape[1]})."
            )
        
        # 5. Adicionar positional embedding apenas aos patches mantidos
        pos_embed = torch.gather(
            pos_embed,
            dim=1,
            index=ids_keep.unsqueeze(-1).repeat(1, 1, pos_embed.shape[2])
        )
        x = x + pos_embed
        
        # 6. Aplicar blocos Transformer (atenção global)
        for blk in self.transformer_blocks:
            x = blk(x)
        
        # 7. Normalização final
        x = self.norm(x)
        
        # 8. Global average pooling para obter representação final
        # Forma: (B, C)
        x = x.mean(dim=1)
        
        return x

