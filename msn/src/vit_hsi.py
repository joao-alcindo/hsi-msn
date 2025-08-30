
import torch
import torch.nn as nn
import torch.nn.functional as F


from itertools import product
import random

from pos_embed import get_3d_sincos_pos_embed


# buscar as funcoes
from einops import rearrange

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
        self.mlp = self.mlp = SwiGLU(dim, mlp_hidden_dim, mlp_ratio, drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x



class VisionTransformerHSI(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self,rand_size = (60,60,150) , focal_size = (30,30,150),patch_size=(5,5,10), in_chans=1, embed_dim=768, depth=12,
                 num_heads=12 ,mlp_ratio=4., qkv_bias=True, drop_rate=0., attn_drop_rate=0., drop_path_rate=0., trunc_init=True,
                 norm_layer=nn.LayerNorm, **kwargs):
        super().__init__()
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads

        self.rand_size = rand_size
        self.focal_size = focal_size
        self.patch_size = patch_size

        assert rand_size[0] % patch_size[0] == 0 and rand_size[1] % patch_size[1] == 0 and rand_size[2] % patch_size[2] == 0, \
            f"Image dimensions must be divisible by the patch size. Got image size {rand_size} and patch size {patch_size}."
        assert focal_size[0] % patch_size[0] == 0 and focal_size[1] % patch_size[1] == 0 and focal_size[2] % patch_size[2] == 0, \
            f"Image dimensions must be divisible by the patch size. Got image size {focal_size} and patch size {patch_size}."


        self.patch_embed = PatchEmbed(patch_size = patch_size, 
                                      in_chans   = in_chans, 
                                      embed_dim  = embed_dim)
        
        num_rand_patches = (rand_size[0] // patch_size[0]) * (rand_size[1] // patch_size[1]) * (rand_size[2] // patch_size[2])
        num_focal_patches = focal_size[0] // patch_size[0] * focal_size[1] // patch_size[1] * focal_size[2] // patch_size[2]

        self.rand_pos_embed = nn.Parameter(torch.zeros(1, num_rand_patches , embed_dim))
        self.foc_pos_embed = nn.Parameter(torch.zeros(1, num_focal_patches, embed_dim))

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))


        self.vit_spatial = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=drop_path_rate,
                norm_layer=norm_layer)
            for i in range(depth)])
        
        print(self.vit_spatial)
        
        self.vit_spectral = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=drop_path_rate,
                norm_layer=norm_layer)
            for i in range(depth)])

        self.trunc_init = trunc_init

        self.norm = norm_layer(embed_dim)

        
        
        self.initialize_weights()


    def initialize_weights(self):
        # rand positional embedding
        rand_pos_embed = get_3d_sincos_pos_embed(self.embed_dim,
                                                 self.rand_size[2] // self.patch_size[2],
                                                 self.rand_size[0] // self.patch_size[0], 
                                                 self.rand_size[1] // self.patch_size[1])
        print("rand_pos_embed", rand_pos_embed.shape)


        self.rand_pos_embed.data.copy_(rand_pos_embed)
        self.rand_pos_embed.requires_grad = False

        # focal positional embedding
        focal_pos_embed = get_3d_sincos_pos_embed(self.embed_dim,
                                                   self.focal_size[2] // self.patch_size[2],
                                                   self.focal_size[0] // self.patch_size[0],
                                                   self.focal_size[1] // self.patch_size[1])
        self.foc_pos_embed.data.copy_(focal_pos_embed)
        self.foc_pos_embed.requires_grad = False

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
        len_all = torch.tensor(list(product(range(2, T + 1), range(2, L + 1))))
        len_keep = (1 - mask_ratio) * T * L
        lens = len_all[:, 0] * len_all[:, 1]
        lens_diff = abs(len_keep - lens)
        ind = torch.where(lens_diff == torch.min(lens_diff))[0]
        r = torch.LongTensor(random.sample(range(len(ind)), 1))[0]
        index = len_all[ind[r]]
        len_t, len_l = index
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
        device = x.device
        N, T, L, C = x.shape  # T: num_patches_t, L: num_patches_l


        x = x.reshape(N, T * L, C)



        
        x, mask, ids_restore, ids_keep = self.spatial_spectral_masking(x, T, L, 
                                                                       mask_ratio= mask_ratio)

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

        x = x_spatial + x_spectral

        x = self.norm(x)

        x = x.mean(dim=1)

        #print(x)

        return x