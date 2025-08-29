
import torch
import torch.nn as nn
import torch.nn.functional as F


from itertools import product
import random

from pos_embed import get_3d_sincos_pos_embed



from timm.models.layers import trunc_normal_, DropPath, Mlp
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
        B, C, T, H, W = x.shape


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
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

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

        assert rand_size[0] % patch_size[0] == 0 and rand_size[1] % patch_size[1] == 0 and rand_size[2] % patch_size[2] == 0, \
            f"Image dimensions must be divisible by the patch size. Got image size {rand_size} and patch size {patch_size}."
        assert focal_size[0] % patch_size[0] == 0 and focal_size[1] % patch_size[1] == 0 and focal_size[2] % patch_size[2] == 0, \
            f"Image dimensions must be divisible by the patch size. Got image size {focal_size} and patch size {patch_size}."


        self.patch_embed = PatchEmbed(patch_size = patch_size, 
                                      in_chans   = in_chans, 
                                      embed_dim  = embed_dim)
        
        num_rand_patches = rand_size[0] // patch_size[0] * rand_size[1] // patch_size[1] * rand_size[2] // patch_size[2]
        num_focal_patches = focal_size[0] // patch_size[0] * focal_size[1] // patch_size[1] * focal_size[2] // patch_size[2]

        self.rand_pos_embed = nn.Parameter(torch.zeros(1, num_rand_patches, embed_dim))
        self.foc_pos_embed = nn.Parameter(torch.zeros(1, num_focal_patches, embed_dim))

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))


        self.vit_spatial = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=drop_path_rate,
                norm_layer=norm_layer)
            for i in range(depth)])
        
        self.vit_spectral = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=drop_path_rate,
                norm_layer=norm_layer)
            for i in range(depth)])
        
        self.norm = norm_layer(embed_dim)

        self.trunc_init = trunc_init

        
        
        self.initialize_weights()


    def initialize_weights(self):
        # rand positional embedding
        rand_pos_embed = get_3d_sincos_pos_embed(self.embed_dim, self.rand_size[0], self.focal_size[1], cls_token=True)
        self.rand_pos_embed.data.copy_(rand_pos_embed)
        self.rand_pos_embed.requires_grad = False

        # focal positional embedding
        focal_pos_embed = get_3d_sincos_pos_embed(self.embed_dim, self.focal_size[0], self.focal_size[1], cls_token=True)
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
    

    def forward(self, x, mask_ratio = 0.6):
        #embed patches
        x = self.patch_embed(x)  # [B, num_patches, embed_dim]
        N, T, L, C = x.shape  # T: num_patches_t, L: num_patches_l

        x = x.reshape(N, T * L, C)


        x, mask, ids_restore, ids_keep = self.spatial_spectral_masking(x, T, L, 
                                                                       mask_ratio= mask_ratio)
        x = x.view(N, -1, C)


        if x.shape[1] == self.rand_pos_embed.shape[1]:

            pos_embed = self.rand_pos_embed[:, 1:, :].expand(N, -1, -1)
            pos_embed_cls = self.rand_pos_embed[:, :1, :].expand(N, -1, -1)
        else:
            pos_embed = self.foc_pos_embed[:, 1:, :].expand(N, -1, -1)  
            pos_embed_cls = self.foc_pos_embed[:, :1, :].expand(N, -1, -1)



        # add cls token
        cls_tokens = self.cls_token.expand(N, -1, -1)
        cls_tokens = cls_tokens + pos_embed_cls

        # add position embedding

        pos_embed = torch.gather(
            pos_embed, # Começa do segundo elemento (ignora o do cls_token)
            dim=1, 
            index=ids_keep.unsqueeze(-1).repeat(1, 1, pos_embed.shape[2])
        )
        x = x.view([N, -1, C] ) + pos_embed

        # concat cls token
        x = torch.cat((cls_tokens, x), dim=1)

        assert x.shape[1] == self.len_t * self.len_l + 1

        x_spatial = rearrange(x, 'b (t l) c -> (b t) l c', t=self.len_t, l=self.len_l)
        x_spectral = rearrange(x, 'b (t l) c -> (b l) t c', t=self.len_t, l=self.len_l)

        for blk in self.vit_spatial:
            x_spatial = blk(x_spatial)
        
        for blk in self.vit_spectral:
            x_spectral = blk(x_spectral)

        x_spatial = rearrange(x_spatial, '(b t) l c -> b (t l) c', b=N, t=self.len_t, l=self.len_l)
        x_spectral = rearrange(x_spectral, '(b l) t c -> b (t l) c', b=N, t=self.len_t, l=self.len_l)

        x = (x_spatial + x_spectral) 

        x = self.norm(x)

        return x[:, 0, :]  # return cls token representation