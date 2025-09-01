# src/model.py

import torch
import torch.nn as nn
import copy

from src.vit_hsi import VisionTransformerHSI



class MSNModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config



        self.student_encoder = VisionTransformerHSI(rand_size= config.rand_size,
                                                    focal_size= config.focal_size,
                                                    patch_size= config.patch_size, 
                                                    in_chans= config.in_chans, 
                                                    embed_dim= config.embed_dim, 
                                                    depth= config.depth,
                                                    num_heads= config.num_heads,
                                                    mlp_ratio= config.mlp_ratio, 
                                                    qkv_bias=True, 
                                                    drop_rate= config.drop_rate, 
                                                    attn_drop_rate= config.attn_drop_rate, 
                                                    drop_path_rate= config.drop_path_rate, 
                                                    trunc_init= config.trunc_init,
                                                    norm_layer=nn.LayerNorm)
        
        #-- target encoder 
        self.target_encoder = copy.deepcopy(self.student_encoder)

        self.mask_ratio = config.mask_ratio

        # Congela os parâmetros do alvo para que não sejam atualizados por backpropagation
        for param in self.target_encoder.parameters():
            param.requires_grad = False

        # --- Protótipos ---
        # Os protótipos são uma matriz de pesos treináveis (Embedding, Num_prototipos)
        self.prototypes = nn.Parameter(torch.randn(self.config.num_prototipos, self.config.embed_dim))

        # inicializa os pesos dos protótipos
        self.initialize_weights_prototypes()

    def initialize_weights_prototypes(self):
        nn.init.normal_(self.prototypes, mean=0.0, std=0.02)
    

    def forward(self, rand_views, focal_views, target_view):
        
        # --- Visões do Estudante ---
        anchor_views = []



        # rand_views: lista de tensores (B, C, T, H, W)
        for view in rand_views:
            anchor_view = self.student_encoder(view, mask_ratio = self.mask_ratio)  # (B, Embedding)
            anchor_views.append(anchor_view)

        
        # focal_views: lista de tensores (B, C, T, H, W)
        for view in focal_views:
            anchor_view = self.student_encoder(view, mask_ratio = self.mask_ratio)  # (B, Embedding)
            anchor_views.append(anchor_view)




        anchor_views = torch.cat(anchor_views, dim=0)  # (B * num_anchor_views, Embedding)

        # --- Visão do Alvo ---
        with torch.no_grad():
            target_view = self.target_encoder(target_view, mask_ratio = 0.0)  # (B, Embedding)


        return anchor_views, target_view, self.prototypes
       


    def update_target_networks(self):
        """
        Atualiza os pesos do 'target_encoder' usando Média Móvel Exponencial (EMA)
        dos pesos do 'student_encoder'.
        """
        for param_student, param_target in zip(self.student_encoder.parameters(), self.target_encoder.parameters()):
            param_target.data = self.config.alpha_ema * param_target.data + (1 - self.config.alpha_ema) * param_student.data