# src/loss.py

import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Algoritmo Sinkhorn-Knopp ---
@torch.no_grad()
def sinkhorn(logits: torch.Tensor, epsilon: float = 0.05, num_itr: int = 3):
    """
    Aplica o algoritmo Sinkhorn-Knopp para obter uma matriz de atribuição balanceada.

    Args:
        logits (torch.Tensor): A saída bruta do modelo (scores), com shape (B, K).
        epsilon (float): Parâmetro de temperatura para suavizar a distribuição.
                         Valores menores resultam em uma distribuição mais "sharp" (concentrada).
        num_itr (int): Número de iterações do algoritmo de normalização.

    Returns:
        torch.Tensor: A matriz de atribuição normalizada Q, com shape (B, K).
    """
    # Passo 1: Suavizar os logits usando a temperatura epsilon e transpor
    # Q se torna uma matriz de "códigos" com shape (K, B)
    Q = torch.exp(logits / epsilon).T
    B = Q.shape[1]  # Tamanho do batch (número de amostras)
    K = Q.shape[0]  # Número de protótipos (ou "códigos")

    # Passo 2: Normalização inicial para que a soma da matriz seja 1
    sum_Q = torch.sum(Q)
    Q /= sum_Q

    # Passo 3: Iterações de normalização de Sinkhorn-Knopp
    for _ in range(num_itr):
        # Normalização das linhas: a soma de cada linha (protótipo) deve ser 1/K
        # Isso garante que cada protótipo seja usado com a mesma frequência no batch.
        Q /= torch.sum(Q, dim=1, keepdim=True)
        Q /= K

        # Normalização das colunas: a soma de cada coluna (amostra) deve ser 1/B
        # Isso garante que cada amostra seja distribuída igualmente entre os protótipos.
        Q /= torch.sum(Q, dim=0, keepdim=True)
        Q /= B

    # Passo 4: Ajuste final
    # As colunas devem somar 1 para que Q seja uma matriz de atribuição válida.
    Q *= B
    
    # Retorna a matriz transposta para o shape original (B, K)
    return Q.T



# --- Função Principal de Perda MSN ---
def msn_loss(
    anchor_representations: torch.Tensor,  # Representações do estudante (Batch*Num_views, Embedding)
    target_representation: torch.Tensor,   # Representações do alvo (Batch, Embedding)
    prototypes: torch.nn.Parameter,        # Matriz de protótipos (Embedding, Num_prototipos)
    num_anchor_views: int,                 # Número de visões estudante
    use_sinkhorn: bool = False,            # Ativar/Desativar Sinkhorn-Knopp para alvos
    temp_anchor: float = 0.1,              # Temperatura para softmax do estudante
    temp_target: float = 0.025,            # Temperatura para softmax/Sinkhorn do alvo
    lambda_reg: float = 1.0,               # Peso para o termo de regularização ME-MAX
):
    """
    Calcula a perda da MSN.
    Retorna a perda total e as perdas componentes para logging.
    """
    # 1. Normalização L2 das representações e protótipos
    anchor_representations = F.normalize(anchor_representations, dim=1)
    target_representation = F.normalize(target_representation, dim=1)
    prototypes = F.normalize(prototypes, dim=0)

    # 2. Calcular predições do alvo (pseudo-labels) - SEM GRADIENTES
    with torch.no_grad():
        target_logits = (target_representation @ prototypes) / temp_target

        if use_sinkhorn:
            # Aplica o algoritmo Sinkhorn-Knopp'
            target_p = sinkhorn(target_logits)
        else:
            # Usa softmax padrão
            target_p = F.softmax(target_logits, dim=1)

        # Repete os alvos para corresponder ao número de visões âncora (estudante)
        target_p_repeated = target_p.repeat_interleave(num_anchor_views, dim=0)

    # 3. Calcular predições do estudante (âncora)
    anchor_logits = (anchor_representations @ prototypes) / temp_anchor

    # 4. Calcular a perda de entropia cruzada principal
    cross_entropy_loss = - (target_p_repeated * F.log_softmax(anchor_logits, dim=1)).sum(dim=1)
    cross_entropy_loss = cross_entropy_loss.mean()  # Média sobre o batch

    # 5. Calcular o regularizador ME-MAX
    # Promove o uso balanceado de todos os protótipos
    avg_anchor_p = F.softmax(anchor_logits, dim=1).mean(dim=0)
    me_max_reg = - (avg_anchor_p * torch.log(avg_anchor_p + 1e-8)).sum()

    # sloss não é utilizada nesta versão simplificada
    sloss = torch.tensor(0.0, device=cross_entropy_loss.device)

    # Métricas para logging (sem rastreamento de gradientes)
    with torch.no_grad():
        num_ps = float(len(set(target_p_repeated.argmax(dim=1).tolist())))
        max_t = target_p_repeated.max(dim=1).values.mean()
        min_t = target_p_repeated.min(dim=1).values.mean()
        log_dct = {'np': num_ps, 'max_t': max_t, 'min_t': min_t}

    # Retorna a perda combinada e seus componentes individuais
    return cross_entropy_loss - lambda_reg * me_max_reg, cross_entropy_loss, me_max_reg, sloss, log_dct