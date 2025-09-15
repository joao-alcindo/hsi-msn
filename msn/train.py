# train.py (Versão Simplificada)

import torch
import torch.optim as optim
import os
import sys
import yaml
import argparse
from types import SimpleNamespace
from logging import getLogger
import logging
import numpy as np
import csv
import math
import tqdm

# Adiciona o diretório 'src' ao PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

# IMPORTAÇÕES
from src.hsi_msn import MSNModel
from src.loss import msn_loss
from src.dataset import init_data
from src.utils import AverageMeter, WarmupCosineSchedule, CosineWDSchedule


_GLOBAL_SEED = 42

# Configurações Globais (movidas para main para maior controle)
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(asctime)s - %(message)s')
logger = getLogger(__name__)

# --- Função para treinar uma única época ---
def train_epoch(model, train_loader, optimizer, scheduler_lr, scheduler_wd, momentum_scheduler, device, epoch, config, loggers):
    
    # Reseta os medidores de média
    loss_meter, ce_loss_meter, memax_reg_meter, proto_used_meter = loggers
    loss_meter.reset(); ce_loss_meter.reset(); memax_reg_meter.reset(); proto_used_meter.reset()

    model.train()
    total_steps_in_epoch = len(train_loader)
    
    for batch_idx, views in enumerate(train_loader):
    #for batch_idx, views in enumerate(tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.num_epochs}")):
        # Move dados para o dispositivo
        views = [v.to(device) for v in views]
        
        # A primeira vista é a 'target', o restante são as 'anchor'
        target_view = views[0]
        rand_views_list = views[1:1 + config.rand_views] # Views aleatórias (exceto a target)
        focal_views_list = views[1 + config.rand_views:] # Views focais
        
        batch_size = target_view.size(0)
        
        # Otimizador
        optimizer.zero_grad()
        
        # Forward pass
        anchor_representations, target_representation, prototypes_weights = model(
            rand_views=rand_views_list,
            focal_views=focal_views_list,
            target_view=target_view
        )

        # print(anchor_representations)

        # Cálculo da perda
        total_loss, cross_entropy_loss, me_max_reg, _, log_dct = msn_loss(
            anchor_representations=anchor_representations,
            target_representation=target_representation,
            prototypes=prototypes_weights,
            num_anchor_views=config.rand_views + config.focal_views,
            use_sinkhorn=config.use_sinkhorn,
            temp_anchor=config.temp_anchor,
            temp_target=config.temp_target,
            lambda_reg=config.lambda_reg
        )
        
        # Backward pass e otimização
        total_loss.backward()
        optimizer.step()
        
        # Atualiza schedulers
        scheduler_lr.step();scheduler_wd.step()

        new_lr = scheduler_lr.get_last_lr()[0]
        new_wd = optimizer.param_groups[0]['weight_decay']
        
        # Atualiza o momentum do EMA
        model.config.alpha_ema = next(momentum_scheduler)
        model.update_target_networks()
        
        # Atualiza métricas de log
        loss_meter.update(total_loss.item(), n=batch_size)
        ce_loss_meter.update(cross_entropy_loss.item(), n=batch_size)
        memax_reg_meter.update(me_max_reg.item(), n=batch_size)
        proto_used_meter.update(log_dct['np'])

        #tqdm.tqdm.write(f"Batch {batch_idx+1}/{total_steps_in_epoch}, CE loss: {ce_loss_meter.avg:.8f}, ME-MAX: {memax_reg_meter.avg:.8f}, Prototypes used: {proto_used_meter.avg:.8f}")
    return loss_meter.avg, ce_loss_meter.avg, memax_reg_meter.avg, proto_used_meter.avg, new_lr, new_wd

# --- Função Principal de Treinamento ---
def main(config_path):
    # 1. Configuração e setup
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    config = SimpleNamespace(**config_dict)

    np.random.seed(_GLOBAL_SEED); torch.manual_seed(_GLOBAL_SEED)
    torch.backends.cudnn.benchmark = True

    # Verifica os tamanhos
    assert all(d % p == 0 for d, p in zip(config.rand_size, config.patch_size)), "rand_size must be divisible by patch_size"
    assert all(d % p == 0 for d, p in zip(config.focal_size, config.patch_size)), "focal_size must be divisible by patch_size"

    run_name = f"proto{config.num_prototipos}_embed{config.embed_dim}"
    output_dir = os.path.join(config.output_dir, run_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # Salva a configuração atual
    config_output_path = os.path.join(output_dir, 'config.yaml')
    config_dict['output_dir'] = output_dir
    with open(config_output_path, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False)
    logger.info(f"Configuração salva em: {config_output_path}")

    # Inicializa logging em arquivo CSV
    log_file_path = os.path.join(output_dir, 'training_log.csv')
    log_header = ['epoch', 'avg_total_loss', 'avg_ce_loss', 'avg_memax_reg', 'lr', 'wd', 'momentum_ema', 'avg_prototypes_used']
    if not os.path.isfile(log_file_path):
        with open(log_file_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(log_header)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Usando dispositivo: {device}")

    # 2. Inicializa o modelo, otimizador e schedulers
    model = MSNModel(config); model.to(device)

    param_groups = [
        {'params': [p for n, p in model.named_parameters() if p.requires_grad and 'prototypes' not in n and ('bias' not in n) and (len(p.shape) != 1)], 'weight_decay': config.weight_decay},
        {'params': [p for n, p in model.named_parameters() if p.requires_grad and 'prototypes' not in n and (('bias' in n) or (len(p.shape) == 1))], 'weight_decay': 0},
        {'params': [model.prototypes], 'lr': config.learning_rate, 'weight_decay': 0}
    ]
    optimizer = optim.AdamW(param_groups, lr=config.learning_rate)
    train_loader = init_data(config_dict)
    
    steps_per_epoch = len(train_loader)
    total_training_steps = config.num_epochs * steps_per_epoch
    scheduler_lr = WarmupCosineSchedule(optimizer, warmup_steps=int(config.warmup_epochs * steps_per_epoch), start_lr=config.learning_rate * 0.1, ref_lr=config.learning_rate, final_lr=config.final_lr, T_max=total_training_steps)
    scheduler_wd = CosineWDSchedule(optimizer, ref_wd=config.weight_decay, final_wd=config.final_weight_decay, T_max=total_training_steps)
    momentum_start = config.alpha_ema
    momentum_final = 0.2
    momentum_increment_per_step = (momentum_final - momentum_start) / total_training_steps
    momentum_scheduler_iter = (momentum_start + (momentum_increment_per_step * i) for i in range(total_training_steps + 1))

    start_epoch = 0; best_loss = float('inf')

    # 3. Lógica para retomar o treinamento
    resume_path = os.path.join(output_dir, 'last_checkpoint.pth') if config.resume_from == 'last' else config.resume_from
    if resume_path and os.path.isfile(resume_path):
        logger.info(f"Continuando do checkpoint: {resume_path}")
        checkpoint = torch.load(resume_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler_lr.load_state_dict(checkpoint['scheduler_lr_state_dict'])
        scheduler_wd.load_state_dict(checkpoint['scheduler_wd_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_loss = checkpoint.get('best_loss', float('inf'))
        logger.info(f"Modelo carregado. Começando da época {start_epoch}. Melhor perda até agora: {best_loss:.4f}")
    else:
        logger.info("Iniciando treinamento do zero.")

    # 4. Loop Principal de Treinamento
    logger.info("Iniciando o ciclo de treinamento...")
    loggers = (AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter())
    for epoch in range(start_epoch, config.num_epochs):
        avg_total_loss, avg_ce_loss, avg_memax_reg, avg_proto_used, new_lr, new_wd = train_epoch(model, train_loader, optimizer, scheduler_lr, scheduler_wd, momentum_scheduler_iter, device, epoch, config, loggers)

        # Tarefas de final de época
        current_momentum = model.config.alpha_ema

        logger.info(
            f"Epoch [{epoch+1}/{config.num_epochs}] - "
            f"Avg Total Loss: {avg_total_loss:.4f}, "
            f"Avg CE Loss: {avg_ce_loss:.4f}, "
            f"Avg ME-MAX Reg: {avg_memax_reg:.4f}, "
            f"LR: {new_lr:.2e}, "
            f"WD: {new_wd:.2e}, "
            f"Momentum EMA: {current_momentum:.4f}, "
            f"Prototypes Used: {avg_proto_used:.1f}/{config.num_prototipos}"
        )
        
        # Para o treinamento dos protótipos
        if epoch >= config.epoch_stop_prototype:
            model.prototypes.requires_grad = False
        
        # Salva log e checkpoints
        with open(log_file_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch + 1, avg_total_loss, avg_ce_loss, avg_memax_reg, new_lr, new_wd, current_momentum, avg_proto_used])

        state = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_lr_state_dict': scheduler_lr.state_dict(),
            'scheduler_wd_state_dict': scheduler_wd.state_dict(),
            'best_loss': best_loss,
        }
        
        if avg_total_loss < best_loss:
            best_loss = avg_total_loss
            logger.info(f"Nova perda mínima encontrada! Salvando 'min_loss_checkpoint.pth'...")
            min_loss_ckpt_path = os.path.join(output_dir, 'min_loss_checkpoint.pth')
            torch.save(state, min_loss_ckpt_path)

        if (epoch + 1) % config.save_freq_epochs == 0 or (epoch + 1) == config.num_epochs:
            checkpoint_path = os.path.join(output_dir, f'ckpt_epoch_{epoch+1}.pth')
            logger.info(f"Salvando checkpoint periódico em {checkpoint_path}...")
            torch.save(state, checkpoint_path)

    logger.info("\nTreinamento concluído!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Treinamento de modelo MSN.")
    parser.add_argument("--config", type=str, default="config.yaml", help="Caminho para o arquivo de configuração YAML.")
    args = parser.parse_args()
    main(args.config)