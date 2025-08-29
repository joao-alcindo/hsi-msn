# train.py (versão com log de protótipos usados)

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
import math
import csv

# Adiciona o diretório 'src' ao PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

# IMPORTAÇÕES CORRIGIDAS
# A classe do dataset agora é a que você definiu em src/dataset.py
from src.hsi_msn import MSNModel
from src.loss import msn_loss
from src.dataset import init_data  # Importa a função de inicialização
from src.utils import AverageMeter, WarmupCosineSchedule, CosineWDSchedule

# --- Configurações Globais ---
_GLOBAL_SEED = 0; np.random.seed(_GLOBAL_SEED); torch.manual_seed(_GLOBAL_SEED)
torch.backends.cudnn.benchmark = True
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(asctime)s - %(message)s')
logger = getLogger(__name__)

# --- Função Principal de Treinamento ---
def main(config_path):

    # --- 1. Configuração Inicial ---
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    config = SimpleNamespace(**config_dict)

    # Verifica se os tamanhos das views e dos patches são válidos
    assert all(d % p == 0 for d, p in zip(config.rand_size, config.patch_size)), "rand_size must be divisible by patch_size"
    assert all(d % p == 0 for d, p in zip(config.focal_size, config.patch_size)), "focal_size must be divisible by patch_size"

    run_name = f"proto{config.num_prototipos}_embed{config.tamanho_embedding}"
    output_dir = os.path.join(config.output_dir, run_name)
    
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Resultados serão salvos em: {output_dir}")
    
    config_output_path = os.path.join(output_dir, 'config.yaml')
    # add outiput_dir to config_dict
    config_dict['output_dir'] = output_dir
    with open(config_output_path, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False)
    logger.info(f"Configuração salva em: {config_output_path}")

    log_file_path = os.path.join(output_dir, 'training_log.csv')
    log_header = ['epoch', 'avg_total_loss', 'avg_ce_loss', 'avg_memax_reg', 'lr', 'wd', 'momentum_ema', 'avg_prototypes_used']
    if not os.path.isfile(log_file_path):
        with open(log_file_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(log_header)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Usando dispositivo: {device}")

    model = MSNModel(config); model.to(device)

    # --- 2. Otimizador e Carregador de Dados ---
    param_groups = [
        {'params': [p for n, p in model.named_parameters() if p.requires_grad and 'prototypes' not in n and ('bias' not in n) and (len(p.shape) != 1)], 'weight_decay': config.weight_decay},
        {'params': [p for n, p in model.named_parameters() if p.requires_grad and 'prototypes' not in n and (('bias' in n) or (len(p.shape) == 1))], 'weight_decay': 0},
        {'params': [model.prototypes], 'lr': config.learning_rate, 'weight_decay': 0}
    ]
    optimizer = optim.AdamW(param_groups, lr=config.learning_rate)

    # CARREGAMENTO DE DADOS CORRIGIDO
    # Usa a função init_data para obter o DataLoader
    train_loader = init_data(config_dict)
    
    # --- 3. Schedulers e Variáveis de Estado ---
    steps_per_epoch = len(train_loader)
    total_training_steps = config.num_epochs * steps_per_epoch
    scheduler_lr = WarmupCosineSchedule(optimizer, warmup_steps=int(config.warmup_epochs * steps_per_epoch), start_lr=config.learning_rate * 0.1, ref_lr=config.learning_rate, final_lr=config.final_lr, T_max=total_training_steps)
    scheduler_wd = CosineWDSchedule(optimizer, ref_wd=config.weight_decay, final_wd=config.final_weight_decay, T_max=total_training_steps)

    start_epoch = 0; global_step = 0; best_loss = float('inf')

    # --- 4. Lógica para Continuar Treinamento ---
    resume_path = os.path.join(output_dir, 'last_checkpoint.pth') if config.resume_from == 'last' else config.resume_from
    if resume_path and os.path.isfile(resume_path):
        logger.info(f"Continuando do checkpoint: {resume_path}")
        checkpoint = torch.load(resume_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler_lr.load_state_dict(checkpoint['scheduler_lr_state_dict'])
        scheduler_wd.load_state_dict(checkpoint['scheduler_wd_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        global_step = checkpoint['global_step']
        best_loss = checkpoint.get('best_loss', float('inf'))
        logger.info(f"Modelo carregado. Começando da época {start_epoch}. Melhor perda até agora: {best_loss:.4f}")
    else:
        logger.info("Iniciando treinamento do zero.")

    # --- 5. Scheduler de Momentum e Medidores ---
    momentum_start = config.alpha_ema
    momentum_final = 0.2
    momentum_increment_per_step = (momentum_final - momentum_start) / total_training_steps
    momentum_scheduler_iter = (momentum_start + (momentum_increment_per_step * i) for i in range(global_step, total_training_steps + 1))

    loss_meter = AverageMeter(); ce_loss_meter = AverageMeter(); memax_reg_meter = AverageMeter()
    proto_used_meter = AverageMeter()

    # --- 6. Loop Principal de Treinamento ---
    logger.info("Iniciando o ciclo de treinamento...")
    for epoch in range(start_epoch, config.num_epochs):
        loss_meter.reset(); ce_loss_meter.reset(); memax_reg_meter.reset()
        proto_used_meter.reset()
        model.train()

        for batch_idx, views in enumerate(train_loader):
            # O dataloader retorna uma lista de tensores: [target_view, focal_view_1, ..., focal_view_N, rand_view_1, ..., rand_view_M]
            views = [v.to(device) for v in views]

            target_view = views[0]
            focal_views_list = views[1:1 + config.rand_views]
            rand_views_list = views[1 + config.rand_views:]
            
            batch_size = target_view.size(0)
            
            optimizer.zero_grad()

            # Passa as listas de views separadamente para o modelo
            anchor_representations, target_representation, prototypes_weights = model(
                focal_views_list=focal_views_list,
                rand_views_list=rand_views_list,
                target_view=target_view
            )

            total_loss, cross_entropy_loss, me_max_reg, _, log_dct = msn_loss(
                anchor_representations=anchor_representations,
                target_representation=target_representation,
                prototypes=prototypes_weights,
                num_anchor_views= config.rand_views + config.focal_views,
                use_sinkhorn=config.use_sinkhorn,
                temp_anchor=config.temp_anchor,
                temp_target=config.temp_target,
                lambda_reg=config.lambda_reg
            )
            total_loss.backward()
            current_lrs, current_wds = scheduler_lr.get_last_lr(), scheduler_wd.get_last_lr()
            for i, param_group in enumerate(optimizer.param_groups):
                param_group['lr'] = current_lrs[i]
                if 'weight_decay' in param_group and param_group['weight_decay'] != 0:
                    param_group['weight_decay'] = current_wds[i]
            optimizer.step()
            scheduler_lr.step(); scheduler_wd.step()
            current_momentum = next(momentum_scheduler_iter)
            model.config.alpha_ema = current_momentum
            model.update_target_networks()
            global_step += 1
            loss_meter.update(total_loss.item(), n=batch_size)
            ce_loss_meter.update(cross_entropy_loss.item(), n=batch_size)
            memax_reg_meter.update(me_max_reg.item(), n=batch_size)
            proto_used_meter.update(log_dct['np'])

        # --- 7. Tarefas de Final de Época ---
        current_lr = optimizer.param_groups[0]['lr']
        current_wd = optimizer.param_groups[0]['weight_decay']
        
        logger.info(
            f"Epoch [{epoch+1}/{config.num_epochs}] - "
            f"Avg Total Loss: {loss_meter.avg:.4f}, "
            f"Avg CE Loss: {ce_loss_meter.avg:.4f}, "
            f"Avg ME-MAX Reg: {memax_reg_meter.avg:.4f}, "
            f"LR: {current_lr:.2e}, "
            f"WD: {current_wd:.2e}, "
            f"Momentum EMA: {current_momentum:.4f}, "
            f"Prototypes Used: {proto_used_meter.avg:.1f}/{config.num_prototipos}"
        )
        
        # stop training prototypes after config.epoch_stop_prototype
        if epoch >= config.epoch_stop_prototype:    
            model.prototypes.requires_grad = False

        with open(log_file_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch + 1, loss_meter.avg, ce_loss_meter.avg, memax_reg_meter.avg, current_lr, current_wd, current_momentum, proto_used_meter.avg])

        if loss_meter.avg < best_loss:
            best_loss = loss_meter.avg
            logger.info(f"Nova perda mínima encontrada! Salvando 'min_loss_checkpoint.pth'...")
            min_loss_ckpt_path = os.path.join(output_dir, 'min_loss_checkpoint.pth')
            torch.save({'epoch': epoch, 'global_step': global_step, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'scheduler_lr_state_dict': scheduler_lr.state_dict(), 'scheduler_wd_state_dict': scheduler_wd.state_dict(), 'best_loss': best_loss}, min_loss_ckpt_path)

        if (epoch + 1) % config.save_freq_epochs == 0 or (epoch + 1) == config.num_epochs:
            checkpoint_path = os.path.join(output_dir, f'ckpt_epoch_{epoch+1}.pth')
            logger.info(f"Salvando checkpoint periódico em {checkpoint_path}...")
            torch.save({'epoch': epoch, 'global_step': global_step, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'scheduler_lr_state_dict': scheduler_lr.state_dict(), 'scheduler_wd_state_dict': scheduler_wd.state_dict(), 'best_loss': best_loss}, checkpoint_path)

    logger.info("\nTreinamento concluído!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Treinamento de modelo MSN.")
    parser.add_argument("--config", type=str, default="config.yaml", help="Caminho para o arquivo de configuração YAML.")
    args = parser.parse_args()
    main(args.config)