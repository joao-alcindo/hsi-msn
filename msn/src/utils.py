import math
import torch



# --- Funções de Utilidade (AverageMeter, Schedulers) ---
class AverageMeter(object):
    """Calcula e armazena a média e o valor atual"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = self.avg = self.sum = self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class WarmupCosineSchedule(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_steps, start_lr, ref_lr, final_lr, T_max, last_epoch=-1):
        self.warmup_steps = warmup_steps; self.start_lr = start_lr; self.ref_lr = ref_lr
        self.final_lr = final_lr; self.T_max = T_max; super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            lr = (self.ref_lr - self.start_lr) / self.warmup_steps * self.last_epoch + self.start_lr
        else:
            progress = (self.last_epoch - self.warmup_steps) / (self.T_max - self.warmup_steps)
            lr = self.final_lr + 0.5 * (self.ref_lr - self.final_lr) * (1. + math.cos(math.pi * progress))
        return [lr for _ in self.optimizer.param_groups]

class CosineWDSchedule(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, ref_wd, final_wd, T_max, last_epoch=-1):
        self.ref_wd = ref_wd; self.final_wd = final_wd; self.T_max = T_max; super().__init__(optimizer, last_epoch)

    def get_lr(self):
        progress = self.last_epoch / self.T_max
        wd = self.final_wd + 0.5 * (self.ref_wd - self.final_wd) * (1. + math.cos(math.pi * progress))
        wds_for_groups = []
        for param_group in self.optimizer.param_groups:
            if 'WD_exclude' in param_group and param_group['WD_exclude']:
                wds_for_groups.append(0.0)
            else:
                wds_for_groups.append(wd)
        return wds_for_groups
