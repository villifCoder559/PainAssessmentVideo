import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, OneCycleLR, LambdaLR, CosineAnnealingWarmRestarts, ReduceLROnPlateau
import math

class CustomWeightDecayScheduler:
  """
  Standard PyTorch schedulers only update 'lr'. 
  This custom class updates 'weight_decay' following a Cosine schedule.
  """
  def __init__(self, optimizer, min_wd, total_steps):
    self.optimizer = optimizer
    self.min_wd = min_wd
    self.total_steps = total_steps
    self.current_step = 0
    
    # Capture initial WD from optimizer
    self.initial_wds = [group['weight_decay'] for group in optimizer.param_groups]

  def step(self):
    self.current_step += 1
    # Calculate cosine factor (0 to 1)
    progress = self.current_step / self.total_steps
    cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
    
    for i, group in enumerate(self.optimizer.param_groups):
      if ('WD_exclude' in group) and group['WD_exclude']:
        continue  # Skip groups excluded from WD scheduling
      initial_wd = self.initial_wds[i]
      # Anneal from Initial WD -> Min WD
      new_wd = self.min_wd + (initial_wd - self.min_wd) * cosine_decay
      group['weight_decay'] = new_wd


def get_optimizer(model: nn.Module, optimizer_name: str, config: dict):
  lr = config['lr']
  wd = config['weight_decay']
  exclude_bias_wd = config.get('exclude_bias_wd', True)
  if exclude_bias_wd:
    list_wd_params = []
    list_no_wd_params = []
    for name, param in model.named_parameters():
      if not param.requires_grad:
        continue
      if 'bias' in name or len(param.shape) == 1:
        list_no_wd_params.append(param)
      else:
        list_wd_params.append(param)
    params = [
        {'params': list_wd_params, 'weight_decay': wd},
        {'params': list_no_wd_params, 'weight_decay': 0.0, 'WD_exclude': True}
    ]
  else:
    params = [{'params': model.parameters(), 'weight_decay': wd}]
  
  if optimizer_name.lower() == 'adamw':
    optimizer = optim.AdamW(params, lr=lr)
  elif optimizer_name.lower() == 'sgd':
    optimizer = optim.SGD(params, lr=lr)
  elif optimizer_name.lower() == 'adam':
    optimizer = optim.Adam(params, lr=lr)
  else:
    raise ValueError(f"Optimizer {optimizer_name} not supported.")
  return optimizer

def get_lr_scheduler(optimizer: optim.Optimizer, scheduler_name: str, config: dict):
  lr = config['lr']
  epochs = config['epochs']
  steps_per_epoch = config.get('steps_per_epoch', 100)
  total_steps = epochs * steps_per_epoch

  if scheduler_name.lower() == 'cosine':
    # T_max is usually the total number of epochs
    lr_scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-7)
  
  elif scheduler_name.lower() == 'cosine_restart':
    # T_0 is the number of epochs for the first restart
    T_0 = config.get('first_restart_epochs', 10)
    T_mul = config.get('multiplier_restart', 2)
    lr_scheduler = CosineAnnealingWarmRestarts(optimizer, 
                                                T_0=T_0,
                                                T_mult=T_mul,
                                                eta_min=1e-7)
  # elif scheduler_name.lower() == 'plateau':
    
  elif scheduler_name.lower() == 'step':
    # Drops LR by gamma every step_size epochs
    step_size = config.get('step_size', 5)
    lr_scheduler = StepLR(optimizer, step_size=step_size, gamma=0.1)
    
  elif scheduler_name.lower() == 'onecycle':
    # Requires max_lr and total steps
    lr_scheduler = OneCycleLR(optimizer, 
                              max_lr=lr, 
                              total_steps=total_steps,  # OneCycle works at every optimizer step
                              pct_start=0.3) # 30% warm-up
  else:
    # Default to identity (no change) if None or unknown
    lr_scheduler = None
  return lr_scheduler

def get_training_components(model: nn.Module, 
                            optimizer_name: str, 
                            scheduler_name: str, 
                            wd_scheduler_name: str = None, 
                            config: dict = None):
  """
  Factory function to initialize Optimizer, LR Scheduler, and WD Scheduler.
  """
  if config is None: config = {}
  
  # 1. Initialize Optimizer
  # -----------------------
  optim = get_optimizer(model, optimizer_name, config)
  # 2. Initialize LR Scheduler
  # --------------------------
  lr_scheduler = get_lr_scheduler(optim, scheduler_name, config)
  # 3. Initialize Weight Decay Scheduler
  # ------------------------------------
  wd_scheduler = None
  
  if wd_scheduler_name and wd_scheduler_name.lower() == 'cosine_wd':
    # Wraps the custom class we wrote above
    total_steps = config['epochs'] * config.get('steps_per_epoch', 100)
    wd_scheduler = CustomWeightDecayScheduler(optim, 
                                              min_wd=config['wd_weight_decay'], 
                                              total_steps=config['epochs'] if not isinstance(lr_scheduler,OneCycleLR) else total_steps) # Using epochs for WD scheduling, step applies every epoch

  return optim, lr_scheduler, wd_scheduler