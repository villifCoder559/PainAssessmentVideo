import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LinearLR,SequentialLR,CosineAnnealingLR, StepLR, OneCycleLR, LambdaLR, CosineAnnealingWarmRestarts, ReduceLROnPlateau
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

class OptimizerFactory:
  """
  Factory class to create an optimizer, learning rate scheduler, and weight decay scheduler.
  """
  def __init__(self, config: dict = None):
    """
    Initializes the factory with a configuration dictionary.
    Args:
        config (dict): A dictionary containing settings for the optimizer and schedulers.
    """
    self.config = config if config is not None else {}

  def get_config(self):
    """Returns the configuration dictionary."""
    return self.config
  def create(self, model: nn.Module):
    """
    Creates and returns the optimizer, lr_scheduler, and wd_scheduler.
    Args:
        model (nn.Module): The model to optimize.
        optimizer_name (str): The name of the optimizer (e.g., 'adamw', 'sgd').
        scheduler_name (str): The name of the LR scheduler (e.g., 'cosine', 'step').
        wd_scheduler_name (str, optional): The name of the WD scheduler (e.g., 'cosine_wd'). Defaults to None.
    Returns:
        tuple: A tuple containing the optimizer, lr_scheduler, and wd_scheduler.
    """
    optimizer = self._get_optimizer(model, self.config['optimizer_name'])
    lr_scheduler = self._get_lr_scheduler(optimizer, self.config['scheduler_name'])
    wd_scheduler = self._get_wd_scheduler(optimizer, self.config['wd_scheduler_name'])
    return optimizer, lr_scheduler, wd_scheduler

  def _get_optimizer(self, model: nn.Module, optimizer_name: str) -> optim.Optimizer:
    """Creates the optimizer."""
    lr = self.config['lr']
    wd = self.config['weight_decay']
    exclude_bias_wd = self.config['exclude_bias_wd']

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
      return optim.AdamW(params, lr=lr)
    elif optimizer_name.lower() == 'sgd':
      return optim.SGD(params, lr=lr)
    elif optimizer_name.lower() == 'adam':
      return optim.Adam(params, lr=lr)
    else:
      raise ValueError(f"Optimizer {optimizer_name} not supported.")
  dict_check_scheduler = {
    'cosine':{
      'list_params': ['min_lr'],
    },
    'cosine_restart':{
      'list_params': ['first_restart_epochs', 'multiplier_restart', 'min_lr'],
    },
    'step':{
      'list_params': ['step_size_epochs', 'gamma'],
    },
    'onecycle':{
      'list_params': ['onecycle_max_lr', 'onecycle_div_factor', 'onecycle_final_div_factor', 'onecycle_pct_start', 'onecycle_anneal_strategy'],
    },
    'warm_up':{
      'list_params': ['warm_up_percent', 'warm_up_scheduler', 'warm_up_start_factor'],
    },
    'wd_scheduler':{
      'list_params': ['wd_scheduler_name', 'min_wd'],
    },
    'lambda':{
      'list_params': [],
    },
  }
  
  @staticmethod
  def check_schedulers_parms(scheduler_name: str, config: dict):
    """Checks if all required parameters for the scheduler are present in the config."""
    if scheduler_name.lower() not in OptimizerFactory.dict_check_scheduler:
      raise ValueError(f"Scheduler {scheduler_name} not supported for parameter checking.")
    
    required_params = OptimizerFactory.dict_check_scheduler[scheduler_name.lower()]['list_params']
    missing_params = [param for param in required_params if param not in config or config[param] is None]
    
    if missing_params:
      raise ValueError(f"Missing parameters for {scheduler_name} scheduler: {missing_params}")
    
  def _get_lr_scheduler(self, optimizer: optim.Optimizer, scheduler_name: str):
    """Creates the learning rate scheduler."""
    epochs = self.config['epochs']
    steps_per_epoch = self.config['steps_per_epoch']
    total_steps = epochs * steps_per_epoch
    warm_up_epochs = self.config['warm_up_epochs'] if self.config['warm_up_epochs'] is not None else 0.0
    warm_up_steps = int(warm_up_epochs * steps_per_epoch)

    main_scheduler_steps = total_steps - warm_up_steps if warm_up_steps > 0 else total_steps
    
    if scheduler_name.lower() == 'cosine':
      lr_scheduler = CosineAnnealingLR(optimizer, 
                                       T_max=main_scheduler_steps,
                                       eta_min=self.config['min_lr'])
    
    elif scheduler_name.lower() == 'cosine_restart':
      T_0 = self.config['first_restart_epochs'] * steps_per_epoch
      T_mul = self.config['multiplier_restart']
      lr_scheduler = CosineAnnealingWarmRestarts(optimizer, 
                                                 T_0=T_0,
                                                 T_mult=T_mul,
                                                 eta_min=self.config['min_lr'])
    
    elif scheduler_name.lower() == 'step':
      step_size = self.config['step_size_epochs'] * steps_per_epoch
      lr_scheduler = StepLR(optimizer, step_size=step_size, gamma=self.config['gamma']) 
      
    elif scheduler_name.lower() == 'onecycle':
      # not chainable with warm-up
      lr_scheduler = OneCycleLR(optimizer, 
                                total_steps=total_steps, 
                                max_lr=self.config['onecycle_max_lr'],
                                div_factor=self.config['onecycle_div_factor'],
                                final_div_factor=self.config['onecycle_final_div_factor'],
                                pct_start=self.config['onecycle_pct_start'],
                                anneal_strategy=self.config['onecycle_anneal_strategy']
                                ) 
    elif scheduler_name.lower() == 'lambda':
      lr_scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: 1.0)
    else:
      raise ValueError(f"Scheduler {scheduler_name} not supported.")

    if warm_up_steps > 0:
      warm_up_name = self.config['warm_up_scheduler']
      if warm_up_name.lower() == 'linear':
        start_factor = self.config['warm_up_start_factor']
        warm_up_scheduler = LinearLR(optimizer, 
                                     start_factor=start_factor, 
                                     total_iters=warm_up_steps)
      else:
        raise ValueError(f"Warm-up scheduler {warm_up_name} not supported.")
      
      return SequentialLR(optimizer, schedulers=[warm_up_scheduler, lr_scheduler], milestones=[warm_up_steps])

    return lr_scheduler

  def _get_wd_scheduler(self, optimizer: optim.Optimizer, wd_scheduler_name: str):
    """Creates the weight decay scheduler."""
    if not wd_scheduler_name or wd_scheduler_name.lower() != 'cosine_wd':
      return None
    
    epochs = self.config['epochs']
    steps_per_epoch = self.config['steps_per_epoch']
    total_steps = epochs * steps_per_epoch
    
    # Use epochs for WD scheduling, unless it's OneCycleLR which steps per batch
    wd_total_steps = total_steps

    return CustomWeightDecayScheduler(
      optimizer,
      min_wd=self.config['wd_weight_decay'],
      total_steps=wd_total_steps
    )
  
  @staticmethod
  def get_monitoring_values(optimizer: optim.Optimizer):
    """
    Returns a dictionary with current learning rate and weight decay values
    for bias and non-bias parameter groups.
    Args:
        optimizer (optim.Optimizer): The optimizer instance.
    Returns:
        dict: A dictionary containing 'lr', 'wd_no_bias', and 'wd_bias'.
    """
    # lr is typically the same across all groups, so we take it from the first one
    lr = optimizer.param_groups[0]['lr']
    
    wd_bias = 0 
    wd_no_bias = 0

    for group in optimizer.param_groups:
      if group.get('WD_exclude', False):
        wd_bias = group['weight_decay']
      else:
        wd_no_bias = group['weight_decay']

    # Handle case where there's only one parameter group
    if wd_bias is None:
        wd_bias = wd_no_bias

    return {'lr': lr, 'wd_no_bias': wd_no_bias, 'wd_bias': wd_bias}
