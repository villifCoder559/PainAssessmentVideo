import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LinearLR, SequentialLR, CosineAnnealingLR, CosineAnnealingWarmRestarts


class OptimizerFactory:
  """
  Factory class to create the optimizer and the learning rate scheduler.
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
    Creates and returns the optimizer and the lr_scheduler.

    Args:
      model (nn.Module): The model to optimize.

    Returns:
      tuple: (optimizer, lr_scheduler)
    """
    optimizer = self._get_optimizer(model, self.config['optimizer_name'])
    lr_scheduler = self._get_lr_scheduler(optimizer, self.config['scheduler_name'])
    return optimizer, lr_scheduler

  def _get_optimizer(self, model: nn.Module, optimizer_name: str) -> optim.Optimizer:
    """Creates AdamW with weight decay on weights only (not on biases and norm parameters)."""
    if optimizer_name.lower() != 'adamw':
      raise ValueError(f"Optimizer {optimizer_name} not supported. Use adamw.")
    list_wd_params = []
    list_no_wd_params = []
    for name, param in model.named_parameters():
      if not param.requires_grad:
        continue
      if 'bias' in name or len(param.shape) == 1 or 'norm' in name:
        list_no_wd_params.append(param)
      else:
        list_wd_params.append(param)
    params = [
      {'params': list_wd_params, 'weight_decay': self.config['weight_decay']},
      {'params': list_no_wd_params, 'weight_decay': 0.0, 'WD_exclude': True}
    ]
    return optim.AdamW(params, lr=self.config['lr'])

  dict_check_scheduler = {
    'cosine': ['min_lr'],
    'cosine_restart': ['first_restart_epochs', 'multiplier_restart', 'min_lr'],
  }

  @staticmethod
  def check_schedulers_parms(scheduler_name: str, config: dict):
    """Checks if all required parameters for the scheduler are present in the config."""
    if scheduler_name.lower() not in OptimizerFactory.dict_check_scheduler:
      raise ValueError(f"Scheduler {scheduler_name} not supported. Use one of {list(OptimizerFactory.dict_check_scheduler)}.")
    required_params = OptimizerFactory.dict_check_scheduler[scheduler_name.lower()]
    missing_params = [param for param in required_params if param not in config or config[param] is None]
    if missing_params:
      raise ValueError(f"Missing parameters for {scheduler_name} scheduler: {missing_params}")

  def _get_lr_scheduler(self, optimizer: optim.Optimizer, scheduler_name: str):
    """Creates the learning rate scheduler, preceded by an optional linear warm-up."""
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
      lr_scheduler = CosineAnnealingWarmRestarts(optimizer,
                                                 T_0=self.config['first_restart_epochs'] * steps_per_epoch,
                                                 T_mult=self.config['multiplier_restart'],
                                                 eta_min=self.config['min_lr'])
    else:
      raise ValueError(f"Scheduler {scheduler_name} not supported.")

    if warm_up_steps > 0:
      warm_up_name = self.config['warm_up_scheduler']
      if warm_up_name.lower() != 'linear':
        raise ValueError(f"Warm-up scheduler {warm_up_name} not supported.")
      warm_up_scheduler = LinearLR(optimizer,
                                   start_factor=self.config['warm_up_start_factor'],
                                   total_iters=warm_up_steps)
      return SequentialLR(optimizer, schedulers=[warm_up_scheduler, lr_scheduler], milestones=[warm_up_steps])

    return lr_scheduler

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
    lr = optimizer.param_groups[0]['lr']
    wd_bias = 0
    wd_no_bias = 0
    for group in optimizer.param_groups:
      if group.get('WD_exclude', False):
        wd_bias = group['weight_decay']
      else:
        wd_no_bias = group['weight_decay']
    return {'lr': lr, 'wd_no_bias': wd_no_bias, 'wd_bias': wd_bias}
