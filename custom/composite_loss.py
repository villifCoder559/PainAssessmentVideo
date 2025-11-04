import torch
import torch.nn as nn
import torch.nn.functional as F

class CompositeLoss(nn.Module):
  def __init__(self, losses):
    """
    losses: list of dicts, example:
      [{'type':'l1', 'loss_weight':1.0},
        {'type':'ce', 'loss_weight':0.5},
        {'type':'contrastive', 'loss_weight':1.0, ...}]
    """
    super().__init__()
    self.list_losses = nn.ModuleList()
    self.list_loss_weights = []
    self.return_embeddings = False
    self.config_losses = losses

    for dict_loss in losses:
      t = dict_loss['type'].lower()
      weight = dict_loss.get('loss_weight', 1.0)

      if t == 'l1':
        self.list_losses.append(nn.L1Loss())
      elif t == 'l2':
        self.list_losses.append(nn.MSELoss())
      elif t == 'huber':
        # If you need compatibility with older torch, swap to SmoothL1Loss
        self.list_losses.append(nn.HuberLoss(delta=dict_loss.get('delta_huber', 1.0)))
      elif t == 'ce':
        self.list_losses.append(nn.CrossEntropyLoss())
      elif t == 'ce_weight':
        class_weights = dict_loss.get('class_weights', None)
        if class_weights is None:
            raise ValueError("ce_weight requires 'class_weights' in the config dict")
        # move weights to float tensor if needed will be handled in forward device transfer
        self.list_losses.append(nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float)))
      elif t == 'contrastive':
        self.list_losses.append(
            SupConWeightedLoss(
                temperature=dict_loss.get('temperature', 0.07),
                base_temperature=dict_loss.get('base_temperature', 0.07),
                weight_mode=dict_loss.get('weight_mode', 'linear'),
                weight_scale=dict_loss.get('weight_scale', 1.0),
                max_label_distance=dict_loss.get('max_label_distance', None)
            )
        )
        self.return_embeddings = True
      else:
        raise ValueError(f"Unsupported loss type: {dict_loss['type']}")

      self.list_loss_weights.append(weight)

  def forward(self, **dict_inputs):
    # Basic input validation and device determination
    if not self.list_losses:
      raise ValueError("No losses configured in CompositeLoss")

    # Determine device from available tensor inputs
    # device = None
    # for v in dict_inputs.values():
    #   if isinstance(v, torch.Tensor):
    #     device = v.device
    #     break
    # if device is None:
      # fallback to CPU
    device = torch.device('cuda')

    total_loss = torch.tensor(0.0, device=device, requires_grad=True, dtype=torch.float32)
    component_losses = {}

    # iterate with index to access weight reliably
    for idx, loss_module in enumerate(self.list_losses):
      weight = float(self.list_loss_weights[idx])

      # MSE/L1/Huber: expect (predictions, targets)
      if isinstance(loss_module, (nn.L1Loss, nn.MSELoss, nn.HuberLoss, nn.SmoothL1Loss)):
        if 'logits' not in dict_inputs or 'targets' not in dict_inputs:
          raise KeyError(f"Loss {loss_module.__class__.__name__} requires 'logits' and 'targets' in forward()")
        pred = dict_inputs['logits']
        targ = dict_inputs['targets'].float()
        loss = loss_module(pred, targ) * weight
        total_loss = total_loss + loss  # preserve gradient

      # CrossEntropyLoss: expects (N, C) and (N,)
      elif isinstance(loss_module, nn.CrossEntropyLoss):
        if 'logits' not in dict_inputs or 'targets' not in dict_inputs:
          raise KeyError("CrossEntropyLoss requires 'logits' and 'targets' in forward()")
        logits = dict_inputs['logits']
        targets = dict_inputs['targets'].long()
        # ensure weight tensor (if present) is on correct device
        if loss_module.weight is not None and loss_module.weight.device != logits.device:
          loss_module.weight = loss_module.weight.to(logits.device)
        loss = loss_module(logits, targets) * weight
        total_loss = total_loss + loss

      # Contrastive / custom: check by class name if class isn't importable here
      elif loss_module.__class__.__name__ == 'SupConWeightedLoss':
        if 'embeddings' not in dict_inputs or 'targets' not in dict_inputs:
          raise KeyError("SupConWeightedLoss requires 'embeddings' and 'targets' in forward()")
        features = dict_inputs['embeddings']
        labels = dict_inputs['targets']
        loss = loss_module(features, labels) * weight
        total_loss = total_loss + loss

      else:
        raise ValueError(f"Unsupported loss module type: {type(loss_module).__name__}")

      # Save scalar for logging (detach to move to CPU)
      component_losses[type(loss_module).__name__] = loss.detach().cpu().item()

    out = {'total_loss': total_loss, 'component_losses': component_losses}
    # Optionally return embeddings if requested and present
    # if self.return_embeddings and 'embeddings' in dict_inputs:
    #   out['embeddings'] = dict_inputs['embeddings']
    return out



class CCCLoss(nn.Module):
  """
  Concordance Correlation Coefficient loss: returns (1 - CCC).

  Args:
    eps (float): small value for numeric stability.
    per_sample (bool):
      - False (default): compute CCC across all elements (flattened) and return a scalar.
      - True: compute CCC per-sample across dims 1.. and reduce according to `reduction`.
    reduction (str): 'mean'|'sum'|'none' used only when per_sample=True.
  """
  def __init__(self, eps: float = 1e-8, per_sample: bool = False, reduction: str = "mean"):
    super().__init__()
    assert reduction in ("mean", "sum", "none")
    self.eps = eps
    self.per_sample = per_sample
    self.reduction = reduction

  def forward(self, preds: torch.Tensor, target: torch.Tensor):
    
    if target.dtype.is_floating_point is False:
      target = target.float()
    if preds.dtype.is_floating_point is False:
      preds = preds.float()
    if preds.shape != target.shape:
      raise ValueError(f"preds and target must have same shape, got {preds.shape} vs {target.shape}")

    if not self.per_sample:
      # Flatten everything and compute a single CCC
      p = preds.view(-1)
      t = target.view(-1)

      mu_p = p.mean()
      mu_t = t.mean()
      var_p = p.var(unbiased=False)
      var_t = t.var(unbiased=False)
      cov = ((p - mu_p) * (t - mu_t)).mean()
      ccc = (2.0 * cov) / (var_p + var_t + (mu_p - mu_t) ** 2 + self.eps)
      return 1.0 - ccc

    # Per-sample CCC: compute CCC for each sample across dims 1..
    p = preds.view(preds.size(0), -1)   # [B, L]
    t = target.view(target.size(0), -1) # [B, L]

    mu_p = p.mean(dim=1)    # [B]
    mu_t = t.mean(dim=1)    # [B]
    var_p = p.var(dim=1, unbiased=False)  # [B]
    var_t = t.var(dim=1, unbiased=False)  # [B]
    cov = ((p - mu_p.unsqueeze(1)) * (t - mu_t.unsqueeze(1))).mean(dim=1)  # [B]

    ccc = (2.0 * cov) / (var_p + var_t + (mu_p - mu_t) ** 2 + self.eps)  # [B]
    loss = 1.0 - ccc  # [B]

    if self.reduction == "mean":
      return loss.mean()
    elif self.reduction == "sum":
      return loss.sum()
    else:  # 'none'
      return loss

class SupConWeightedLoss(torch.nn.Module):
  """
  Supervised Contrastive Loss but with continuous pairwise weights derived from label distances.
  - features: [B, D] (or [B, n_views, D] supported)
  - labels:   [B] integer ordinal labels (e.g., 0..4)
  - weight_mode: 'linear' or 'exp'  (how weight depends on abs(label_i - label_j))
  - weight_scale: for 'exp', larger -> faster decay (alpha)
  - max_label_distance: normalization denominator for linear
  Case: LossSummationLocation.OUTSIDE (link: https://dl.acm.org/doi/abs/10.5555/3495724.3497291)
  """
  def __init__(self, temperature=0.07, base_temperature=0.07,
                weight_mode='linear', weight_scale=1.0, max_label_distance=None):
    super().__init__()
    self.temperature = temperature
    self.base_temperature = base_temperature
    assert weight_mode in ('linear', 'exp'), "weight_mode must be 'linear' or 'exp'"
    self.weight_mode = weight_mode
    self.weight_scale = float(weight_scale)
    self.max_label_distance = max_label_distance  # if None, computed from labels at runtime

  def _compute_weight_matrix(self, labels):
    """
    labels: [B] integer tensor on CPU or device
    returns weights: [B, B] with 1 for identical labels (max), lower for distant labels
    """
    device = labels.device
    labels = labels.view(-1, 1).float()
    absdiff = torch.abs(labels - labels.T)  # [B,B]

    # decide denominator
    if self.max_label_distance is None:
      max_dist = float(torch.max(absdiff).item()) if absdiff.numel() > 0 else 1.0
      if max_dist == 0:
        max_dist = 1.0
    else:
      max_dist = float(self.max_label_distance)

    if self.weight_mode == 'linear':
      # map distance 0 -> weight 1, distance max_dist -> weight 0
      w = 1.0 - (absdiff / max_dist)
      # clamp to [0,1]
      w = torch.clamp(w, min=0.0, max=1.0)
    else:  # 'exp'
      # exp(-alpha * distance) -> distance 0 -> 1; larger distance -> smaller
      alpha = self.weight_scale
      w = torch.exp(-alpha * absdiff)

    # zero out diagonal (we exclude self-contrast)
    w = w * (1.0 - torch.eye(w.size(0), device=device))
    return w

  def forward(self, features, labels=None, mask=None):
    """
    features: [B, D] or [B, n_views, D]
    labels:   [B] integer labels (required unless mask provided)
    mask:     optional [B, B] continuous weights (overrides labels)
    """
    device = features.device

    if features.dim() == 3:
      batch_size = features.shape[0]
      n_views = features.shape[1]
      feat_dim = features.shape[2]
      features = features.view(batch_size * n_views, feat_dim)
    else:
      batch_size = features.shape[0]

    # normalize features 
    features = F.normalize(features, p=2, dim=1)

    if labels is None and mask is None:
      raise ValueError("Either labels or mask must be provided.")
    if mask is not None:
      # mask must be [B,B]
      weights = mask.clone().to(device).float()
      weights = weights * (1.0 - torch.eye(weights.size(0), device=device))
    else:
      weights = self._compute_weight_matrix(labels.to(device))

    # compute logits (similarity matrix)
    logits = torch.div(torch.matmul(features, features.T), self.temperature)

    # numerical stability
    logits_max, _ = torch.max(logits, dim=1, keepdim=True)
    logits = logits - logits_max.detach()

    # mask to remove self-contrast from denominator
    logits_mask = (1.0 - torch.eye(batch_size, device=device))
    exp_logits = torch.exp(logits) * logits_mask

    # log_prob: log softmax over positives+negatives (self excluded)
    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)

    # weighted positive log-probabilities
    # weights: [B,B], log_prob: [B,B]
    weights_sum = weights.sum(1)  # sum of weights for each anchor
    valid = weights_sum > 0

    if valid.sum() == 0:
      # no positives with nonzero weight in batch -> return zero loss
      return torch.tensor(0.0, device=device, requires_grad=True)

    # compute mean log-likelihood over positives for each anchor
    mean_log_prob_pos = (weights * log_prob).sum(1) / (weights_sum + 1e-12)

    loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
    loss = loss[valid].mean()
    return loss
