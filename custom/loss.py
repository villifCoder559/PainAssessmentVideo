from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsort import soft_rank

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

"""
Author: Yonglong Tian (yonglong@mit.edu)
Date: May 07, 2020
"""
## From github repo https://github.com/HobbitLong/SupContrast/blob/master/losses.py
class SupConLoss(nn.Module):
  """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
  It also supports the unsupervised contrastive loss in SimCLR"""
  def __init__(self, temperature=0.07, contrast_mode='all',
                base_temperature=0.07):
    super(SupConLoss, self).__init__()
    self.temperature = temperature
    self.contrast_mode = contrast_mode
    self.base_temperature = base_temperature

  def forward(self, features, labels=None, mask=None):
    """Compute loss for model. If both `labels` and `mask` are None,
    it degenerates to SimCLR unsupervised loss:
    https://arxiv.org/pdf/2002.05709.pdf

    Args:
        features: hidden vector of shape [bsz, C]. Feature must be normalized.
        labels: ground truth of shape [bsz].
        mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
            has the same class as sample i. Can be asymmetric.
    Returns:
        A loss scalar.
    """
    device = (torch.device('cuda')
              if features.is_cuda
              else torch.device('cpu'))
    if features.ndim == 2:
      features = features.unsqueeze(1)
    if len(features.shape) != 3:
      raise ValueError('`features` needs to be [bsz, n_views, ...],'
                        'at least 3 dimensions are required')
    # if len(features.shape) > 3:
    #   features = features.view(features.shape[0], features.shape[1], -1)

    batch_size = features.shape[0]
    if labels is not None and mask is not None:
      raise ValueError('Cannot define both `labels` and `mask`')
    elif labels is None and mask is None:
      mask = torch.eye(batch_size, dtype=torch.float32).to(device)
    elif labels is not None:
      labels = labels.contiguous().view(-1, 1)
      if labels.shape[0] != batch_size:
        raise ValueError('Num of labels does not match num of features')
      mask = torch.eq(labels, labels.T).float().to(device)
    else:
      mask = mask.float().to(device)

    contrast_count = features.shape[1]
    contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0) 
    if self.contrast_mode == 'one':
      anchor_feature = features[:, 0]
      anchor_count = 1
    elif self.contrast_mode == 'all':
      anchor_feature = contrast_feature
      anchor_count = contrast_count
    else:
      raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

    # compute logits
    anchor_dot_contrast = torch.div(
        torch.matmul(anchor_feature, contrast_feature.T),
        self.temperature)
    # for numerical stability
    logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
    logits = anchor_dot_contrast - logits_max.detach()

    # tile mask
    mask = mask.repeat(anchor_count, contrast_count)
    # mask-out self-contrast cases
    logits_mask = torch.scatter(
        torch.ones_like(mask),
        1,
        torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
        0
    )
    mask = mask * logits_mask

    # compute log_prob
    exp_logits = torch.exp(logits) * logits_mask
    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

    # compute mean of log-likelihood over positive
    # modified to handle edge cases when there is no positive pair
    # for an anchor point. 
    # Edge case e.g.:- 
    # features of shape: [4,1,...]
    # labels:            [0,1,1,2]
    # loss before mean:  [nan, ..., ..., nan] 
    mask_pos_pairs = mask.sum(1)
    mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, 1, mask_pos_pairs)
    mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs

    # loss
    loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
    loss = loss.view(anchor_count, batch_size).mean()

    return loss



class SpearmanCorrelationLossOptimized(nn.Module):
  """
  Optimized version that computes all correlations in a more vectorized way.
  """
  
  def __init__(self, regularization="l2", regularization_strength=1.0):
    super(SpearmanCorrelationLossOptimized, self).__init__()
    self.regularization = regularization
    self.regularization_strength = regularization_strength
  
  def forward(self, representations, labels, normalize_features=True):
    """
    Compute Spearman correlation ranking loss (optimized version).
    
    Args:
      representations: Tensor of shape (batch_size, feature_dim)
      labels: Tensor of shape (batch_size,) or (batch_size, 1)
        
    Returns:
      Loss value (negative mean Spearman correlation)
    """
    
    if labels.dim() == 2:
      labels = labels.squeeze(-1)
    
    if normalize_features:
      representations = F.normalize(representations, dim=1)
    
    # Compute pairwise cosine similarities, 
    similarities = torch.mm(representations, representations.t())
    
    # Compute pairwise label distances
    label_distances = torch.abs(labels.unsqueeze(1) - labels.unsqueeze(0))
    
    # Use negative similarity as distance
    rep_distances = -similarities
    
    # Get soft ranks for all rows at once
    # Shape: (batch_size, batch_size)
    rep_ranks = soft_rank(
      rep_distances,
      regularization=self.regularization,
      regularization_strength=self.regularization_strength
    )
    
    label_ranks = soft_rank(
      label_distances,
      regularization=self.regularization,
      regularization_strength=self.regularization_strength
    )
    
    # Center the ranks
    rep_ranks_centered = rep_ranks - rep_ranks.mean(dim=1, keepdim=True)
    label_ranks_centered = label_ranks - label_ranks.mean(dim=1, keepdim=True)
    
    # Compute correlation for each row
    covariance = (rep_ranks_centered * label_ranks_centered).sum(dim=1)
    rep_std = torch.sqrt((rep_ranks_centered ** 2).sum(dim=1))
    label_std = torch.sqrt((label_ranks_centered ** 2).sum(dim=1))
    
    # Compute correlation coefficients
    correlations = covariance / (rep_std * label_std + 1e-8)
    
    # Return negative mean correlation as loss
    loss = - correlations.mean()  # mean over batch
    
    return loss


class SupConLossModified(nn.Module):
  """
  Modified Supervised Contrastive Loss (L_SupConM)
  
  Uses label distance threshold instead of exact label matching.
  P(i) = {j | |y_i - y_j| < theta}
  
  Paper: "Ranking Enhanced Supervised Contrastive Learning for Regression"
  Zhou et al., PAKDD 2024, Section 4.2
  """
  
  def __init__(self, temperature=0.1, theta=1.1):
    """
    Args:
      temperature: Temperature parameter for scaling similarities
      base_temperature: Base temperature for loss scaling
      theta: Threshold for label distance to define positive pairs
             P(i) = {j | |y_i - y_j| < theta}
    """
    super(SupConLossModified, self).__init__()
    self.temperature = temperature
    self.theta = theta
  
  def get_label_mask(self, labels, theta):
    """
    Generate mask for positive pairs based on label distances.
    P(i) = {j | |y_i - y_j| < theta}
    
    Args:
      labels: Tensor of shape (batch_size,)
      theta: Threshold value
      
    Returns:
      mask: Boolean tensor of shape (batch_size, batch_size)
            mask[i,j] = True if |y_i - y_j| < theta and i != j
    """
    # Compute pairwise label distances
    label_distances = torch.abs(
      labels.unsqueeze(1) - labels.unsqueeze(0)
    )
    
    # Create mask where distance < theta
    mask = label_distances < theta
    
    # Remove self-comparisons
    mask.fill_diagonal_(False)
    
    return mask # [batch_size, batch_size]
  
  def forward(self, features, labels, normalize_features=True):
    """
    Compute modified SupCon loss with label distance threshold.
    
    Args:
      features: features (batch_size, feature_dim)
      labels: Target labels (batch_size,)
      
    Returns:
      SupCon loss value
    """
    device = features.device
    # batch_size = features.shape[0]
    
    # Ensure features are L2-normalized
    if normalize_features:
      features = F.normalize(features, dim=1)
    
    # Compute similarity matrix
    similarity_matrix = torch.mm(features, features.T) / self.temperature # [batch_size, batch_size]
    
    # Get positive mask based on theta
    positive_mask = self.get_label_mask(labels, self.theta).float() # [batch_size, batch_size]
    
    # Mask for all comparisons except self
    logits_mask = torch.ones_like(similarity_matrix)
    logits_mask.fill_diagonal_(0)
    
    # For numerical stability
    logits_max, _ = torch.max(similarity_matrix, dim=1, keepdim=True)
    logits = similarity_matrix - logits_max.detach()
    
    # Compute log probabilities
    exp_logits = torch.exp(logits) * logits_mask
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-8) # [batch_size, batch_size]
    
    # Compute mean of log-likelihood over positive pairs
    num_positives = positive_mask.sum(dim=1)  # [batch_size,]
    
    # Handle samples with no positive pairs
    valid_samples = num_positives > 0
    num_positives = num_positives.clamp_min_(1).float()  # avoid division by zero
    if valid_samples.sum() == 0:
      # No valid positive pairs in batch
      return torch.tensor(0.0, device=device, requires_grad=True)
    
    # Mean log probability over positives
    mean_log_prob_pos = (positive_mask * log_prob).sum(dim=1) / (num_positives + 1e-8)
    
    # Loss with temperature scaling
    loss = - mean_log_prob_pos[valid_samples].mean()
    
    return loss


class RESupConLoss(nn.Module):
  """
  Ranking Enhanced Supervised Contrastive Loss (RESupCon)
  Single-view implementation without augmentation.
  
  Paper: "Ranking Enhanced Supervised Contrastive Learning for Regression"
  Zhou et al., PAKDD 2024
  
  This version:
  - Works with single-view batches (no augmentation required)
  - Uses all samples in batch as anchors and comparisons
  - Uses label distance threshold (theta) for positive pair selection
  - Combines SupCon with Spearman ranking loss
  """
  
  def __init__(self, 
               temperature=0.1,
               theta=1.1,
               lambda_weight=4.0,
               spearman_reg="l2",
               spearman_reg_strength=1.0):
    """
    Args:
      temperature: Temperature for SupCon loss (paper uses 0.1)
      base_temperature: Base temperature for SupCon loss
      theta: Threshold for label distance to define positive pairs
             P(i) = {j | |y_i - y_j| < theta}
             Adjust based on your label range
      lambda_weight: Weight for Spearman ranking loss (λ in paper)
                     Paper suggests λ=4.0
      spearman_reg: Regularization for differentiable ranking ('l2' or 'kl')
      spearman_reg_strength: Regularization strength for ranking
    """
    super(RESupConLoss, self).__init__()
    self.supcon_loss = SupConLossModified(
      temperature=temperature,
      theta=theta
    )
    self.spearman_loss = SpearmanCorrelationLossOptimized(
      regularization=spearman_reg,
      regularization_strength=spearman_reg_strength
    )
    self.lambda_weight = lambda_weight
    self.sup_cont_theta = theta
    self.sup_cont_temperature = temperature
    self.spearman_reg = spearman_reg
    self.spearman_reg_strength = spearman_reg_strength

  def forward(self, features, labels, return_components=False):
    """
    Compute RESupCon loss: L_RESupCon = L_SupConM + λ * L_scr
    
    Args:
      features: Tensor of shape (batch_size, feature_dim)
                Features from encoder (will be L2-normalized)
      labels: Tensor of shape (batch_size,) or (batch_size, 1)
              Target labels
      return_components: If True, returns (loss, loss_dict), 
                         else returns just loss
              
    Returns:
      If return_components=True:
        total_loss: Combined RESupCon loss
        loss_dict: Dictionary with loss components
      else:
        total_loss: Combined RESupCon loss
    """
    # Ensure labels are 1D
    if labels.dim() == 2:
      labels = labels.squeeze(-1)
    
    batch_size = features.shape[0]
    
    # Ensure features are L2-normalized
    features = F.normalize(features, dim=1)

    # Compute L_SupConM with label-distance-based positive pairs
    supcon_loss_value = self.supcon_loss(features, labels, normalize_features=False)
    
    # Compute L_scr (Spearman correlation ranking loss)
    spearman_loss_value = self.spearman_loss(features, labels, normalize_features=False)

    # Combined RESupCon loss (Eq 3 in paper)
    total_loss = supcon_loss_value + self.lambda_weight * spearman_loss_value
    
    if return_components:
      loss_dict = {
        'total': total_loss.item(),
        'supcon': supcon_loss_value.item(),
        'spearman': spearman_loss_value.item(),
        'lambda': self.lambda_weight,
        'theta': self.sup_cont_theta,
        'batch_size': batch_size
      }
      return total_loss, loss_dict
    else:
      return total_loss
  
  
  def get_config(self):
    """Get current configuration"""
    return {
      'temperature': self.sup_cont_temperature,
      'sup_cont_theta': self.sup_cont_theta,
      'sup_con_temperature': self.sup_cont_temperature,
      'lambda_weight': self.lambda_weight,
      'spearman_reg': self.spearman_loss.regularization,
      'spearman_reg_strength': self.spearman_loss.regularization_strength,
      'lambda_weight': self.lambda_weight,
      'supcon_loss': type(self.supcon_loss).__name__,
      'spearman_loss': type(self.spearman_loss).__name__
    }
    
  def __str__(self):
    cfg = self.get_config()
    return f'RESupConLoss(temp={cfg["sup_con_temperature"]}, theta={cfg["sup_cont_theta"]}, λ={cfg["lambda_weight"]}, spearman_reg={cfg["spearman_reg"]}, spearman_reg_strength={cfg["spearman_reg_strength"]})'

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
  def __init__(self, temperature=0.1, base_temperature=0.1,
                weight_mode='exp', weight_scale=1.0, max_label_distance=None):
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
