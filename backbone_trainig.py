import argparse
import datetime
import json
import math
import os
import random
from functools import partial
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from MAE_DFER import modeling_pretrain as dfer_modeling


class Logger:
  def __init__(self, file_path: Path | None, rank: int):
    self.file_path = file_path
    self.rank = rank
    if self.rank == 0 and self.file_path is not None:
      self.file_path.parent.mkdir(parents=True, exist_ok=True)
      with self.file_path.open('w', encoding='utf-8') as f:
        f.write('')

  def log(self, msg: str):
    if self.rank == 0:
      print(msg, flush=True)
      if self.file_path is not None:
        with self.file_path.open('a', encoding='utf-8') as f:
          f.write(msg + '\n')


def parse_args():
  parser = argparse.ArgumentParser('Fine-tuning script (split + train + val + test).')
  parser.add_argument('--dataset_root_path', type=str, required=True)
  parser.add_argument('--csv_path', type=str, required=True)
  parser.add_argument('--output_root', type=str, default='finetune_runs')
  parser.add_argument('--run_name', type=str, default='')
  parser.add_argument('--force_regenerate_split', action='store_true')

  parser.add_argument('--train_ratio', type=float, default=0.70)
  parser.add_argument('--val_ratio', type=float, default=0.15)
  parser.add_argument('--test_ratio', type=float, default=0.15)
  parser.add_argument('--seed', type=int, default=42)

  parser.add_argument('--batch_size', type=int, default=2)
  parser.add_argument('--num_workers', type=int, default=4)
  parser.add_argument('--epochs', type=int, default=200)
  parser.add_argument('--lr', type=float, default=1e-4)
  parser.add_argument('--weight_decay', type=float, default=0.05)
  parser.add_argument('--warmup_epochs', type=int, default=10)
  parser.add_argument('--min_lr', type=float, default=1e-6)
  parser.add_argument('--loss', type=str, default='ce', choices=['ce', 'l1', 'l2'])

  parser.add_argument('--checkpoint_path', type=str, required=True)
  parser.add_argument('--resume', type=str, default='')

  parser.add_argument('--chunk_size', type=int, default=16)
  parser.add_argument('--save_every', type=int, default=50)
  parser.add_argument('--plot_every', type=int, default=5)
  parser.add_argument('--log_every', type=int, default=10)
  parser.add_argument('--gpu_id', type=int, default=0)
  
  # Model architecture parameters (should match the pretrained model)
  parser.add_argument('--encoder_depth', type=int, default=16)
  parser.add_argument('--decoder_depth', type=int, default=4)
  parser.add_argument('--img_size', type=int, default=160)
  parser.add_argument('--patch_size', type=int, default=16)
  parser.add_argument('--encoder_embed_dim', type=int, default=512)
  parser.add_argument('--encoder_num_heads', type=int, default=8)
  parser.add_argument('--decoder_num_classes', type=int, default=1536)
  parser.add_argument('--decoder_embed_dim', type=int, default=384)
  parser.add_argument('--decoder_num_heads', type=int, default=6)
  parser.add_argument('--attn_type', type=str, default='local_global')
  parser.add_argument('--lg_region_size_t', type=int, default=2)
  parser.add_argument('--lg_region_size_h', type=int, default=5)
  parser.add_argument('--lg_region_size_w', type=int, default=10)
  parser.add_argument('--lg_first_attn_type', type=str, default='self')
  parser.add_argument('--lg_third_attn_type', type=str, default='cross')
  parser.add_argument('--lg_attn_param_sharing_first_third', action='store_true')
  parser.add_argument('--lg_attn_param_sharing_all', action='store_true')
  parser.add_argument('--lg_no_second', action='store_true')
  parser.add_argument('--lg_no_third', action='store_true')

  return parser.parse_args()


def setup_seed(seed: int):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)


def setup_distributed(gpu_id: int):
  if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl', init_method='env://')
    distributed = True
    device = torch.device('cuda', local_rank)
  else:
    rank = 0
    world_size = 1
    distributed = False
    if torch.cuda.is_available():
      torch.cuda.set_device(gpu_id)
      device = torch.device('cuda', gpu_id)
    else:
      device = torch.device('cpu')
  return distributed, rank, world_size, device


def cleanup_distributed(distributed: bool):
  if distributed and dist.is_initialized():
    dist.destroy_process_group()


def read_csv_auto(csv_path: str) -> pd.DataFrame:
  return pd.read_csv(csv_path, sep=None, engine='python', dtype={'sample_name': str, 'subject_name': str})


def split_subjects_stratified(
  df: pd.DataFrame,
  train_ratio: float,
  val_ratio: float,
  test_ratio: float,
  seed: int,
):
  if not math.isclose(train_ratio + val_ratio + test_ratio, 1.0, rel_tol=1e-5):
    raise ValueError('train_ratio + val_ratio + test_ratio must be 1.0')

  rng = random.Random(seed)

  class_ids = sorted(df['class_id'].unique().tolist())
  class_to_idx = {c: i for i, c in enumerate(class_ids)}
  n_classes = len(class_ids)

  grouped = df.groupby('subject_id')
  subject_ids = list(grouped.groups.keys())
  n_subjects = len(subject_ids)

  target_subjects = {
    'train': int(round(train_ratio * n_subjects)),
    'val': int(round(val_ratio * n_subjects)),
  }
  target_subjects['test'] = n_subjects - target_subjects['train'] - target_subjects['val']
  if target_subjects['test'] < 0:
    target_subjects['test'] = 0

  total_class_counts = np.zeros(n_classes, dtype=np.float64)
  subject_vectors = {}
  subject_sizes = {}
  for subject_id, sdf in grouped:
    vec = np.zeros(n_classes, dtype=np.float64)
    for class_id, count in sdf['class_id'].value_counts().items():
      vec[class_to_idx[int(class_id)]] = float(count)
    subject_vectors[subject_id] = vec
    subject_sizes[subject_id] = len(sdf)
    total_class_counts += vec

  target_class = {
    'train': total_class_counts * train_ratio,
    'val': total_class_counts * val_ratio,
    'test': total_class_counts * test_ratio,
  }

  assignment = {'train': [], 'val': [], 'test': []}
  current_class = {k: np.zeros(n_classes, dtype=np.float64) for k in assignment}

  subjects_order = sorted(subject_ids, key=lambda s: subject_sizes[s], reverse=True)
  rng.shuffle(subjects_order)
  subjects_order = sorted(subjects_order, key=lambda s: subject_sizes[s], reverse=True)

  for subject_id in subjects_order:
    best_split = None
    best_score = None
    vec = subject_vectors[subject_id]

    for split_name in ['train', 'val', 'test']:
      if len(assignment[split_name]) >= target_subjects[split_name] and sum(
        len(assignment[k]) for k in assignment
      ) < n_subjects:
        continue

      after_subject_count = len(assignment[split_name]) + 1
      subject_ratio_error = abs(after_subject_count - target_subjects[split_name])

      after_class = current_class[split_name] + vec
      class_error = np.abs(after_class - target_class[split_name]).mean()

      score = 10.0 * subject_ratio_error + class_error
      if best_score is None or score < best_score:
        best_score = score
        best_split = split_name

    if best_split is None:
      best_split = min(['train', 'val', 'test'], key=lambda k: len(assignment[k]))

    assignment[best_split].append(subject_id)
    current_class[best_split] += vec

  if len(assignment['val']) == 0 and len(assignment['train']) > 1:
    assignment['val'].append(assignment['train'].pop())
  if len(assignment['test']) == 0 and len(assignment['train']) > 1:
    assignment['test'].append(assignment['train'].pop())

  return assignment


def save_or_load_splits(df: pd.DataFrame, args, split_dir: Path, logger: Logger):
  split_dir.mkdir(parents=True, exist_ok=True)
  train_csv = split_dir / 'train.csv'
  val_csv = split_dir / 'val.csv'
  test_csv = split_dir / 'test.csv'

  if (
    train_csv.exists()
    and val_csv.exists()
    and test_csv.exists()
    and not args.force_regenerate_split
  ):
    logger.log(f'Using existing splits from: {split_dir}')
    train_df = read_csv_auto(str(train_csv))
    val_df = read_csv_auto(str(val_csv))
    test_df = read_csv_auto(str(test_csv))
    return train_df, val_df, test_df

  assignment = split_subjects_stratified(
    df=df,
    train_ratio=args.train_ratio,
    val_ratio=args.val_ratio,
    test_ratio=args.test_ratio,
    seed=args.seed,
  )

  train_df = df[df['subject_id'].isin(assignment['train'])].copy()
  val_df = df[df['subject_id'].isin(assignment['val'])].copy()
  test_df = df[df['subject_id'].isin(assignment['test'])].copy()

  train_df.to_csv(train_csv, sep='\t', index=False)
  val_df.to_csv(val_csv, sep='\t', index=False)
  test_df.to_csv(test_csv, sep='\t', index=False)

  logger.log(f'Created splits and saved to: {split_dir}')
  logger.log(
    f"Split subjects -> train: {train_df['subject_id'].nunique()}, "
    f"val: {val_df['subject_id'].nunique()}, test: {test_df['subject_id'].nunique()}"
  )
  return train_df, val_df, test_df


class VideoFullDataset(Dataset):
  def __init__(
    self,
    df: pd.DataFrame,
    dataset_root_path: str,
    chunk_size: int,
    img_size: int,
  ):
    self.df = df.reset_index(drop=True)
    self.dataset_root_path = dataset_root_path
    self.chunk_size = chunk_size
    self.img_size = img_size
    self.mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
    self.std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)

  def __len__(self):
    return len(self.df)

  def _load_video_frames(self, video_path: str):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
      raise RuntimeError(f'Cannot open video: {video_path}')

    frames = []
    while True:
      ret, frame = cap.read()
      if not ret:
        break
      frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
      frames.append(frame)
    cap.release()

    if len(frames) == 0:
      raise RuntimeError(f'Video has zero frames: {video_path}')

    frames = np.stack(frames, axis=0)
    return torch.from_numpy(frames)

  def _preprocess_and_chunk(self, frames: torch.Tensor):
    # [T, H, W, C] -> [T, C, H, W]
    frames = frames.permute(0, 3, 1, 2).float()
    frames = F.interpolate(frames, size=(self.img_size, self.img_size), mode='bilinear', align_corners=False)
    frames = frames * (1.0 / 255.0)
    frames = (frames - self.mean) / self.std

    num_frames = frames.shape[0]
    remainder = num_frames % self.chunk_size
    if remainder != 0:
      pad_count = self.chunk_size - remainder
      pad_frame = frames[-1:].repeat(pad_count, 1, 1, 1)
      frames = torch.cat([frames, pad_frame], dim=0)

    num_chunks = frames.shape[0] // self.chunk_size
    chunks = frames.view(num_chunks, self.chunk_size, 3, self.img_size, self.img_size)
    chunks = chunks.permute(0, 2, 1, 3, 4).contiguous()  # [N, C, T, H, W]
    return chunks

  def __getitem__(self, idx):
    row = self.df.iloc[idx]
    video_path = os.path.join(
      self.dataset_root_path,
      str(row['subject_name']),
      f"{row['sample_name']}.mp4",
    )
    frames = self._load_video_frames(video_path)
    chunks = self._preprocess_and_chunk(frames)
    class_id = int(row['class_id'])
    return {
      'video': chunks,
      'class_id': class_id,
      'sample_name': str(row['sample_name']),
      'subject_id': int(row['subject_id']),
      'video_path': video_path,
    }


def collate_video_batch(batch):
  videos = [x['video'] for x in batch]
  labels = torch.tensor([x['class_id'] for x in batch], dtype=torch.long)
  samples = [x['sample_name'] for x in batch]
  subjects = [x['subject_id'] for x in batch]
  paths = [x['video_path'] for x in batch]
  return videos, labels, samples, subjects, paths


class FineTuneVideoModel(nn.Module):
  def __init__(self, args, num_outputs: int):
    super().__init__()
    dfer_args = {
      'encoder_depth': args.encoder_depth,
      'decoder_depth': args.decoder_depth,
      'attn_type': args.attn_type,
      'lg_region_size': (args.lg_region_size_t, args.lg_region_size_h, args.lg_region_size_w),
      'img_size': args.img_size,
      'patch_size': args.patch_size,
      'encoder_embed_dim': args.encoder_embed_dim,
      'encoder_num_heads': args.encoder_num_heads,
      'encoder_num_classes': 0,
      'decoder_num_classes': args.decoder_num_classes,
      'decoder_embed_dim': args.decoder_embed_dim,
      'decoder_num_heads': args.decoder_num_heads,
      'mlp_ratio': 4,
      'qkv_bias': True,
      'norm_layer': partial(nn.LayerNorm, eps=1e-6),
      'lg_first_attn_type': args.lg_first_attn_type,
      'lg_third_attn_type': args.lg_third_attn_type,
      'lg_attn_param_sharing_first_third': args.lg_attn_param_sharing_first_third,
      'lg_attn_param_sharing_all': args.lg_attn_param_sharing_all,
      'lg_no_second': args.lg_no_second,
      'lg_no_third': args.lg_no_third,
    }
    pretrain_model = dfer_modeling.PretrainVisionTransformer(**dfer_args)
    self.backbone = pretrain_model.encoder
    self.head = nn.Linear(args.encoder_embed_dim, num_outputs)

  def forward(self, x):
    # x: [N_chunks, C, T, H, W]
    feats = self.backbone.forward_features(x, mask=None)
    pooled = feats.mean(dim=1)
    out = self.head(pooled)
    return out


def build_scheduler(optimizer, total_steps: int, warmup_steps: int, min_lr_ratio: float):
  def lr_lambda(step: int):
    if step < warmup_steps:
      return float(step) / float(max(1, warmup_steps))
    progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

  return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def safe_all_reduce_sum(tensor: torch.Tensor, distributed: bool):
  if distributed:
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
  return tensor


def gather_lists(local_list, distributed: bool):
  if not distributed:
    return local_list
  gathered = [None for _ in range(dist.get_world_size())]
  dist.all_gather_object(gathered, local_list)
  merged = []
  for g in gathered:
    merged.extend(g)
  return merged


def compute_confusion_matrix(y_true, y_pred, class_ids):
  idx = {c: i for i, c in enumerate(class_ids)}
  cm = np.zeros((len(class_ids), len(class_ids)), dtype=np.int64)
  for t, p in zip(y_true, y_pred):
    if t in idx and p in idx:
      cm[idx[t], idx[p]] += 1
  return cm


def robust_load_pretrained(backbone: nn.Module, checkpoint_path: str, logger: Logger):
  ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

  candidates = []
  if isinstance(ckpt, dict):
    for key in ['state_dict', 'model', 'module']:
      if key in ckpt and isinstance(ckpt[key], dict):
        candidates.append((key, ckpt[key]))
    candidates.append(('root', ckpt))
  else:
    raise RuntimeError('Unsupported checkpoint format for robust loading.')

  target = backbone.state_dict()
  best_loaded = {}
  best_name = None

  for name, src in candidates:
    loaded = {}
    for k, v in src.items():
      if not isinstance(v, torch.Tensor):
        continue
      k_new = k
      if k_new.startswith('module.'):
        k_new = k_new[len('module.'):]
      if k_new.startswith('encoder.'):
        k_new = k_new[len('encoder.'):]
      if k_new.startswith('backbone.'):
        k_new = k_new[len('backbone.'):]

      if k_new.startswith('decoder') or 'encoder_to_decoder' in k_new or 'mask_token' in k_new:
        continue
      if k_new.startswith('head'):
        continue

      if k_new in target and target[k_new].shape == v.shape:
        loaded[k_new] = v

    if len(loaded) > len(best_loaded):
      best_loaded = loaded
      best_name = name

  if len(best_loaded) == 0:
    raise RuntimeError('No compatible pretrained weights found for backbone.')

  missing, unexpected = backbone.load_state_dict(best_loaded, strict=False)
  logger.log(
    f'Robust loading from "{checkpoint_path}" using source "{best_name}" -> '
    f'loaded={len(best_loaded)}, missing={len(missing)}, unexpected={len(unexpected)}'
  )


def video_level_forward(model, videos, labels, class_to_idx, criterion, loss_name, device):
  batch_losses = []
  pred_class_ids = []
  true_class_ids = labels.cpu().tolist()

  for i, video_chunks in enumerate(videos):
    video_chunks = video_chunks.to(device, non_blocking=True)
    outputs_per_chunk = model(video_chunks)
    video_output = outputs_per_chunk.mean(dim=0, keepdim=True)

    class_id = int(labels[i].item())

    if loss_name == 'ce':
      target_idx = torch.tensor([class_to_idx[class_id]], dtype=torch.long, device=device)
      loss = criterion(video_output, target_idx)
      pred_idx = int(torch.argmax(torch.softmax(video_output, dim=-1), dim=-1).item())
      inv = {v: k for k, v in class_to_idx.items()}
      pred_class_id = int(inv[pred_idx])
    else:
      target_value = torch.tensor([float(class_id)], dtype=torch.float32, device=device)
      pred_value = video_output.view(-1)
      loss = criterion(pred_value, target_value)
      pred_class_id = int(torch.round(pred_value.detach()).item())

    batch_losses.append(loss)
    pred_class_ids.append(pred_class_id)

  loss_tensor = torch.stack(batch_losses).mean()
  return loss_tensor, pred_class_ids, true_class_ids


def run_epoch(
  model,
  loader,
  optimizer,
  scheduler,
  criterion,
  class_to_idx,
  loss_name,
  device,
  distributed,
  train_mode,
  logger=None,
  epoch_idx=None,
  total_epochs=None,
  phase='train',
  log_every=10,
):
  if train_mode:
    model.train()
  else:
    model.eval()

  total_loss = 0.0
  total_count = 0
  total_correct = 0
  all_true = []
  all_pred = []

  for step_idx, (videos, labels, _, _, _) in enumerate(loader, start=1):
    labels = labels.to(device, non_blocking=True)

    if train_mode:
      optimizer.zero_grad(set_to_none=True)

    with torch.set_grad_enabled(train_mode):
      loss, pred_class_ids, true_class_ids = video_level_forward(
        model=model,
        videos=videos,
        labels=labels,
        class_to_idx=class_to_idx,
        criterion=criterion,
        loss_name=loss_name,
        device=device,
      )
      if train_mode:
        loss.backward()
        optimizer.step()
        scheduler.step()

    total_loss += float(loss.item()) * len(true_class_ids)
    total_count += len(true_class_ids)
    total_correct += sum(int(p == t) for p, t in zip(pred_class_ids, true_class_ids))
    all_true.extend(true_class_ids)
    all_pred.extend(pred_class_ids)

    if logger is not None and (step_idx % max(1, log_every) == 0 or step_idx == len(loader)):
      avg_loss = total_loss / max(1, total_count)
      avg_acc = total_correct / max(1, total_count)
      if epoch_idx is not None and total_epochs is not None:
        logger.log(
          f'[{phase}] epoch {epoch_idx + 1}/{total_epochs} '
          f'step {step_idx}/{len(loader)} '
          f'loss={avg_loss:.6f} acc={avg_acc:.4f}'
        )
      else:
        logger.log(
          f'[{phase}] step {step_idx}/{len(loader)} '
          f'loss={avg_loss:.6f} acc={avg_acc:.4f}'
        )

  loss_tensor = torch.tensor([total_loss, total_count, total_correct], dtype=torch.float64, device=device)
  loss_tensor = safe_all_reduce_sum(loss_tensor, distributed)
  global_loss = float(loss_tensor[0].item() / max(1.0, loss_tensor[1].item()))
  global_acc = float(loss_tensor[2].item() / max(1.0, loss_tensor[1].item()))

  all_true = gather_lists(all_true, distributed)
  all_pred = gather_lists(all_pred, distributed)
  return global_loss, global_acc, all_true, all_pred


def save_confusion_csv(path: Path, y_true, y_pred, class_ids):
  cm = compute_confusion_matrix(y_true, y_pred, class_ids)
  cm_df = pd.DataFrame(cm, index=class_ids, columns=class_ids)
  cm_df.to_csv(path)


def save_loss_plot(path: Path, train_losses, val_losses):
  plt.figure(figsize=(8, 5))
  plt.plot(train_losses, label='train_loss')
  plt.plot(val_losses, label='val_loss')
  plt.xlabel('Epoch')
  plt.ylabel('Loss')
  plt.legend()
  plt.tight_layout()
  plt.savefig(path)
  plt.close()


def create_run_dir(args):
  timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
  base = f'finetune_{timestamp}_loss-{args.loss}_lr-{args.lr}_bs-{args.batch_size}_seed-{args.seed}'
  if args.run_name:
    base = f'{base}_{args.run_name}'
  run_dir = Path(args.output_root) / base
  run_dir.mkdir(parents=True, exist_ok=False)
  return run_dir


def maybe_resume(
  model,
  optimizer,
  scheduler,
  resume_path,
  device,
  logger,
):
  if not resume_path:
    return 0, None, None

  ckpt = torch.load(resume_path, map_location=device, weights_only=False)
  model.load_state_dict(ckpt['model'], strict=True)
  optimizer.load_state_dict(ckpt['optimizer'])
  scheduler.load_state_dict(ckpt['scheduler'])
  start_epoch = int(ckpt['epoch']) + 1
  best_val_score = ckpt.get('best_val_score', None)
  best_metric_name = ckpt.get('best_metric_name', None)
  logger.log(f'Resumed training from {resume_path} at epoch {start_epoch}.')
  return start_epoch, best_val_score, best_metric_name


def main():
  args = parse_args()
  distributed, rank, world_size, device = setup_distributed(args.gpu_id)
  setup_seed(args.seed + rank)

  if rank == 0:
    run_dir = create_run_dir(args)
  else:
    run_dir = None

  if distributed:
    obj = [str(run_dir) if rank == 0 else '']
    dist.broadcast_object_list(obj, src=0)
    run_dir = Path(obj[0])

  logger = Logger(run_dir / 'console_output.txt', rank)
  logger.log(f'Run dir: {run_dir}')
  logger.log(f'Distributed: {distributed} | rank={rank} | world_size={world_size} | device={device}')

  full_df = read_csv_auto(args.csv_path)
  required_cols = ['subject_id', 'subject_name', 'class_id', 'sample_name']
  for c in required_cols:
    if c not in full_df.columns:
      raise ValueError(f'Missing required column in CSV: {c}')

  class_ids = sorted(int(x) for x in full_df['class_id'].unique().tolist())
  class_to_idx = {c: i for i, c in enumerate(class_ids)}
  logger.log(f'Adaptive classes from CSV: {class_ids} (num_classes={len(class_ids)})')

  split_dir = run_dir / 'splits'
  train_df, val_df, test_df = save_or_load_splits(full_df, args, split_dir, logger)

  if rank == 0:
    with (run_dir / 'class_mapping.json').open('w', encoding='utf-8') as f:
      json.dump({'class_ids': class_ids, 'class_to_idx': class_to_idx}, f, indent=2)
    with (run_dir / 'args.json').open('w', encoding='utf-8') as f:
      json.dump(vars(args), f, indent=2)

  train_dataset = VideoFullDataset(
    train_df,
    args.dataset_root_path,
    args.chunk_size,
    args.img_size,
  )
  val_dataset = VideoFullDataset(
    val_df,
    args.dataset_root_path,
    args.chunk_size,
    args.img_size,
  )
  test_dataset = VideoFullDataset(
    test_df,
    args.dataset_root_path,
    args.chunk_size,
    args.img_size,
  )

  if distributed:
    train_sampler = DistributedSampler(train_dataset, shuffle=True, drop_last=False)
    val_sampler = DistributedSampler(val_dataset, shuffle=False, drop_last=False)
    test_sampler = DistributedSampler(test_dataset, shuffle=False, drop_last=False)
  else:
    train_sampler = None
    val_sampler = None
    test_sampler = None

  train_loader = DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    shuffle=(train_sampler is None),
    sampler=train_sampler,
    num_workers=args.num_workers,
    pin_memory=True,
    collate_fn=collate_video_batch,
    drop_last=False,
  )
  val_loader = DataLoader(
    val_dataset,
    batch_size=args.batch_size,
    shuffle=False,
    sampler=val_sampler,
    num_workers=args.num_workers,
    pin_memory=True,
    collate_fn=collate_video_batch,
    drop_last=False,
  )
  test_loader = DataLoader(
    test_dataset,
    batch_size=args.batch_size,
    shuffle=False,
    sampler=test_sampler,
    num_workers=args.num_workers,
    pin_memory=True,
    collate_fn=collate_video_batch,
    drop_last=False,
  )

  if args.loss == 'ce':
    num_outputs = len(class_ids)
    criterion = nn.CrossEntropyLoss()
    best_metric_name = 'val_accuracy'
  elif args.loss == 'l1':
    num_outputs = 1
    criterion = nn.L1Loss()
    best_metric_name = 'val_loss'
  else:
    num_outputs = 1
    criterion = nn.MSELoss()
    best_metric_name = 'val_loss'

  model = FineTuneVideoModel(args=args, num_outputs=num_outputs).to(device)
  robust_load_pretrained(model.backbone, args.checkpoint_path, logger)

  for p in model.parameters():
    p.requires_grad = True

  if distributed:
    model = DDP(model, device_ids=[device.index], output_device=device.index, find_unused_parameters=False)

  optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

  steps_per_epoch = len(train_loader)
  total_steps = max(1, args.epochs * steps_per_epoch)
  warmup_steps = max(1, args.warmup_epochs * steps_per_epoch)
  min_lr_ratio = args.min_lr / max(args.lr, 1e-12)
  scheduler = build_scheduler(optimizer, total_steps, warmup_steps, min_lr_ratio)

  start_epoch = 0
  best_val_score = None
  if args.resume:
    start_epoch, best_val_score, _ = maybe_resume(
      model=model,
      optimizer=optimizer,
      scheduler=scheduler,
      resume_path=args.resume,
      device=device,
      logger=logger,
    )

  train_losses = []
  val_losses = []

  ckpt_dir = run_dir / 'checkpoints'
  ckpt_dir.mkdir(exist_ok=True, parents=True)

  for epoch in range(start_epoch, args.epochs):
    if distributed and train_sampler is not None:
      train_sampler.set_epoch(epoch)

    train_loss, train_acc, _, _ = run_epoch(
      model=model,
      loader=train_loader,
      optimizer=optimizer,
      scheduler=scheduler,
      criterion=criterion,
      class_to_idx=class_to_idx,
      loss_name=args.loss,
      device=device,
      distributed=distributed,
      train_mode=True,
      logger=logger,
      epoch_idx=epoch,
      total_epochs=args.epochs,
      phase='train',
      log_every=args.log_every,
    )

    with torch.no_grad():
      val_loss, val_acc, val_true, val_pred = run_epoch(
        model=model,
        loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        class_to_idx=class_to_idx,
        loss_name=args.loss,
        device=device,
        distributed=distributed,
        train_mode=False,
        logger=logger,
        epoch_idx=epoch,
        total_epochs=args.epochs,
        phase='val',
        log_every=args.log_every,
      )

    train_losses.append(train_loss)
    val_losses.append(val_loss)

    if rank == 0:
      logger.log(
        f'Epoch [{epoch + 1}/{args.epochs}] '
        f'train_loss={train_loss:.6f} train_acc={train_acc:.4f} '
        f'val_loss={val_loss:.6f} val_acc={val_acc:.4f}'
      )

      if (epoch + 1) % args.plot_every == 0:
        save_loss_plot(run_dir / 'loss_curve.png', train_losses, val_losses)

      if (epoch + 1) % args.save_every == 0:
        checkpoint = {
          'epoch': epoch,
          'model': model.state_dict(),
          'optimizer': optimizer.state_dict(),
          'scheduler': scheduler.state_dict(),
          'best_val_score': best_val_score,
          'best_metric_name': best_metric_name,
          'args': vars(args),
        }
        torch.save(checkpoint, ckpt_dir / f'epoch_{epoch + 1}.pth')

      current_score = val_acc if args.loss == 'ce' else val_loss
      is_better = False
      if best_val_score is None:
        is_better = True
      elif args.loss == 'ce' and current_score > best_val_score:
        is_better = True
      elif args.loss in ['l1', 'l2'] and current_score < best_val_score:
        is_better = True

      if is_better:
        best_val_score = current_score
        best_ckpt = {
          'epoch': epoch,
          'model': model.state_dict(),
          'optimizer': optimizer.state_dict(),
          'scheduler': scheduler.state_dict(),
          'best_val_score': best_val_score,
          'best_metric_name': best_metric_name,
          'args': vars(args),
        }
        torch.save(best_ckpt, ckpt_dir / 'best.pth')

        save_confusion_csv(run_dir / 'val_confusion_matrix.csv', val_true, val_pred, class_ids)

  if rank == 0:
    last_ckpt = {
      'epoch': args.epochs - 1,
      'model': model.state_dict(),
      'optimizer': optimizer.state_dict(),
      'scheduler': scheduler.state_dict(),
      'best_val_score': best_val_score,
      'best_metric_name': best_metric_name,
      'args': vars(args),
    }
    torch.save(last_ckpt, ckpt_dir / 'last.pth')

  if distributed:
    dist.barrier()

  best_path = ckpt_dir / 'best.pth'
  map_loc = {'cuda:0': f'cuda:{device.index}'} if device.type == 'cuda' else 'cpu'
  best_ckpt = torch.load(best_path, map_location=map_loc, weights_only=False)
  model.load_state_dict(best_ckpt['model'], strict=True)

  with torch.no_grad():
    test_loss, test_acc, test_true, test_pred = run_epoch(
      model=model,
      loader=test_loader,
      optimizer=optimizer,
      scheduler=scheduler,
      criterion=criterion,
      class_to_idx=class_to_idx,
      loss_name=args.loss,
      device=device,
      distributed=distributed,
      train_mode=False,
      logger=logger,
      epoch_idx=args.epochs - 1,
      total_epochs=args.epochs,
      phase='test',
      log_every=args.log_every,
    )

  if rank == 0:
    save_confusion_csv(run_dir / 'test_confusion_matrix.csv', test_true, test_pred, class_ids)
    logger.log(f'TEST -> loss={test_loss:.6f}, accuracy={test_acc:.4f}')
    logger.log(f'Best validation metric ({best_metric_name}) = {best_val_score}')

  cleanup_distributed(distributed)


if __name__ == '__main__':
  main()