
# # import os, sys, argparse, math, time, random
# # import numpy as np
# # import torch
# # import torch.nn as nn
# # import torch.optim as optim
# # import torch.nn.functional as F
# # from torch.utils.data import DataLoader, Sampler
# # from data_from_merged_2 import SingleCellFromMerged
# # from metrics import confusion_matrix, macro_f1_from_cm
# # from utils import set_seed
# # # 盡量相容不同 torchvision 版本
# # try:
# #     from torchvision.models import vit_b_16, ViT_B_16_Weights
# #     _HAS_VIT_WEIGHTS = True
# # except Exception:
# #     from torchvision.models.vision_transformer import vit_b_16
# #     _HAS_VIT_WEIGHTS = False

# # # ==================== 全域設定：3 類 ====================
# # NUM_CLASSES = 3  # 0:1to1, 1:1to2, 2:1to4(含k=3或4)
# # # -------------------- Vision Transformer (ViT-B/16) classifier --------------------
# # class ViTCellClassifier(nn.Module):
# #     """
# #     - in_ch 可不是 3（支援 RGB/Gray + mask + 幾何通道）
# #     - 直接改 patch-embed 的 conv 輸入通道
# #     - head 改成 Dropout -> 256 -> ReLU -> Dropout -> num_classes（與你 DenseNet 版一致）
# #     - freeze_until: >0 則凍結 backbone，只訓練 heads
# #     """
# #     def __init__(self, in_ch=4, num_classes=3, pretrained=True, freeze_until=0, drop=0.2):
# #         super().__init__()

# #         # 建 ViT-B/16
# #         if _HAS_VIT_WEIGHTS:
# #             weights = ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
# #             m = vit_b_16(weights=weights)
# #         else:
# #             # 舊版 API 相容：weights 也接受字串
# #             m = vit_b_16(weights=("IMAGENET1K_V1" if pretrained else None))

# #         # ---- 改 patch embedding 的 conv 讓它吃 in_ch ----
# #         conv = getattr(m, "conv_proj", None)
# #         if conv is None:
# #             # timm/其它實作可能是 patch_embed.proj
# #             patch_embed = getattr(m, "patch_embed", None)
# #             conv = getattr(patch_embed, "proj", None)
# #         if conv is None:
# #             raise RuntimeError("ViT patch-embed conv not found (conv_proj or patch_embed.proj).")

# #         if in_ch != conv.in_channels:
# #             new = nn.Conv2d(
# #                 in_ch, conv.out_channels,
# #                 kernel_size=conv.kernel_size, stride=conv.stride,
# #                 padding=conv.padding, bias=(conv.bias is not None)
# #             )
# #             with torch.no_grad():
# #                 w = conv.weight  # [out, 3, k, k]（原本）
# #                 if in_ch >= 3:
# #                     new.weight[:, :3] = w
# #                     if in_ch > 3:
# #                         mean_extra = w[:, :3].mean(dim=1, keepdim=True)
# #                         for c in range(3, in_ch):
# #                             new.weight[:, c:c+1] = mean_extra
# #                 else:
# #                     new.weight[:, :in_ch] = w[:, :in_ch]
# #                 if conv.bias is not None:
# #                     new.bias.copy_(conv.bias)

# #             if hasattr(m, "conv_proj"):
# #                 m.conv_proj = new
# #             else:
# #                 m.patch_embed.proj = new

# #         # ---- 改 classification head：Dropout -> 256 -> ReLU -> Dropout -> num_classes ----
# #         # 取出 head 輸入維度（不同版本 head 結構稍不同，保守寫法）
# #         feat_ch = None
# #         for mod in m.heads.modules():
# #             if isinstance(mod, nn.Linear):
# #                 feat_ch = mod.in_features
# #         if feat_ch is None:
# #             raise RuntimeError("Cannot infer ViT head input dimension.")

# #         m.heads = nn.Sequential(
# #             nn.Dropout(drop),
# #             nn.Linear(feat_ch, 256),
# #             nn.ReLU(inplace=True),
# #             nn.Dropout(drop),
# #             nn.Linear(256, num_classes),
# #         )

# #         self.vit = m

# #         # ---- 凍結策略（>0 就凍結 backbone，只訓練 heads）----
# #         if freeze_until > 0:
# #             for name, p in self.vit.named_parameters():
# #                 if "heads" not in name:
# #                     p.requires_grad = False

# #     def forward(self, x):
# #         return self.vit(x)

# # def build_ema_full_state_dict(model: nn.Module, ema):
# #     """
# #     以 model.state_dict() 為底，將 EMA 追蹤到的「參數張量」覆蓋進去；
# #     BN 的 buffers（running_mean/var, num_batches_tracked）等會原封保留，
# #     這樣就能 strict=True 載入。
# #     """
# #     sd = model.state_dict()  # 包含參數+buffers 的完整 dict
# #     if ema is None:
# #         return sd
# #     with torch.no_grad():
# #         # 只覆蓋「參數」鍵；buffers 不動
# #         for name, _ in model.named_parameters():
# #             if name in ema.shadow:
# #                 sd[name] = ema.shadow[name].detach().clone()
# #     return sd

# # # -------------------- DenseNet-161 classifier --------------------
# # class DenseNetCellClassifier(nn.Module):
# #     def __init__(self, in_ch=4, num_classes=NUM_CLASSES, pretrained=True, freeze_until=0, drop=0.2):
# #         super().__init__()
# #         m = densenet161(weights="IMAGENET1K_V1" if pretrained else None)

# #         # 改第一層輸入通道
# #         conv0 = m.features.conv0
# #         if in_ch != 3:
# #             w = conv0.weight
# #             new = nn.Conv2d(in_ch, conv0.out_channels, kernel_size=conv0.kernel_size,
# #                             stride=conv0.stride, padding=conv0.padding, bias=False)
# #             with torch.no_grad():
# #                 if in_ch >= 3:
# #                     new.weight[:, :3] = w
# #                     if in_ch > 3:
# #                         mean_extra = w.mean(dim=1, keepdim=True)
# #                         for i in range(3, in_ch):
# #                             new.weight[:, i:i+1] = mean_extra
# #                 else:
# #                     new.weight[:, :in_ch] = w[:, :in_ch]
# #             m.features.conv0 = new

# #         self.features = m.features
# #         feat_ch = m.classifier.in_features  # 最後特徵維度（DN161=2208）

# #         self.norm = nn.BatchNorm2d(feat_ch)
# #         self.relu = nn.ReLU(inplace=True)
# #         self.pool = nn.AdaptiveAvgPool2d(1)
# #         self.head = nn.Sequential(
# #             nn.Dropout(drop),
# #             nn.Linear(feat_ch, 256),
# #             nn.ReLU(inplace=True),
# #             nn.Dropout(drop),
# #             nn.Linear(256, num_classes)
# #         )

# #         # freeze_until：0=不凍；1≈凍到 denseblock2 以前（含 stem）
# #         if freeze_until > 0:
# #             for name, p in self.features.named_parameters():
# #                 p.requires_grad = False
# #                 if "denseblock2" in name:
# #                     break

# #     def forward(self, x):
# #         f = self.features(x)
# #         f = self.norm(f); f = self.relu(f)
# #         g = self.pool(f).flatten(1)
# #         return self.head(g)

# # # -------------------- Losses --------------------
# # class FocalLoss(nn.Module):
# #     def __init__(self, weight=None, gamma=2.0, alpha=None, reduction="mean", label_smoothing=0.0):
# #         super().__init__()
# #         self.gamma = gamma
# #         self.alpha = alpha
# #         self.reduction = reduction
# #         self.weight = weight
# #         self.eps = 1e-8
# #         self.label_smoothing = label_smoothing

# #     def forward(self, logits, target):
# #         num_classes = logits.size(1)
# #         log_probs = torch.log_softmax(logits, dim=1)
# #         probs = log_probs.exp()

# #         # label smoothing one-hot
# #         with torch.no_grad():
# #             true_dist = torch.zeros_like(logits)
# #             true_dist.fill_(self.label_smoothing / (num_classes - 1) if num_classes > 1 else 0.0)
# #             true_dist.scatter_(1, target.unsqueeze(1), 1.0 - self.label_smoothing)

# #         pt = (probs * true_dist).sum(dim=1).clamp(min=self.eps)
# #         focal = (1 - pt).pow(self.gamma)

# #         # alpha
# #         if self.alpha is not None:
# #             if isinstance(self.alpha, (list, tuple, torch.Tensor)):
# #                 alpha_vec = torch.as_tensor(self.alpha, device=logits.device, dtype=logits.dtype)
# #                 at = alpha_vec.gather(0, target)
# #             else:
# #                 at = torch.full_like(pt, float(self.alpha))
# #         else:
# #             at = torch.ones_like(pt)

# #         # class weight 乘在 target 類別
# #         if self.weight is not None:
# #             wt = self.weight.gather(0, target)
# #         else:
# #             wt = torch.ones_like(pt)

# #         loss = -at * wt * focal * (true_dist * log_probs).sum(dim=1)
# #         if self.reduction == "mean":
# #             return loss.mean()
# #         elif self.reduction == "sum":
# #             return loss.sum()
# #         return loss

# # # CB-Focal 權重（Effective Number of Samples）
# # def class_balanced_weights(counts, beta=0.9999, device=None):
# #     counts = np.array([max(1, c) for c in counts], dtype=np.float32)
# #     eff_num = 1.0 - np.power(beta, counts)
# #     weights = (1.0 - beta) / eff_num
# #     weights = weights / weights.sum() * len(counts)
# #     t = torch.tensor(weights, dtype=torch.float32)
# #     return t if device is None else t.to(device)

# # # -------------------- EMA --------------------
# # class EMA:
# #     def __init__(self, model, decay=0.999):
# #         self.decay = decay
# #         self.shadow = {}
# #         self.backup = {}
# #         for name, p in model.named_parameters():
# #             if p.requires_grad:
# #                 self.shadow[name] = p.data.clone()

# #     def update(self, model):
# #         with torch.no_grad():
# #             for name, p in model.named_parameters():
# #                 if p.requires_grad:
# #                     self.shadow[name].mul_(self.decay).add_(p.data, alpha=1.0 - self.decay)

# #     def apply_shadow(self, model):
# #         self.backup = {}
# #         for name, p in model.named_parameters():
# #             if name in self.shadow:
# #                 self.backup[name] = p.data.clone()
# #                 p.data.copy_(self.shadow[name])

# #     def restore(self, model):
# #         for name, p in model.named_parameters():
# #             if name in self.backup:
# #                 p.data.copy_(self.backup[name])
# #         self.backup = {}

# # # -------------------- misc utils --------------------
# # def _infer_device(user_choice=None):
# #     if user_choice is not None:
# #         user_choice = user_choice.lower()
# #         if user_choice == "cpu":
# #             return torch.device("cpu")
# #         if user_choice == "cuda" and torch.cuda.is_available():
# #             return torch.device("cuda")
# #     return torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # def _labels_of_items(items):
# #     # k=1 -> 0 ; k=2 -> 1 ; k∈{3,4} -> 2
# #     labels = []
# #     for it in items:
# #         k = int(it.get("k", 1))
# #         if k == 1: labels.append(0)
# #         elif k == 2: labels.append(1)
# #         else: labels.append(2)
# #     return labels

# # def _class_counts(ds, num_classes=NUM_CLASSES):
# #     ys = _labels_of_items(ds.items)
# #     cnt = [0]*num_classes
# #     for y in ys: cnt[y]+=1
# #     return cnt, ys

# # # 嚴格類別均衡的批次抽樣
# # class BalancedBatchSampler(Sampler):
# #     def __init__(self, labels, batch_size, num_classes=NUM_CLASSES):
# #         self.labels = np.array(labels, dtype=np.int64)
# #         self.batch_size = int(batch_size)
# #         self.num_classes = num_classes
# #         self.m = max(1, self.batch_size // self.num_classes)
# #         self.class_indices = {c: np.where(self.labels==c)[0].tolist() for c in range(num_classes)}
# #         for c in self.class_indices:
# #             if len(self.class_indices[c]) == 0:
# #                 raise ValueError(f"Class {c} has zero samples; cannot build balanced batches.")
# #             random.shuffle(self.class_indices[c])
# #         self.ptr = {c: 0 for c in range(num_classes)}
# #         self.num_batches = int(math.ceil(len(self.labels) / float(self.batch_size)))

# #     def __iter__(self):
# #         for _ in range(self.num_batches):
# #             batch = []
# #             for c in range(self.num_classes):
# #                 for _ in range(self.m):
# #                     idxs = self.class_indices[c]
# #                     p = self.ptr[c]
# #                     if p >= len(idxs):
# #                         random.shuffle(idxs)
# #                         self.ptr[c] = 0
# #                         p = 0
# #                     batch.append(idxs[p])
# #                     self.ptr[c] += 1
# #             yield batch[:self.batch_size]

# #     def __len__(self):
# #         return self.num_batches

# # # -------------------- simple GPU aug --------------------
# # def random_augment(x, p_flip=0.5, p_vflip=0.5, p_rot=0.5, p_bc=0.5, p_erase=0.25):
# #     # x: (B, C, H, W)；前 3/4 通道是影像，最後一個通道是 mask
# #     B, C, H, W = x.shape
# #     img_ch = min(3, C)
# #     img = x[:, :img_ch]

# #     if torch.rand(1).item() < p_flip:
# #         img = torch.flip(img, dims=[3]); x = torch.flip(x, dims=[3]); x[:, :img_ch] = img
# #     if torch.rand(1).item() < p_vflip:
# #         img = torch.flip(img, dims=[2]); x = torch.flip(x, dims=[2]); x[:, :img_ch] = img
# #     if torch.rand(1).item() < p_rot:
# #         k = torch.randint(0, 4, (1,)).item()
# #         img = torch.rot90(img, k, dims=[2,3]); x = torch.rot90(x, k, dims=[2,3]); x[:, :img_ch] = img
# #     if torch.rand(1).item() < p_bc:
# #         a = 0.8 + 0.4 * torch.rand(1, device=x.device); b = (torch.rand(1, device=x.device) - 0.5) * 0.2
# #         img = (img * a + b).clamp(0, 1); x[:, :img_ch] = img
# #     if torch.rand(1).item() < p_erase:
# #         h = int(H * (0.05 + 0.2*torch.rand(1).item())); w = int(W * (0.05 + 0.2*torch.rand(1).item()))
# #         y0 = torch.randint(0, max(1, H - h), (1,)).item(); x0 = torch.randint(0, max(1, W - w), (1,)).item()
# #         val = torch.rand((img_ch, 1, 1), device=x.device)
# #         x[:, :img_ch, y0:y0+h, x0:x0+w] = val
# #     return x

# # # -------------------- geometry channels from mask --------------------
# # def make_edge_from_mask(mask):
# #     dil = F.max_pool2d(mask, kernel_size=3, stride=1, padding=1)
# #     ero = -F.max_pool2d(-mask, kernel_size=3, stride=1, padding=1)
# #     edge = (dil - ero).clamp_(0, 1)
# #     edge = (edge > 0).float()
# #     edge = (edge - (mask>0).float()).abs().clamp(0,1)
# #     return edge

# # @torch.no_grad()
# # def make_pseudodist_from_mask(mask, iters=4):
# #     x = mask.float(); acc = torch.zeros_like(x); cur = x
# #     for _ in range(iters):
# #         cur = F.avg_pool2d(cur, kernel_size=3, stride=1, padding=1)
# #         acc = acc + cur
# #     acc = acc / acc.amax(dim=[2,3], keepdim=True).clamp(min=1e-6)
# #     return acc

# # def add_geom_channels(x, rgb_mode=True, use_edge=True, use_dist=False):
# #     mask = x[:, -1:, ...]
# #     extra = []
# #     if use_edge: extra.append(make_edge_from_mask(mask))
# #     if use_dist: extra.append(make_pseudodist_from_mask(mask))
# #     if len(extra): x = torch.cat([x, *extra], dim=1)
# #     return x

# # # -------------------- train --------------------
# # def main():
# #     ap = argparse.ArgumentParser()
# #     ap.add_argument("--merged_root", required=True)
# #     ap.add_argument("--save_dir", required=True)
# #     ap.add_argument("--img_size", type=int, default=224)
# #     ap.add_argument("--rgb", type=int, default=1)               # 1->RGB+mask, 0->Gray+mask
# #     ap.add_argument("--bs", type=int, default=32)
# #     ap.add_argument("--epochs", type=int, default=60)
# #     ap.add_argument("--lr", type=float, default=3e-4)
# #     ap.add_argument("--seed", type=int, default=2025)
# #     ap.add_argument("--workers", type=int, default=8)
# #     ap.add_argument("--device", type=str, default=None)
# #     ap.add_argument("--amp", type=int, default=1)
# #     ap.add_argument("--class_weight", type=str, default="auto") # none/auto
# #     ap.add_argument("--balance_sampler", type=int, default=1)
# #     ap.add_argument("--balanced_batch", type=int, default=1)
# #     ap.add_argument("--freeze_until", type=int, default=0)
# #     ap.add_argument("--label_smoothing", type=float, default=0.05)
# #     ap.add_argument("--loss", type=str, default="cb_focal", choices=["ce","focal","cb_focal"])
# #     ap.add_argument("--gamma", type=float, default=2.0)
# #     ap.add_argument("--alpha", type=float, default=None)
# #     ap.add_argument("--early_stop", type=int, default=50)
# #     ap.add_argument("--ema", type=int, default=1)
# #     ap.add_argument("--ema_decay", type=float, default=0.999)
# #     ap.add_argument("--aug", type=int, default=1)
# #     ap.add_argument("--tta_val", type=int, default=1)           # 1:4-TTA, 2:8-TTA
# #     ap.add_argument("--geom_edge", type=int, default=1)
# #     ap.add_argument("--geom_dist", type=int, default=1)         # 開啟 pseudo-distance
# #     ap.add_argument("--accum_steps", type=int, default=1)       # 梯度累積
# #     ap.add_argument("--la_tau", type=float, default=1.0,   help="Logit Adjustment 強度；0 關閉")
# #     ap.add_argument("--delay_reweight_epochs", type=int, default=8, help="前 N 個 epoch 先用 CE（不做 reweight）")

# #     args = ap.parse_args()

# #     set_seed(args.seed)
# #     os.makedirs(args.save_dir, exist_ok=True)
# #     device = _infer_device(args.device)
# #     use_cuda = device.type == "cuda"
# #     if use_cuda:
# #         torch.backends.cudnn.benchmark = True

# #     # ---- datasets ----
# #     ds_tr = SingleCellFromMerged(args.merged_root, split="train", img_size=args.img_size, use_rgb=bool(args.rgb))
# #     ds_va = SingleCellFromMerged(args.merged_root, split="val",   img_size=args.img_size, use_rgb=bool(args.rgb))

# #     # base 通道數 + 幾何通道
# #     base_in = 4 if args.rgb else 2  # (RGB or Gray) + mask
# #     added = (1 if args.geom_edge==1 else 0) + (1 if args.geom_dist==1 else 0)
# #     in_ch = base_in + added

# #     model = ViTCellClassifier(in_ch=in_ch, num_classes=NUM_CLASSES, pretrained=True, freeze_until=args.freeze_until).to(device)

# #     model = model.to(memory_format=torch.channels_last)

# #     # ---- loss ----
# #     counts, ys_tr = _class_counts(ds_tr, NUM_CLASSES)
# #     if args.class_weight == "auto":
# #         cls_w = torch.tensor([sum(counts)/max(1,c) for c in counts], dtype=torch.float32, device=device)
# #         cls_w = cls_w / (cls_w.mean().clamp(min=1e-6))
# #     else:
# #         cls_w = None

# #     if args.loss == "cb_focal":
# #         cb_w = class_balanced_weights(counts, beta=0.9999, device=device)
# #         criterion = FocalLoss(weight=cb_w, gamma=args.gamma, alpha=args.alpha, label_smoothing=0.0)
# #     elif args.loss == "focal":
# #         criterion = FocalLoss(weight=cls_w, gamma=args.gamma, alpha=args.alpha, label_smoothing=args.label_smoothing)
# #     else:
# #         criterion = nn.CrossEntropyLoss(weight=cls_w, label_smoothing=args.label_smoothing)

# #     # ---- dataloaders ----
# #     loader_kwargs = dict(num_workers=max(0, args.workers), pin_memory=use_cuda)
# #     if args.workers > 0:
# #         loader_kwargs.update(persistent_workers=True, prefetch_factor=4)

# #     if args.balanced_batch == 1:
# #         batch_sampler = BalancedBatchSampler(ys_tr, batch_size=args.bs, num_classes=NUM_CLASSES)
# #         dl_tr = DataLoader(ds_tr, batch_sampler=batch_sampler, **loader_kwargs)
# #     else:
# #         dl_tr = DataLoader(ds_tr, batch_size=args.bs, shuffle=True, **loader_kwargs)
# #     dl_va = DataLoader(ds_va, batch_size=args.bs, shuffle=False, **loader_kwargs)

# #     # ---- optimizer（差分學習率：head ×5）----
# #     base_lr = args.lr
# #     decay, nodecay, head_decay, head_nodecay = [], [], [], []
# #     for n,p in model.named_parameters():
# #         if not p.requires_grad: 
# #             continue
# #         is_head = ("head" in n)
# #         is_nodecay = (p.dim()==1) or n.endswith(".bias") or ("bn" in n) or ("norm" in n)
# #         if is_head:
# #             (head_nodecay if is_nodecay else head_decay).append(p)
# #         else:
# #             (nodecay if is_nodecay else decay).append(p)
# #     param_groups = [
# #         {"params": decay,        "lr": base_lr,     "weight_decay": 1e-4},
# #         {"params": nodecay,      "lr": base_lr,     "weight_decay": 0.0},
# #         {"params": head_decay,   "lr": base_lr*5.0, "weight_decay": 1e-4},
# #         {"params": head_nodecay, "lr": base_lr*5.0, "weight_decay": 0.0},
# #     ]
# #     opt = optim.AdamW(param_groups)

# #     # ---- Cosine with warmup（以「optimizer step」為單位）----
# #     steps_per_epoch = len(dl_tr)
# #     accum = max(1, args.accum_steps)
# #     opt_steps_per_epoch = math.ceil(steps_per_epoch / accum)
# #     total_steps = opt_steps_per_epoch * args.epochs
# #     warmup = max(1, int(0.10 * total_steps))  # 10% warmup

# #     def lr_lambda(step):
# #         if step < warmup:
# #             return step / warmup
# #         progress = (step - warmup) / max(1,(total_steps - warmup))
# #         return 0.5 * (1 + math.cos(math.pi * progress))
# #     sched = optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_lambda)

# #     scaler = torch.amp.GradScaler('cuda', enabled=(args.amp==1 and use_cuda))
# #     ema = EMA(model, decay=args.ema_decay) if args.ema==1 else None

# #     best_f1, best_ep = -1.0, -1
# #     no_improve = 0
# #     global_step = 0  # optimizer steps

# #     # 小工具：將 xb 加上幾何通道 & 正規化 & 截取 in_ch
# #     def prep_input(xb):
# #         xb = add_geom_channels(xb, rgb_mode=bool(args.rgb), use_edge=bool(args.geom_edge), use_dist=bool(args.geom_dist))
# #         xb = xb[:, :in_ch, ...]
# #         # ImageNet normalization for first 3 channels（RGB）
# #         if args.rgb == 1 and xb.size(1) >= 3:
# #             mean = torch.tensor([0.485, 0.456, 0.406], device=xb.device, dtype=xb.dtype).view(1,3,1,1)
# #             std  = torch.tensor([0.229, 0.224, 0.225], device=xb.device, dtype=xb.dtype).view(1,3,1,1)
# #             xb[:, :3] = (xb[:, :3] - mean) / std
# #         return xb

# #     print(f"[INFO] steps_per_epoch={steps_per_epoch}, opt_steps_per_epoch={opt_steps_per_epoch}, total_steps={total_steps}, warmup={warmup}")

# #     for ep in range(1, args.epochs+1):
# #         model.train()
# #         run_loss = 0.0
# #         t0 = time.time()
# #         opt.zero_grad(set_to_none=True)

# #         for bi, (xb, yb) in enumerate(dl_tr, 1):
# #             xb = xb.to(device, non_blocking=use_cuda).to(memory_format=torch.channels_last)
# #             yb = yb.to(device, non_blocking=use_cuda)

# #             xb = prep_input(xb)
# #             if args.aug == 1:
# #                 xb = random_augment(xb)

# #             with torch.autocast('cuda', enabled=(args.amp==1 and use_cuda)):
# #                 logits = model(xb)
# #                 loss = criterion(logits, yb) / accum

# #             scaler.scale(loss).backward()

# #             did_step = False
# #             if bi % accum == 0:
# #                 # 先反縮放再裁剪梯度，避免 AMP 導致裁剪無效
# #                 scaler.unscale_(opt)
# #                 torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# #                 scaler.step(opt)
# #                 scaler.update()
# #                 opt.zero_grad(set_to_none=True)

# #                 if ema is not None:
# #                     ema.update(model)

# #                 sched.step()
# #                 global_step += 1
# #                 did_step = True

# #             run_loss += loss.item() * accum  # 還原原始 loss

# #             if bi % 200 == 0:
# #                 cur_lr = sched.get_last_lr()[0]
# #                 print(f"[E{ep:02d}] {bi}/{steps_per_epoch} loss={run_loss/bi:.4f} lr={cur_lr:.2e} gs={global_step}")

# #         # 如果這個 epoch 最後沒剛好整除 accum，補一次 optimizer step
# #         if (steps_per_epoch % accum) != 0:
# #             scaler.unscale_(opt)
# #             torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
# #             scaler.step(opt); scaler.update()
# #             opt.zero_grad(set_to_none=True)
# #             if ema is not None: ema.update(model)
# #             sched.step(); global_step += 1

# #         # -------- eval（可使用 EMA 權重 + TTA） --------
# #         def eval_with(model_for_eval):
# #             model_for_eval.eval()
# #             y_true, y_pred = [], []
# #             with torch.no_grad():
# #                 for xb, yb in dl_va:
# #                     xb = xb.to(device, non_blocking=use_cuda).to(memory_format=torch.channels_last)
# #                     yb = yb.to(device, non_blocking=use_cuda)

# #                     def forward_once(x):
# #                         x = prep_input(x)
# #                         return model_for_eval(x)

# #                     x0 = xb
# #                     if args.tta_val == 2:
# #                         # 8-TTA：rot(0/90/180/270) × {id, hflip}
# #                         logits_list = []
# #                         for k in [0,1,2,3]:
# #                             xk = torch.rot90(x0, k, dims=[2,3])
# #                             logits_list.append(forward_once(xk))
# #                             logits_list.append(forward_once(torch.flip(xk, dims=[3])))
# #                         logits = torch.stack(logits_list, dim=0).mean(0)
# #                     elif args.tta_val == 1:
# #                         # 4-TTA：id, hflip, vflip, hv
# #                         logits_list = []
# #                         logits_list.append(forward_once(x0))
# #                         logits_list.append(forward_once(torch.flip(x0, dims=[3])))
# #                         logits_list.append(forward_once(torch.flip(x0, dims=[2])))
# #                         logits_list.append(forward_once(torch.flip(torch.flip(x0, dims=[2]), dims=[3])))
# #                         logits = torch.stack(logits_list, dim=0).mean(0)
# #                     else:
# #                         logits = forward_once(x0)

# #                     pred = logits.argmax(1)
# #                     y_true += yb.tolist(); y_pred += pred.tolist()

# #             cm = confusion_matrix(y_true, y_pred, num_classes=NUM_CLASSES)
# #             macro_f1, per_cls = macro_f1_from_cm(cm)
# #             return macro_f1, per_cls, cm

# #         if ema is not None:
# #             ema.apply_shadow(model)
# #             macro_f1, per_cls, cm = eval_with(model)
# #             ema.restore(model)
# #         else:
# #             macro_f1, per_cls, cm = eval_with(model)

# #         print(f"[VAL] E{ep:02d} macroF1={macro_f1:.3f} per_class={[f'{x:.3f}' for x in per_cls]}")
# #         print("CM rows=gt cols=pred:")
# #         for r in cm: print(r.tolist())

# #         improved = macro_f1 > best_f1 + 1e-6
# #         if improved:
# #             best_f1, best_ep = macro_f1, ep
# #             no_improve = 0
# #             save_p = os.path.join(args.save_dir, f"best_densenet_cell_cls_3cls.pt")

# #     # ←← 這一行是關鍵：以完整 state_dict 為底，覆蓋 EMA 參數（若有）
# #             state_to_save = build_ema_full_state_dict(model, ema)

# #             payload = {
# #                 "state_dict": state_to_save,
# #                 "best_f1": best_f1,
# #                 "epoch": ep,
# #                 "ema": (ema is not None),
# #                 "in_ch": in_ch,
# #                 "num_classes": NUM_CLASSES,
# #                 "args": vars(args),
# #             }
# #             torch.save(payload, save_p)
# #             print(f"[SAVE] best at epoch {ep} (macroF1={best_f1:.3f})")
# #         else:
# #             no_improve += 1

# #         dt = time.time() - t0
# #         print(f"[E{ep:02d}] epoch_time={dt/60:.1f} min  (no_improve={no_improve}/{args.early_stop})")

# #         if args.early_stop > 0 and no_improve >= args.early_stop:
# #             print("[EARLY STOP] no improvement, stop training.")
# #             break

# #     print(f"[DONE] best macroF1={best_f1:.3f} @ epoch {best_ep}")

# # if __name__ == "__main__":
# #     main()
# import os, math, time, argparse, random
# # train_vit.py
# import os, argparse, math, time, random
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import torch.nn.functional as F
# from torch.utils.data import DataLoader, Sampler

# from data_from_merged_2 import SingleCellFromMerged
# from metrics import confusion_matrix, macro_f1_from_cm
# from utils import set_seed

# # ======== Backbones ========
# # We keep the old ones for args.arch comparison, but add ViT
# from torchvision.models import densenet161, vit_b_16, ViT_B_16_Weights
# try:
#     from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
#     _HAS_CNX_WEIGHTS = True
# except Exception:
#     from torchvision.models import convnext_tiny
#     _HAS_CNX_WEIGHTS = False

# NUM_CLASSES = 3  # 0:1to1, 1:1to2, 2:1to4(含k=3或4)

# # ---------------- EMA ----------------
# class EMA:
#     def __init__(self, model, decay=0.999):
#         self.decay = decay
#         self.shadow = {}
#         self.backup = {}
#         for n, p in model.named_parameters():
#             if p.requires_grad:
#                 self.shadow[n] = p.data.clone()

#     @torch.no_grad()
#     def update(self, model):
#         for n, p in model.named_parameters():
#             if p.requires_grad:
#                 self.shadow[n].mul_(self.decay).add_(p.data, alpha=1.0 - self.decay)

#     @torch.no_grad()
#     def apply_shadow(self, model):
#         self.backup = {}
#         for n, p in model.named_parameters():
#             if n in self.shadow:
#                 self.backup[n] = p.data.clone()
#                 p.data.copy_(self.shadow[n])

#     @torch.no_grad()
#     def restore(self, model):
#         for n, p in model.named_parameters():
#             if n in self.backup:
#                 p.data.copy_(self.backup[n])
#         self.backup = {}

# def build_ema_full_state_dict(model: nn.Module, ema: EMA):
#     sd = model.state_dict()
#     if ema is None:
#         return sd
#     with torch.no_grad():
#         for name, _ in model.named_parameters():
#             if name in ema.shadow:
#                 sd[name] = ema.shadow[name].detach().clone()
#     return sd

# # ---------------- Losses ----------------
# class FocalLoss(nn.Module):
#     def __init__(self, weight=None, gamma=2.0, alpha=None, reduction="mean", label_smoothing=0.0):
#         super().__init__()
#         self.gamma = gamma
#         self.alpha = alpha
#         self.reduction = reduction
#         self.weight = weight
#         self.eps = 1e-8
#         self.label_smoothing = label_smoothing

#     def forward(self, logits, target):
#         num_classes = logits.size(1)
#         log_probs = torch.log_softmax(logits, dim=1)
#         probs = log_probs.exp()

#         with torch.no_grad():
#             true_dist = torch.zeros_like(logits)
#             true_dist.fill_(self.label_smoothing / (num_classes - 1) if num_classes > 1 else 0.0)
#             true_dist.scatter_(1, target.unsqueeze(1), 1.0 - self.label_smoothing)

#         pt = (probs * true_dist).sum(dim=1).clamp(min=self.eps)
#         focal = (1 - pt).pow(self.gamma)

#         if self.alpha is not None:
#             if isinstance(self.alpha, (list, tuple, torch.Tensor)):
#                 alpha_vec = torch.as_tensor(self.alpha, device=logits.device, dtype=logits.dtype)
#                 at = alpha_vec.gather(0, target)
#             else:
#                 at = torch.full_like(pt, float(self.alpha))
#         else:
#             at = torch.ones_like(pt)

#         if self.weight is not None:
#             wt = self.weight.gather(0, target)
#         else:
#             wt = torch.ones_like(pt)

#         loss = -at * wt * focal * (true_dist * log_probs).sum(dim=1)
#         return loss.mean() if self.reduction == "mean" else (loss.sum() if self.reduction == "sum" else loss)

# def class_balanced_weights(counts, beta=0.9999, device=None):
#     counts = np.array([max(1, c) for c in counts], dtype=np.float32)
#     eff = 1.0 - np.power(beta, counts)
#     w = (1.0 - beta) / eff
#     w = w / w.sum() * len(counts)
#     t = torch.tensor(w, dtype=torch.float32)
#     return t if device is None else t.to(device)

# # ---------------- Sampler ----------------
# class BalancedBatchSampler(Sampler):
#     def __init__(self, labels, batch_size, num_classes=NUM_CLASSES):
#         self.labels = np.array(labels, dtype=np.int64)
#         self.batch_size = int(batch_size)
#         self.num_classes = num_classes
#         self.m = max(1, self.batch_size // self.num_classes)
#         self.class_indices = {c: np.where(self.labels == c)[0].tolist() for c in range(num_classes)}
#         for c in self.class_indices:
#             if len(self.class_indices[c]) == 0:
#                 raise ValueError(f"Class {c} has zero samples.")
#             random.shuffle(self.class_indices[c])
#         self.ptr = {c: 0 for c in range(num_classes)}
#         self.num_batches = int(math.ceil(len(self.labels) / float(self.batch_size)))

#     def __iter__(self):
#         for _ in range(self.num_batches):
#             batch = []
#             for c in range(self.num_classes):
#                 for _ in range(self.m):
#                     idxs = self.class_indices[c]
#                     p = self.ptr[c]
#                     if p >= len(idxs):
#                         random.shuffle(idxs)
#                         self.ptr[c] = 0
#                         p = 0
#                     batch.append(idxs[p])
#                     self.ptr[c] += 1
#             yield batch[:self.batch_size]

#     def __len__(self):
#         return self.num_batches

# # ---------------- GPU aug & geom ----------------
# def random_augment(x, p_flip=0.5, p_vflip=0.5, p_rot=0.5, p_bc=0.5, p_erase=0.25):
#     B, C, H, W = x.shape
#     img_ch = min(3, C)
#     img = x[:, :img_ch]

#     if torch.rand(1).item() < p_flip:
#         img = torch.flip(img, dims=[3]); x = torch.flip(x, dims=[3]); x[:, :img_ch] = img
#     if torch.rand(1).item() < p_vflip:
#         img = torch.flip(img, dims=[2]); x = torch.flip(x, dims=[2]); x[:, :img_ch] = img
#     if torch.rand(1).item() < p_rot:
#         k = torch.randint(0, 4, (1,)).item()
#         img = torch.rot90(img, k, dims=[2,3]); x = torch.rot90(x, k, dims=[2,3]); x[:, :img_ch] = img
#     if torch.rand(1).item() < p_bc:
#         a = 0.8 + 0.4 * torch.rand(1, device=x.device); b = (torch.rand(1, device=x.device) - 0.5) * 0.2
#         img = (img * a + b).clamp(0, 1); x[:, :img_ch] = img
#     if torch.rand(1).item() < p_erase:
#         h = int(H * (0.05 + 0.2*torch.rand(1).item())); w = int(W * (0.05 + 0.2*torch.rand(1).item()))
#         y0 = torch.randint(0, max(1, H - h), (1,)).item(); x0 = torch.randint(0, max(1, W - w), (1,)).item()
#         val = torch.rand((img_ch, 1, 1), device=x.device)
#         x[:, :img_ch, y0:y0+h, x0:x0+w] = val
#     return x

# def make_edge_from_mask(mask):
#     dil = F.max_pool2d(mask, kernel_size=3, stride=1, padding=1)
#     ero = -F.max_pool2d(-mask, kernel_size=3, stride=1, padding=1)
#     edge = (dil - ero).clamp_(0, 1)
#     edge = (edge > 0).float()
#     edge = (edge - (mask > 0).float()).abs().clamp(0, 1)
#     return edge

# @torch.no_grad()
# def make_pseudodist_from_mask(mask, iters=4):
#     x = mask.float(); acc = torch.zeros_like(x); cur = x
#     for _ in range(iters):
#         cur = F.avg_pool2d(cur, kernel_size=3, stride=1, padding=1)
#         acc = acc + cur
#     acc = acc / acc.amax(dim=[2,3], keepdim=True).clamp(min=1e-6)
#     return acc

# def add_geom_channels(x, use_edge=True, use_dist=False):
#     # x 的最後一個通道必須是 mask
#     mask = x[:, -1:, ...]
#     extra = []
#     if use_edge: extra.append(make_edge_from_mask(mask))
#     if use_dist: extra.append(make_pseudodist_from_mask(mask))
#     if extra:
#         x = torch.cat([x, *extra], dim=1)
#     return x

# # ---------------- Backbones: ViT / DenseNet / ConvNeXt ----------------
# class ViTCellClassifier(nn.Module):
#     def __init__(self, in_ch=4, num_classes=NUM_CLASSES, pretrained=True, freeze_until=0, drop=0.2, img_size=224):
#         """
#         ViT Classifier.
#         freeze_until (int):
#             0: train all
#             1: freeze patch_embed, class_token, pos_embed
#             2: freeze ... + first 6 encoder blocks
#             3: freeze ... + all 12 encoder blocks (only train head)
#         """
#         super().__init__()
#         m = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None, image_size=img_size)

#         # 1. Modify the patch embedding layer (conv_proj)
#         stem_conv = m.conv_proj
#         if in_ch != stem_conv.in_channels:
#             new = nn.Conv2d(in_ch, stem_conv.out_channels,
#                             kernel_size=stem_conv.kernel_size, stride=stem_conv.stride,
#                             padding=stem_conv.padding, bias=(stem_conv.bias is not None))
#             with torch.no_grad():
#                 w = stem_conv.weight
#                 if in_ch >= 3:
#                     new.weight[:, :3] = w # Copy RGB weights
#                     if in_ch > 3:
#                         # Use mean of RGB weights for extra channels
#                         mean_extra = w[:, :3].mean(dim=1, keepdim=True)
#                         for c in range(3, in_ch):
#                             new.weight[:, c:c+1] = mean_extra
#                 else:
#                     new.weight[:, :in_ch] = w[:, :in_ch] # Copy R or RG
#                 if stem_conv.bias is not None:
#                     new.bias.copy_(stem_conv.bias)
#             m.conv_proj = new # Replace the layer

#         # 2. Store the feature extractor parts
#         self.conv_proj = m.conv_proj
#         self.class_token = m.class_token
#         self.pos_embedding = m.encoder.pos_embedding
#         self.encoder_layers = m.encoder.layers
#         self.norm = m.encoder.ln
#         self.dropout = m.encoder.dropout

#         # 3. Store the feature dimension
#         feat_ch = m.hidden_dim # 768 for ViT-B

#         # 4. Create the new head (same as the other models for optimizer compatibility)
#         self.head = nn.Sequential(
#             nn.Dropout(drop),
#             nn.Linear(feat_ch, 256),
#             nn.ReLU(inplace=True),
#             nn.Dropout(drop),
#             nn.Linear(256, num_classes),
#         )

#         # 5. Handle freezing
#         if freeze_until >= 1:
#             for p in self.conv_proj.parameters(): p.requires_grad = False
#             self.class_token.requires_grad = False
#             self.pos_embedding.requires_grad = False
#         if freeze_until >= 2:
#             num_to_freeze = 6 if freeze_until == 2 else 12 # ViT-B has 12 blocks
#             for i, layer in enumerate(self.encoder_layers):
#                 if i < num_to_freeze:
#                     for p in layer.parameters():
#                         p.requires_grad = False

#     def forward(self, x):
#         # 1. Patch Embedding
#         x = self.conv_proj(x)
#         # 2. Reshape and transpose
#         x = x.flatten(2).transpose(1, 2)  # (B, N, E)
#         # 3. Prepend [CLS] token
#         x = torch.cat((self.class_token.expand(x.shape[0], -1, -1), x), dim=1) # (B, N+1, E)
#         # 4. Add positional embedding
#         x = x + self.pos_embedding
#         x = self.dropout(x)
#         # 5. Pass through encoder blocks
#         x = self.encoder_layers(x)
#         # 6. Final norm
#         x = self.norm(x)
#         # 7. Select [CLS] token output
#         g = x[:, 0]
#         # 8. Pass through classifier head
#         return self.head(g)

# class DenseNetCellClassifier(nn.Module):
#     def __init__(self, in_ch=4, num_classes=NUM_CLASSES, pretrained=True, freeze_until=0, drop=0.2):
#         super().__init__()
#         m = densenet161(weights="IMAGENET1K_V1" if pretrained else None)
#         conv0 = m.features.conv0
#         if in_ch != 3:
#             new = nn.Conv2d(in_ch, conv0.out_channels, kernel_size=conv0.kernel_size,
#                             stride=conv0.stride, padding=conv0.padding, bias=False)
#             with torch.no_grad():
#                 w = conv0.weight
#                 if in_ch >= 3:
#                     new.weight[:, :3] = w
#                     if in_ch > 3:
#                         mean_extra = w.mean(dim=1, keepdim=True)
#                         for i in range(3, in_ch):
#                             new.weight[:, i:i+1] = mean_extra
#                 else:
#                     new.weight[:, :in_ch] = w[:, :in_ch]
#             m.features.conv0 = new

#         self.features = m.features
#         feat_ch = m.classifier.in_features
#         self.norm = nn.BatchNorm2d(feat_ch)
#         self.relu = nn.ReLU(inplace=True)
#         self.pool = nn.AdaptiveAvgPool2d(1)
#         self.head = nn.Sequential(
#             nn.Dropout(drop),
#             nn.Linear(feat_ch, 256),
#             nn.ReLU(inplace=True),
#             nn.Dropout(drop),
#             nn.Linear(256, num_classes),
#         )

#         if freeze_until > 0:
#             for name, p in self.features.named_parameters():
#                 p.requires_grad = False
#                 if "denseblock2" in name:
#                     break

#     def forward(self, x):
#         f = self.features(x)
#         f = self.norm(f); f = self.relu(f)
#         g = self.pool(f).flatten(1)
#         return self.head(g)

# class ConvNeXtCellClassifier(nn.Module):
#     def __init__(self, in_ch=4, num_classes=NUM_CLASSES, pretrained=True, freeze_until=0, drop=0.2):
#         super().__init__()
#         if _HAS_CNX_WEIGHTS and pretrained:
#             m = convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
#         else:
#             m = convnext_tiny(weights=("IMAGENET1K_V1" if pretrained else None))

#         stem_conv = m.features[0][0]
#         if in_ch != stem_conv.in_channels:
#             new = nn.Conv2d(in_ch, stem_conv.out_channels,
#                             kernel_size=stem_conv.kernel_size, stride=stem_conv.stride,
#                             padding=stem_conv.padding, bias=(stem_conv.bias is not None))
#             with torch.no_grad():
#                 w = stem_conv.weight
#                 if in_ch >= 3:
#                     new.weight[:, :3] = w
#                     if in_ch > 3:
#                         mean_extra = w[:, :3].mean(dim=1, keepdim=True)
#                         for c in range(3, in_ch):
#                             new.weight[:, c:c+1] = mean_extra
#                 else:
#                     new.weight[:, :in_ch] = w[:, :in_ch]
#                 if stem_conv.bias is not None:
#                     new.bias.copy_(stem_conv.bias)
#             m.features[0][0] = new

#         self.features = m.features
#         feat_ch = m.classifier[2].in_features if isinstance(m.classifier[-1], nn.Linear) else 768
#         self.pool = nn.AdaptiveAvgPool2d(1)
#         self.norm = nn.LayerNorm(feat_ch, eps=1e-6)
#         self.head = nn.Sequential(
#             nn.Dropout(drop),
#             nn.Linear(feat_ch, 256),
#             nn.ReLU(inplace=True),
#             nn.Dropout(drop),
#             nn.Linear(256, num_classes),
#         )

#         if freeze_until > 0:
#             for p in self.features.parameters():
#                 p.requires_grad = False
#             # 若想只凍到前兩個 stage，可在此細分 stage index 解凍後段

#     def forward(self, x):
#         f = self.features(x)
#         g = self.pool(f).flatten(1)
#         g = self.norm(g)
#         return self.head(g)

# # ---------------- misc ----------------
# def _infer_device(user_choice=None):
#     if user_choice is not None:
#         user_choice = user_choice.lower()
#         if user_choice == "cpu":
#             return torch.device("cpu")
#         if user_choice == "cuda" and torch.cuda.is_available():
#             return torch.device("cuda")
#     return torch.device("cuda" if torch.cuda.is_available() else "cpu")

# def _labels_of_items(items):
#     labels = []
#     for it in items:
#         k = int(it.get("k", 1))
#         labels.append(0 if k == 1 else (1 if k == 2 else 2))
#     return labels

# def _class_counts(ds, num_classes=NUM_CLASSES):
#     ys = _labels_of_items(ds.items)
#     cnt = [0]*num_classes
#     for y in ys: cnt[y]+=1
#     return cnt, ys

# # ---------------- train ----------------
# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--merged_root", required=True)
#     ap.add_argument("--save_dir", required=True)
#     # MODIFIED: Added vit_b_16 and made it the default
#     ap.add_argument("--arch", type=str, default="vit_b_16", choices=["vit_b_16", "convnext_tiny","densenet161"])
#     ap.add_argument("--img_size", type=int, default=224)
#     ap.add_argument("--rgb", type=int, default=1)               # 1: RGB, 0: Gray
#     ap.add_argument("--include_mask", type=int, default=1)      # ★ 是否把 mask 拼成通道
#     ap.add_argument("--bs", type=int, default=32)
#     ap.add_argument("--epochs", type=int, default=60)
#     ap.add_argument("--lr", type=float, default=3e-4)
#     ap.add_argument("--seed", type=int, default=2025)
#     ap.add_argument("--workers", type=int, default=8)
#     ap.add_argument("--device", type=str, default=None)
#     ap.add_argument("--amp", type=int, default=1)
#     ap.add_argument("--class_weight", type=str, default="auto") # none/auto
#     ap.add_argument("--balanced_batch", type=int, default=1)
#     ap.add_argument("--freeze_until", type=int, default=0, help="0:all, 1:freeze patch/pos/cls, 2:..+6blocks, 3:..+12blocks (ViT)")
#     ap.add_argument("--label_smoothing", type=float, default=0.05)
#     ap.add_argument("--loss", type=str, default="cb_focal", choices=["ce","focal","cb_focal"])
#     ap.add_argument("--gamma", type=float, default=2.0)
#     ap.add_argument("--alpha", type=float, default=None)
#     ap.add_argument("--early_stop", type=int, default=50)
#     ap.add_argument("--ema", type=int, default=1)
#     ap.add_argument("--ema_decay", type=float, default=0.999)
#     ap.add_argument("--aug", type=int, default=1)
#     ap.add_argument("--tta_val", type=int, default=1)           # 0:off, 1:4-TTA, 2:8-TTA
#     ap.add_argument("--geom_edge", type=int, default=1)
#     ap.add_argument("--geom_dist", type=int, default=1)         # pseudo-distance
#     ap.add_argument("--accum_steps", type=int, default=1)

#     args = ap.parse_args()

#     # 若開啟幾何通道，強制需要 mask 通道作為來源
#     if (args.geom_edge == 1 or args.geom_dist == 1) and args.include_mask == 0:
#         print("[WARN] geom_edge/dist 需要 mask 通道；自動將 --include_mask 設為 1")
#         args.include_mask = 1

#     set_seed(args.seed)
#     os.makedirs(args.save_dir, exist_ok=True)
#     device = _infer_device(args.device)
#     use_cuda = (device.type == "cuda")
#     if use_cuda:
#         torch.backends.cudnn.benchmark = True

#     # ---- datasets ----
#     ds_tr = SingleCellFromMerged(
#         args.merged_root, split="train",
#         img_size=args.img_size, use_rgb=bool(args.rgb),
#         include_mask_channel=bool(args.include_mask)
#     )
#     ds_va = SingleCellFromMerged(
#         args.merged_root, split="val",
#         img_size=args.img_size, use_rgb=bool(args.rgb),
#         include_mask_channel=bool(args.include_mask)
#     )

#     # 計算輸入通道數：RGB/Gray + mask + 幾何
#     base_in = (3 if args.rgb == 1 else 1) + (1 if args.include_mask == 1 else 0)
#     added = (1 if args.geom_edge == 1 else 0) + (1 if args.geom_dist == 1 else 0)
#     in_ch = base_in + added

#     # ---- model ----
#     # MODIFIED: Added ViT logic
#     if args.arch == "vit_b_16":
#         model = ViTCellClassifier(in_ch=in_ch, num_classes=NUM_CLASSES,
#                                   pretrained=True, freeze_until=args.freeze_until,
#                                   img_size=args.img_size) # Pass img_size to ViT
#     elif args.arch == "convnext_tiny":
#         model = ConvNeXtCellClassifier(in_ch=in_ch, num_classes=NUM_CLASSES,
#                                        pretrained=True, freeze_until=args.freeze_until)
#     else:
#         model = DenseNetCellClassifier(in_ch=in_ch, num_classes=NUM_CLASSES,
#                                        pretrained=True, freeze_until=args.freeze_until)
#     model = model.to(device).to(memory_format=torch.channels_last) # channels_last is fine

#     # ---- loss ----
#     counts, ys_tr = _class_counts(ds_tr, NUM_CLASSES)
#     if args.class_weight == "auto":
#         cls_w = torch.tensor([sum(counts)/max(1, c) for c in counts], dtype=torch.float32, device=device)
#         cls_w = cls_w / (cls_w.mean().clamp(min=1e-6))
#     else:
#         cls_w = None

#     if args.loss == "cb_focal":
#         cb_w = class_balanced_weights(counts, beta=0.9999, device=device)
#         criterion = FocalLoss(weight=cb_w, gamma=args.gamma, alpha=args.alpha, label_smoothing=0.0)
#     elif args.loss == "focal":
#         criterion = FocalLoss(weight=cls_w, gamma=args.gamma, alpha=args.alpha, label_smoothing=args.label_smoothing)
#     else:
#         criterion = nn.CrossEntropyLoss(weight=cls_w, label_smoothing=args.label_smoothing)

#     # ---- dataloaders ----
#     loader_kwargs = dict(num_workers=max(0, args.workers), pin_memory=use_cuda)
#     if args.workers > 0:
#         loader_kwargs.update(persistent_workers=True, prefetch_factor=4)

#     if args.balanced_batch == 1:
#         batch_sampler = BalancedBatchSampler(ys_tr, batch_size=args.bs, num_classes=NUM_CLASSES)
#         dl_tr = DataLoader(ds_tr, batch_sampler=batch_sampler, **loader_kwargs)
#     else:
#         dl_tr = DataLoader(ds_tr, batch_size=args.bs, shuffle=True, **loader_kwargs)
#     dl_va = DataLoader(ds_va, batch_size=args.bs, shuffle=False, **loader_kwargs)

#     # ---- optimizer（head ×5 LR）----
#     # This logic works for ViT as long as the new head is named `self.head`
#     base_lr = args.lr
#     decay, nodecay, head_decay, head_nodecay = [], [], [], []
#     for n, p in model.named_parameters():
#         if not p.requires_grad:
#             continue
#         is_head = ("head" in n)
#         is_nodecay = (p.dim() == 1) or n.endswith(".bias") or ("bn" in n) or ("norm" in n) or ("layers_norm" in n)
#         if is_head:
#             (head_nodecay if is_nodecay else head_decay).append(p)
#         else:
#             (nodecay if is_nodecay else decay).append(p)
#     param_groups = [
#         {"params": decay,        "lr": base_lr,     "weight_decay": 1e-4},
#         {"params": nodecay,      "lr": base_lr,     "weight_decay": 0.0},
#         {"params": head_decay,   "lr": base_lr*5.0, "weight_decay": 1e-4},
#         {"params": head_nodecay, "lr": base_lr*5.0, "weight_decay": 0.0},
#     ]
#     opt = optim.AdamW(param_groups)

#     # ---- Cosine + warmup（以 optimizer step 為單位）----
#     steps_per_epoch = len(dl_tr)
#     accum = max(1, args.accum_steps)
#     opt_steps_per_epoch = math.ceil(steps_per_epoch / accum)
#     total_steps = opt_steps_per_epoch * args.epochs
#     warmup = max(1, int(0.10 * total_steps))

#     def lr_lambda(step):
#         if step < warmup:
#             return step / max(1, warmup)
#         progress = (step - warmup) / max(1, (total_steps - warmup))
#         return 0.5 * (1 + math.cos(math.pi * progress))
#     sched = optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_lambda)

#     scaler = torch.amp.GradScaler('cuda', enabled=(args.amp == 1 and use_cuda))
#     ema = EMA(model, decay=args.ema_decay) if args.ema == 1 else None

#     best_f1, best_ep = -1.0, -1
#     no_improve = 0
#     global_step = 0

#     def prep_input(xb):
#         # 若包含 mask，最後一個通道就是 mask；幾何在此拼接
#         if args.include_mask == 1 and (args.geom_edge == 1 or args.geom_dist == 1):
#             xb = add_geom_channels(xb, use_edge=bool(args.geom_edge), use_dist=bool(args.geom_dist))
#         # 截到指定 in_ch（保險）
#         xb = xb[:, :in_ch, ...]
#         # ImageNet normalization for RGB (first 3 channels)
#         # This is correct for ViT as well.
#         if args.rgb == 1 and xb.size(1) >= 3:
#             mean = torch.tensor([0.485, 0.456, 0.406], device=xb.device, dtype=xb.dtype).view(1,3,1,1)
#             std  = torch.tensor([0.229, 0.224, 0.225], device=xb.device, dtype=xb.dtype).view(1,3,1,1)
#             xb[:, :3] = (xb[:, :3] - mean) / std
#         return xb

#     print(f"[INFO] in_ch={in_ch} (base={base_in}, added={added}), steps_per_epoch={steps_per_epoch}, total_steps={total_steps}, warmup={warmup}")

#     for ep in range(1, args.epochs + 1):
#         model.train()
#         run_loss = 0.0
#         t0 = time.time()
#         opt.zero_grad(set_to_none=True)

#         for bi, (xb, yb) in enumerate(dl_tr, 1):
#             xb = xb.to(device, non_blocking=use_cuda).to(memory_format=torch.channels_last)
#             yb = yb.to(device, non_blocking=use_cuda)

#             xb = prep_input(xb)
#             if args.aug == 1:
#                 xb = random_augment(xb)

#             with torch.autocast('cuda', enabled=(args.amp == 1 and use_cuda)):
#                 logits = model(xb)
#                 loss = criterion(logits, yb) / accum

#             scaler.scale(loss).backward()

#             if bi % accum == 0:
#                 scaler.unscale_(opt)
#                 torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
#                 scaler.step(opt)
#                 scaler.update()
#                 opt.zero_grad(set_to_none=True)
#                 if ema is not None:
#                     ema.update(model)
#                 sched.step()
#                 global_step += 1

#             run_loss += loss.item() * accum
#             if bi % 200 == 0:
#                 cur_lr = sched.get_last_lr()[0]
#                 print(f"[E{ep:02d}] {bi}/{steps_per_epoch} loss={run_loss/bi:.4f} lr={cur_lr:.2e} gs={global_step}")

#         # 尾端補 step（若未整除）
#         if (steps_per_epoch % accum) != 0:
#             scaler.unscale_(opt)
#             torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
#             scaler.step(opt); scaler.update()
#             opt.zero_grad(set_to_none=True)
#             if ema is not None: ema.update(model)
#             sched.step(); global_step += 1

#         # -------- eval + TTA --------
#         @torch.no_grad()
#         def eval_with(m_for_eval):
#             m_for_eval.eval()
#             y_true, y_pred = [], []
#             for xb, yb in dl_va:
#                 xb = xb.to(device, non_blocking=use_cuda).to(memory_format=torch.channels_last)
#                 yb = yb.to(device, non_blocking=use_cuda)

#                 def forward_once(x):
#                     x = prep_input(x)
#                     return m_for_eval(x)

#                 x0 = xb
#                 if args.tta_val == 2:
#                     logits_list = []
#                     for k in [0,1,2,3]:
#                         xk = torch.rot90(x0, k, dims=[2,3])
#                         logits_list.append(forward_once(xk))
#                         logits_list.append(forward_once(torch.flip(xk, dims=[3])))
#                     logits = torch.stack(logits_list, dim=0).mean(0)
#                 elif args.tta_val == 1:
#                     logits = torch.stack([
#                         forward_once(x0),
#                         forward_once(torch.flip(x0, dims=[3])),
#                         forward_once(torch.flip(x0, dims=[2])),
#                         forward_once(torch.flip(torch.flip(x0, dims=[2]), dims=[3])),
#                     ], dim=0).mean(0)
#                 else:
#                     logits = forward_once(x0)

#                 pred = logits.argmax(1)
#                 y_true += yb.tolist(); y_pred += pred.tolist()

#             cm = confusion_matrix(y_true, y_pred, num_classes=NUM_CLASSES)
#             macro_f1, per_cls = macro_f1_from_cm(cm)
#             return macro_f1, per_cls, cm

#         if ema is not None:
#             ema.apply_shadow(model)
#             macro_f1, per_cls, cm = eval_with(model)
#             ema.restore(model)
#         else:
#             macro_f1, per_cls, cm = eval_with(model)

#         print(f"[VAL] E{ep:02d} macroF1={macro_f1:.3f} per_class={[f'{x:.3f}' for x in per_cls]}")
#         print("CM rows=gt cols=pred:")
#         for r in cm: print(r.tolist())

#         improved = macro_f1 > best_f1 + 1e-6
#         if improved:
#             best_f1, best_ep = macro_f1, ep
#             no_improve = 0
#             state_to_save = build_ema_full_state_dict(model, ema)
#             payload = {
#                 "state_dict": state_to_save,
#                 "best_f1": best_f1,
#                 "epoch": ep,
#                 "ema": (ema is not None),
#                 "in_ch": in_ch,
#                 "num_classes": NUM_CLASSES,
#                 "args": vars(args),
#             }
#             save_p = os.path.join(args.save_dir, f"best_{args.arch}_cell_cls_3cls.pt")
#             torch.save(payload, save_p)
#             print(f"[SAVE] best at epoch {ep} (macroF1={best_f1:.3f}) -> {save_p}")
#         else:
#             no_improve += 1

#         dt = time.time() - t0
#         print(f"[E{ep:02d}] epoch_time={dt/60:.1f} min  (no_improve={no_improve}/{args.early_stop})")

#         if args.early_stop > 0 and no_improve >= args.early_stop:
#             print("[EARLY STOP] no improvement, stop training.")
#             break

#     print(f"[DONE] best macroF1={best_f1:.3f} @ epoch {best_ep}")

# if __name__ == "__main__":
#     main()

# train_vit_subcrop.py
# train_vit_subcrop_cv2fix.py
import os, argparse, math, time, random
import numpy as np
import cv2  # <--- 加入 OpenCV 匯入
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Sampler
from torchvision import transforms
import torchvision.transforms.functional as TF

# 假設這些檔案存在於同目錄或 PYTHONPATH
from data_from_merge_3 import SingleCellFromMerged # <--- 使用 V3 版本
from metrics import confusion_matrix, macro_f1_from_cm
from utils import set_seed

# ======== Backbones ========
from torchvision.models import densenet161, vit_b_16, ViT_B_16_Weights
try:
    from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
    _HAS_CNX_WEIGHTS = True
except Exception:
    from torchvision.models import convnext_tiny
    _HAS_CNX_WEIGHTS = False

NUM_CLASSES = 3  # 0:1to1, 1:1to2, 2:1to4(含k=3或4)

# ---------------- EMA ----------------
class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        for n, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[n] = p.data.clone()

    @torch.no_grad()
    def update(self, model):
        for n, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[n].mul_(self.decay).add_(p.data, alpha=1.0 - self.decay)

    @torch.no_grad()
    def apply_shadow(self, model):
        self.backup = {}
        for n, p in model.named_parameters():
            if n in self.shadow:
                self.backup[n] = p.data.clone()
                p.data.copy_(self.shadow[n])

    @torch.no_grad()
    def restore(self, model):
        for n, p in model.named_parameters():
            if n in self.backup:
                p.data.copy_(self.backup[n])
        self.backup = {}

def build_ema_full_state_dict(model: nn.Module, ema: EMA):
    sd = model.state_dict()
    if ema is None:
        return sd
    with torch.no_grad():
        for name, _ in model.named_parameters():
            if name in ema.shadow:
                sd[name] = ema.shadow[name].detach().clone()
    return sd

# ---------------- Losses ----------------
class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0, alpha=None, reduction="mean", label_smoothing=0.0):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.weight = weight
        self.eps = 1e-8
        self.label_smoothing = label_smoothing

    def forward(self, logits, target):
        num_classes = logits.size(1)
        log_probs = torch.log_softmax(logits, dim=1)
        probs = log_probs.exp()

        with torch.no_grad():
            true_dist = torch.zeros_like(logits)
            true_dist.fill_(self.label_smoothing / (num_classes - 1) if num_classes > 1 else 0.0)
            true_dist.scatter_(1, target.unsqueeze(1), 1.0 - self.label_smoothing)

        pt = (probs * true_dist).sum(dim=1).clamp(min=self.eps)
        focal = (1 - pt).pow(self.gamma)

        if self.alpha is not None:
            if isinstance(self.alpha, (list, tuple, torch.Tensor)):
                alpha_vec = torch.as_tensor(self.alpha, device=logits.device, dtype=logits.dtype)
                at = alpha_vec.gather(0, target)
            else:
                at = torch.full_like(pt, float(self.alpha))
        else:
            at = torch.ones_like(pt)

        if self.weight is not None:
            wt = self.weight.gather(0, target)
        else:
            wt = torch.ones_like(pt)

        loss = -at * wt * focal * (true_dist * log_probs).sum(dim=1)
        return loss.mean() if self.reduction == "mean" else (loss.sum() if self.reduction == "sum" else loss)

def class_balanced_weights(counts, beta=0.9999, device=None):
    counts = np.array([max(1, c) for c in counts], dtype=np.float32)
    eff = 1.0 - np.power(beta, counts)
    w = (1.0 - beta) / eff
    w = w / w.sum() * len(counts)
    t = torch.tensor(w, dtype=torch.float32)
    return t if device is None else t.to(device)

# ---------------- Sampler ----------------
class BalancedBatchSampler(Sampler):
    def __init__(self, labels, batch_size, num_classes=NUM_CLASSES):
        self.labels = np.array(labels, dtype=np.int64)
        self.batch_size = int(batch_size)
        self.num_classes = num_classes
        self.m = max(1, self.batch_size // self.num_classes)
        self.class_indices = {c: np.where(self.labels == c)[0].tolist() for c in range(num_classes)}
        for c in self.class_indices:
            if len(self.class_indices[c]) == 0:
                raise ValueError(f"Class {c} has zero samples.")
            random.shuffle(self.class_indices[c])
        self.ptr = {c: 0 for c in range(num_classes)}
        self.num_batches = int(math.ceil(len(self.labels) / float(self.batch_size)))

    def __iter__(self):
        for _ in range(self.num_batches):
            batch = []
            for c in range(self.num_classes):
                for _ in range(self.m):
                    idxs = self.class_indices[c]
                    p = self.ptr[c]
                    if p >= len(idxs):
                        random.shuffle(idxs)
                        self.ptr[c] = 0
                        p = 0
                    batch.append(idxs[p])
                    self.ptr[c] += 1
            yield batch[:self.batch_size]

    def __len__(self):
        return self.num_batches

# ---------------- GPU aug & geom ----------------
def random_augment(x, p_flip=0.5, p_vflip=0.5, p_rot=0.5, p_bc=0.5, p_erase=0.25):
    B, C, H, W = x.shape
    img_ch = min(3, C)
    img = x[:, :img_ch]

    if torch.rand(1).item() < p_flip:
        img = torch.flip(img, dims=[3]); x = torch.flip(x, dims=[3]); x[:, :img_ch] = img
    if torch.rand(1).item() < p_vflip:
        img = torch.flip(img, dims=[2]); x = torch.flip(x, dims=[2]); x[:, :img_ch] = img
    if torch.rand(1).item() < p_rot:
        k = torch.randint(0, 4, (1,)).item()
        img = torch.rot90(img, k, dims=[2,3]); x = torch.rot90(x, k, dims=[2,3]); x[:, :img_ch] = img
    if torch.rand(1).item() < p_bc:
        a = 0.8 + 0.4 * torch.rand(1, device=x.device); b = (torch.rand(1, device=x.device) - 0.5) * 0.2
        img = (img * a + b).clamp(0, 1); x[:, :img_ch] = img
    if torch.rand(1).item() < p_erase:
        h = int(H * (0.05 + 0.2*torch.rand(1).item())); w = int(W * (0.05 + 0.2*torch.rand(1).item()))
        y0 = torch.randint(0, max(1, H - h), (1,)).item(); x0 = torch.randint(0, max(1, W - w), (1,)).item()
        val = torch.rand((img_ch, 1, 1), device=x.device)
        x[:, :img_ch, y0:y0+h, x0:x0+w] = val
    return x

def make_edge_from_mask(mask):
    dil = F.max_pool2d(mask, kernel_size=3, stride=1, padding=1)
    ero = -F.max_pool2d(-mask, kernel_size=3, stride=1, padding=1)
    edge = (dil - ero).clamp_(0, 1)
    edge = (edge > 0).float()
    edge = (edge - (mask > 0).float()).abs().clamp(0, 1)
    return edge

@torch.no_grad()
def make_pseudodist_from_mask(mask, iters=4):
    x = mask.float(); acc = torch.zeros_like(x); cur = x
    for _ in range(iters):
        cur = F.avg_pool2d(cur, kernel_size=3, stride=1, padding=1)
        acc = acc + cur
    acc = acc / acc.amax(dim=[2,3], keepdim=True).clamp(min=1e-6)
    return acc

def add_geom_channels(x, use_edge=True, use_dist=False):
    mask = x[:, -1:, ...]
    extra = []
    if use_edge: extra.append(make_edge_from_mask(mask))
    if use_dist: extra.append(make_pseudodist_from_mask(mask))
    if extra:
        x = torch.cat([x, *extra], dim=1)
    return x

# ---------------- Backbones: ViT / DenseNet / ConvNeXt ----------------
class ViTCellClassifier(nn.Module):
    def __init__(self, in_ch=4, num_classes=NUM_CLASSES, pretrained=True, freeze_until=0, drop=0.2, img_size=224):
        super().__init__()
        m = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None, image_size=img_size)

        stem_conv = m.conv_proj
        if in_ch != stem_conv.in_channels:
            new = nn.Conv2d(in_ch, stem_conv.out_channels,
                            kernel_size=stem_conv.kernel_size, stride=stem_conv.stride,
                            padding=stem_conv.padding, bias=(stem_conv.bias is not None))
            with torch.no_grad():
                w = stem_conv.weight
                if in_ch >= 3:
                    new.weight[:, :3] = w
                    if in_ch > 3:
                        mean_extra = w[:, :3].mean(dim=1, keepdim=True)
                        for c in range(3, in_ch):
                            new.weight[:, c:c+1] = mean_extra
                else:
                    new.weight[:, :in_ch] = w[:, :in_ch]
                if stem_conv.bias is not None:
                    new.bias.copy_(stem_conv.bias)
            m.conv_proj = new

        self.conv_proj = m.conv_proj
        self.class_token = m.class_token
        self.pos_embedding = m.encoder.pos_embedding
        self.encoder_layers = m.encoder.layers
        self.norm = m.encoder.ln
        self.dropout = m.encoder.dropout

        feat_ch = m.hidden_dim

        self.head = nn.Sequential(
            nn.Dropout(drop),
            nn.Linear(feat_ch, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(drop),
            nn.Linear(256, num_classes),
        )

        if freeze_until >= 1:
            for p in self.conv_proj.parameters(): p.requires_grad = False
            self.class_token.requires_grad = False
            self.pos_embedding.requires_grad = False
        if freeze_until >= 2:
            num_to_freeze = 6 if freeze_until == 2 else 12
            for i, layer in enumerate(self.encoder_layers):
                if i < num_to_freeze:
                    for p in layer.parameters():
                        p.requires_grad = False

    def forward(self, x):
        x = self.conv_proj(x)
        x = x.flatten(2).transpose(1, 2)
        x = torch.cat((self.class_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = x + self.pos_embedding
        x = self.dropout(x)
        x = self.encoder_layers(x)
        x = self.norm(x)
        g = x[:, 0]
        return self.head(g)

class DenseNetCellClassifier(nn.Module):
    def __init__(self, in_ch=4, num_classes=NUM_CLASSES, pretrained=True, freeze_until=0, drop=0.2):
        super().__init__()
        m = densenet161(weights="IMAGENET1K_V1" if pretrained else None)
        conv0 = m.features.conv0
        if in_ch != 3:
            new = nn.Conv2d(in_ch, conv0.out_channels, kernel_size=conv0.kernel_size,
                            stride=conv0.stride, padding=conv0.padding, bias=False)
            with torch.no_grad():
                w = conv0.weight
                if in_ch >= 3:
                    new.weight[:, :3] = w
                    if in_ch > 3:
                        mean_extra = w.mean(dim=1, keepdim=True)
                        for i in range(3, in_ch):
                            new.weight[:, i:i+1] = mean_extra
                else:
                    new.weight[:, :in_ch] = w[:, :in_ch]
            m.features.conv0 = new

        self.features = m.features
        feat_ch = m.classifier.in_features
        self.norm = nn.BatchNorm2d(feat_ch)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Dropout(drop),
            nn.Linear(feat_ch, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(drop),
            nn.Linear(256, num_classes),
        )

        if freeze_until > 0:
            for name, p in self.features.named_parameters():
                p.requires_grad = False
                if "denseblock2" in name:
                    break

    def forward(self, x):
        f = self.features(x)
        f = self.norm(f); f = self.relu(f)
        g = self.pool(f).flatten(1)
        return self.head(g)

class ConvNeXtCellClassifier(nn.Module):
    def __init__(self, in_ch=4, num_classes=NUM_CLASSES, pretrained=True, freeze_until=0, drop=0.2):
        super().__init__()
        if _HAS_CNX_WEIGHTS and pretrained:
            m = convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        else:
            m = convnext_tiny(weights=("IMAGENET1K_V1" if pretrained else None))

        stem_conv = m.features[0][0]
        if in_ch != stem_conv.in_channels:
            new = nn.Conv2d(in_ch, stem_conv.out_channels,
                            kernel_size=stem_conv.kernel_size, stride=stem_conv.stride,
                            padding=stem_conv.padding, bias=(stem_conv.bias is not None))
            with torch.no_grad():
                w = stem_conv.weight
                if in_ch >= 3:
                    new.weight[:, :3] = w
                    if in_ch > 3:
                        mean_extra = w[:, :3].mean(dim=1, keepdim=True)
                        for c in range(3, in_ch):
                            new.weight[:, c:c+1] = mean_extra
                else:
                    new.weight[:, :in_ch] = w[:, :in_ch]
                if stem_conv.bias is not None:
                    new.bias.copy_(stem_conv.bias)
            m.features[0][0] = new

        self.features = m.features
        feat_ch = m.classifier[2].in_features if isinstance(m.classifier[-1], nn.Linear) else 768
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.norm = nn.LayerNorm(feat_ch, eps=1e-6)
        self.head = nn.Sequential(
            nn.Dropout(drop),
            nn.Linear(feat_ch, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(drop),
            nn.Linear(256, num_classes),
        )

        if freeze_until > 0:
            for p in self.features.parameters():
                p.requires_grad = False

    def forward(self, x):
        f = self.features(x)
        g = self.pool(f).flatten(1)
        g = self.norm(g)
        return self.head(g)

# ---------------- misc ----------------
def _infer_device(user_choice=None):
    if user_choice is not None:
        user_choice = user_choice.lower()
        if user_choice == "cpu":
            return torch.device("cpu")
        if user_choice == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _labels_of_items(items):
    labels = []
    for it in items:
        k = int(it.get("k", 1))
        labels.append(0 if k == 1 else (1 if k == 2 else 2))
    return labels

def _class_counts(ds, num_classes=NUM_CLASSES):
    ys = _labels_of_items(ds.items)
    cnt = [0]*num_classes
    for y in ys: cnt[y]+=1
    return cnt, ys

# ---------------- train ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--merged_root", required=True)
    ap.add_argument("--save_dir", required=True)
    ap.add_argument("--arch", type=str, default="vit_b_16", choices=["vit_b_16", "convnext_tiny","densenet161"])
    ap.add_argument("--center_crop_h", type=int, default=1024, help="中心裁切大圖的高度")
    ap.add_argument("--center_crop_w", type=int, default=512, help="中心裁切大圖的寬度")
    ap.add_argument("--sub_crop_size", type=int, default=224, help="送入 ViT 的子圖塊大小")
    ap.add_argument("--rgb", type=int, default=1)
    ap.add_argument("--include_mask", type=int, default=1)
    ap.add_argument("--bs", type=int, default=32, help="Batch size for *large images*")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=2025)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--amp", type=int, default=1)
    ap.add_argument("--class_weight", type=str, default="auto")
    ap.add_argument("--balanced_batch", type=int, default=1)
    ap.add_argument("--freeze_until", type=int, default=0, help="0:all, 1:freeze patch/pos/cls, 2:..+6blocks, 3:..+12blocks (ViT)")
    ap.add_argument("--label_smoothing", type=float, default=0.05)
    ap.add_argument("--loss", type=str, default="cb_focal", choices=["ce","focal","cb_focal"])
    ap.add_argument("--gamma", type=float, default=2.0)
    ap.add_argument("--alpha", type=float, default=None)
    ap.add_argument("--early_stop", type=int, default=50)
    ap.add_argument("--ema", type=int, default=1)
    ap.add_argument("--ema_decay", type=float, default=0.999)
    ap.add_argument("--aug", type=int, default=1, help="是否對 '子圖塊' 進行 TTA")
    ap.add_argument("--tta_val", type=int, default=0, help="[已棄用] 論文方法使用 '子圖投票'，此參數無效")
    ap.add_argument("--geom_edge", type=int, default=1)
    ap.add_argument("--geom_dist", type=int, default=1)
    ap.add_argument("--accum_steps", type=int, default=1)
    ap.add_argument("--train_rot_deg", type=float, default=30.0, help="論文 2.5.3.1 節，訓練時隨機旋轉角度")
    ap.add_argument("--val_crop_step_ratio", type=float, default=0.25, help="論文 3.3 節，驗證時等距裁切的步長比例 (1/4)")
    ap.add_argument("--val_dark_thresh", type=float, default=30.0, help="論文 2.5.3.2 節，品質控制的亮度閾值 (0-255)")
    ap.add_argument("--val_dark_drop_ratio", type=float, default=0.65, help="論文 2.5.3.2 節，品質控制的暗部比例閾值")
    ap.add_argument("--eval_chunk_size", type=int, default=256, help="驗證時子圖塊分批大小，避免一次吃爆 VRAM")

    args = ap.parse_args()

    if (args.geom_edge == 1 or args.geom_dist == 1) and args.include_mask == 0:
        print("[WARN] geom_edge/dist 需要 mask 通道；自動將 --include_mask 設為 1")
        args.include_mask = 1

    set_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)
    device = _infer_device(args.device)
    use_cuda = (device.type == "cuda")
    if use_cuda:
        torch.backends.cudnn.benchmark = True

    # ---- datasets ----
    large_img_size = (args.center_crop_h, args.center_crop_w)
    print(f"[INFO] 載入 '中心裁切大圖' (H, W) = {large_img_size}，將從中裁切 {args.sub_crop_size}x{args.sub_crop_size} 的子圖塊")

    ds_tr = SingleCellFromMerged(
        args.merged_root, split="train",
        img_size=large_img_size,
        use_rgb=bool(args.rgb),
        include_mask_channel=bool(args.include_mask)
    )
    ds_va = SingleCellFromMerged(
        args.merged_root, split="val",
        img_size=large_img_size,
        use_rgb=bool(args.rgb),
        include_mask_channel=bool(args.include_mask)
    )

    base_in = (3 if args.rgb == 1 else 1) + (1 if args.include_mask == 1 else 0)
    added = (1 if args.geom_edge == 1 else 0) + (1 if args.geom_dist == 1 else 0)
    in_ch = base_in + added

    # ---- model ----
    if args.arch == "vit_b_16":
        model = ViTCellClassifier(in_ch=in_ch, num_classes=NUM_CLASSES,
                                  pretrained=True, freeze_until=args.freeze_until,
                                  img_size=args.sub_crop_size)
    elif args.arch == "convnext_tiny":
        model = ConvNeXtCellClassifier(in_ch=in_ch, num_classes=NUM_CLASSES,
                                       pretrained=True, freeze_until=args.freeze_until)
    else:
        model = DenseNetCellClassifier(in_ch=in_ch, num_classes=NUM_CLASSES,
                                       pretrained=True, freeze_until=args.freeze_until)
    model = model.to(device).to(memory_format=torch.channels_last)

    # ---- loss ----
    counts, ys_tr = _class_counts(ds_tr, NUM_CLASSES)
    if args.class_weight == "auto":
        cls_w = torch.tensor([sum(counts)/max(1, c) for c in counts], dtype=torch.float32, device=device)
        cls_w = cls_w / (cls_w.mean().clamp(min=1e-6))
    else:
        cls_w = None

    if args.loss == "cb_focal":
        cb_w = class_balanced_weights(counts, beta=0.9999, device=device)
        criterion = FocalLoss(weight=cb_w, gamma=args.gamma, alpha=args.alpha, label_smoothing=0.0)
    elif args.loss == "focal":
        criterion = FocalLoss(weight=cls_w, gamma=args.gamma, alpha=args.alpha, label_smoothing=args.label_smoothing)
    else:
        criterion = nn.CrossEntropyLoss(weight=cls_w, label_smoothing=args.label_smoothing)

    # ---- dataloaders ----
    loader_kwargs = dict(num_workers=max(0, args.workers), pin_memory=use_cuda)
    if args.workers > 0:
        loader_kwargs.update(persistent_workers=True, prefetch_factor=4)

    if args.balanced_batch == 1:
        batch_sampler = BalancedBatchSampler(ys_tr, batch_size=args.bs, num_classes=NUM_CLASSES)
        dl_tr = DataLoader(ds_tr, batch_sampler=batch_sampler, **loader_kwargs)
    else:
        dl_tr = DataLoader(ds_tr, batch_size=args.bs, shuffle=True, **loader_kwargs)
    dl_va = DataLoader(ds_va, batch_size=args.bs, shuffle=False, **loader_kwargs)

    # ---- optimizer ----
    base_lr = args.lr
    decay, nodecay, head_decay, head_nodecay = [], [], [], []
    for n, p in model.named_parameters():
        if not p.requires_grad: continue
        is_head = ("head" in n)
        is_nodecay = (p.dim() == 1) or n.endswith(".bias") or ("bn" in n) or ("norm" in n) or ("layers_norm" in n)
        if is_head:
            (head_nodecay if is_nodecay else head_decay).append(p)
        else:
            (nodecay if is_nodecay else decay).append(p)
    param_groups = [
        {"params": decay,        "lr": base_lr,     "weight_decay": 1e-4},
        {"params": nodecay,      "lr": base_lr,     "weight_decay": 0.0},
        {"params": head_decay,   "lr": base_lr*5.0, "weight_decay": 1e-4},
        {"params": head_nodecay, "lr": base_lr*5.0, "weight_decay": 0.0},
    ]
    opt = optim.AdamW(param_groups)

    # ---- scheduler ----
    steps_per_epoch = len(dl_tr)
    accum = max(1, args.accum_steps)
    opt_steps_per_epoch = math.ceil(steps_per_epoch / accum)
    total_steps = opt_steps_per_epoch * args.epochs
    warmup = max(1, int(0.10 * total_steps))

    def lr_lambda(step):
        if step < warmup:
            return step / max(1, warmup)
        progress = (step - warmup) / max(1, (total_steps - warmup))
        return 0.5 * (1 + math.cos(math.pi * progress))
    sched = optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_lambda)

    scaler = torch.amp.GradScaler('cuda', enabled=(args.amp == 1 and use_cuda))
    ema = EMA(model, decay=args.ema_decay) if args.ema == 1 else None

    best_f1, best_ep = -1.0, -1
    no_improve = 0
    global_step = 0

    def prep_input(xb):
        if args.include_mask == 1 and (args.geom_edge == 1 or args.geom_dist == 1):
            xb = add_geom_channels(xb, use_edge=bool(args.geom_edge), use_dist=bool(args.geom_dist))
        xb = xb[:, :in_ch, ...]
        if args.rgb == 1 and xb.size(1) >= 3:
            mean = torch.tensor([0.485, 0.456, 0.406], device=xb.device, dtype=xb.dtype).view(1,3,1,1)
            std  = torch.tensor([0.229, 0.224, 0.225], device=xb.device, dtype=xb.dtype).view(1,3,1,1)
            xb[:, :3] = (xb[:, :3] - mean) / std
        return xb

    print(f"[INFO] in_ch={in_ch} (base={base_in}, added={added}), steps_per_epoch={steps_per_epoch}, total_steps={total_steps}, warmup={warmup}")

    sub_crop_h = args.sub_crop_size
    sub_crop_w = args.sub_crop_size

    for ep in range(1, args.epochs + 1):
        model.train()
        run_loss = 0.0
        t0 = time.time()
        opt.zero_grad(set_to_none=True)

        for bi, (xb_large, yb) in enumerate(dl_tr, 1):
            xb_large = xb_large.to(device, non_blocking=use_cuda).to(memory_format=torch.channels_last)
            yb = yb.to(device, non_blocking=use_cuda)

            xb_large_prep = prep_input(xb_large)
            angle = random.uniform(-args.train_rot_deg, args.train_rot_deg)
            xb_large_rotated = TF.rotate(xb_large_prep, angle)
            i, j, h, w = transforms.RandomCrop.get_params(xb_large_rotated, output_size=(sub_crop_h, sub_crop_w))
            xb_subcrop = TF.crop(xb_large_rotated, i, j, h, w)

            if args.aug == 1:
                xb = random_augment(xb_subcrop)
            else:
                xb = xb_subcrop

            with torch.autocast('cuda', enabled=(args.amp == 1 and use_cuda)):
                logits = model(xb)
                loss = criterion(logits, yb) / accum

            scaler.scale(loss).backward()

            if bi % accum == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                if ema is not None:
                    ema.update(model)
                sched.step()
                global_step += 1

            run_loss += loss.item() * accum
            if bi % 200 == 0:
                cur_lr = sched.get_last_lr()[0]
                print(f"[E{ep:02d}] {bi}/{steps_per_epoch} loss={run_loss/bi:.4f} lr={cur_lr:.2e} gs={global_step}")

        if (steps_per_epoch % accum) != 0:
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(opt); scaler.update()
            opt.zero_grad(set_to_none=True)
            if ema is not None: ema.update(model)
            sched.step(); global_step += 1

        # -------- eval: 子圖塊投票 + 分批跑避免 OOM --------
        @torch.no_grad()
        def eval_with(m_for_eval):
            m_for_eval.eval()
            y_true_img, y_pred_img = [], []

            S = args.sub_crop_size
            step = int(S * args.val_crop_step_ratio)
            intensity_thresh_cv2 = args.val_dark_thresh
            drop_ratio_thresh = args.val_dark_drop_ratio
            chunk_size = max(1, int(args.eval_chunk_size))

            for xb_large_unnorm, yb_large in dl_va:
                xb_large_unnorm = xb_large_unnorm.to(device, non_blocking=use_cuda).to(memory_format=torch.channels_last)
                B, C_unnorm, H, W = xb_large_unnorm.shape

                y_steps = list(range(0, H - S + 1, step))
                if not y_steps or y_steps[-1] != H - S: y_steps.append(H - S)
                x_steps = list(range(0, W - S + 1, step))
                if not x_steps or x_steps[-1] != W - S: x_steps.append(W - S)

                sub_crops_list = []
                img_indices_list = []

                for y in y_steps:
                    for x in x_steps:
                        sub_crop = xb_large_unnorm[:, :, y:y+S, x:x+S]
                        sub_crops_list.append(sub_crop)
                        img_indices_list.append(torch.arange(B, device=device))

                if not sub_crops_list:
                    y_true_img.extend(yb_large.tolist())
                    y_pred_img.extend([0] * B)
                    continue

                sub_crops_batch_unnorm = torch.cat(sub_crops_list, dim=0)  # [B*N_crops, C_unnorm, S, S]
                img_indices = torch.cat(img_indices_list, dim=0)           # [B*N_crops]

                # 2. 品質控制 (OpenCV)
                rgb_sub_crops = sub_crops_batch_unnorm[:, :3, :, :]
                rgb_sub_crops_np = rgb_sub_crops.cpu().numpy().transpose(0, 2, 3, 1)
                rgb_sub_crops_np_uint8 = (rgb_sub_crops_np * 255).astype(np.uint8)
                hsv_list = [cv2.cvtColor(img, cv2.COLOR_RGB2HSV) for img in rgb_sub_crops_np_uint8]
                hsv_np = np.stack(hsv_list, axis=0)
                brightness_np = hsv_np[:, :, :, 2]

                dark_pixels = (brightness_np <= intensity_thresh_cv2).astype(np.float32).sum(axis=(1, 2))
                dark_ratio = dark_pixels / (S * S)

                preserved_mask_np = (dark_ratio < drop_ratio_thresh)
                preserved_indices = np.where(preserved_mask_np)[0]

                if preserved_indices.size == 0:
                    y_true_img.extend(yb_large.tolist())
                    y_pred_img.extend([0] * B)
                    continue

                # 3. 對 '保留' 的子圖塊分批預測，避免一次塞爆 VRAM
                preserved_sub_crops_unnorm = sub_crops_batch_unnorm[preserved_indices]
                preserved_img_indices = img_indices[preserved_indices]  # GPU tensor

                preserved_sub_crops_norm = prep_input(preserved_sub_crops_unnorm)

                N = preserved_sub_crops_norm.shape[0]
                pred_chunks = []
                for s in range(0, N, chunk_size):
                    e = min(s + chunk_size, N)
                    xb_chunk = preserved_sub_crops_norm[s:e]
                    with torch.autocast('cuda', enabled=(args.amp == 1 and use_cuda)):
                        logits_chunk = m_for_eval(xb_chunk)
                    pred_chunks.append(logits_chunk.argmax(1).cpu())

                preds = torch.cat(pred_chunks, dim=0)                  # on CPU
                preserved_img_indices_cpu = preserved_img_indices.cpu() # 同樣搬到 CPU

                # 4. 投票 (Voting)
                final_preds = []
                for b_idx in range(B):
                    votes_for_img = preds[preserved_img_indices_cpu == b_idx]
                    if votes_for_img.numel() == 0:
                        final_pred = 0
                    else:
                        final_pred = torch.bincount(votes_for_img).argmax().item()
                    final_preds.append(final_pred)

                y_true_img.extend(yb_large.tolist())
                y_pred_img.extend(final_preds)

            cm = confusion_matrix(y_true_img, y_pred_img, num_classes=NUM_CLASSES)
            macro_f1, per_cls = macro_f1_from_cm(cm)
            return macro_f1, per_cls, cm
        # --- eval_with end ---

        if ema is not None:
            ema.apply_shadow(model)
            macro_f1, per_cls, cm = eval_with(model)
            ema.restore(model)
        else:
            macro_f1, per_cls, cm = eval_with(model)

        print(f"[VAL] E{ep:02d} (Sub-crop Voting) macroF1={macro_f1:.3f} per_class={[f'{x:.3f}' for x in per_cls]}")
        print("CM rows=gt cols=pred:")
        for r in cm: print(r.tolist())

        improved = macro_f1 > best_f1 + 1e-6
        if improved:
            best_f1, best_ep = macro_f1, ep
            no_improve = 0
            state_to_save = build_ema_full_state_dict(model, ema)
            payload = {
                "state_dict": state_to_save,
                "best_f1": best_f1,
                "epoch": ep,
                "ema": (ema is not None),
                "in_ch": in_ch,
                "num_classes": NUM_CLASSES,
                "args": vars(args),
            }
            save_p = os.path.join(args.save_dir, f"best_{args.arch}_cell_cls_3cls_SUBCROP_CV2FIX.pt")
            torch.save(payload, save_p)
            print(f"[SAVE] best at epoch {ep} (macroF1={best_f1:.3f}) -> {save_p}")
        else:
            no_improve += 1

        dt = time.time() - t0
        print(f"[E{ep:02d}] epoch_time={dt/60:.1f} min  (no_improve={no_improve}/{args.early_stop})")

        if args.early_stop > 0 and no_improve >= args.early_stop:
            print("[EARLY STOP] no improvement, stop training.")
            break

    print(f"[DONE] best macroF1={best_f1:.3f} @ epoch {best_ep}")

if __name__ == "__main__":
    main()


