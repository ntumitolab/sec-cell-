import os, sys, argparse, math, time, random, json, re, subprocess
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Sampler
from data_from_merged_2 import SingleCellFromMerged
from metrics import confusion_matrix, macro_f1_from_cm
from utils import set_seed

# ==================== 全域設定：3 類 ====================
NUM_CLASSES = 3  # 0:1to1, 1:1to2, 2:1to4(含k=3或4)

# -------------------- UNet building blocks --------------------
class DoubleConv(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.block(x)

class Down(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_c, out_c)
    def forward(self, x):
        return self.conv(self.pool(x))

# -------------------- Attention / Pooling / ASPP --------------------
class SEBlock(nn.Module):
    """簡潔版 SE 注意力"""
    def __init__(self, ch, r=16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(ch, max(1, ch//r)), nn.ReLU(True),
            nn.Linear(max(1, ch//r), ch), nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _, _ = x.shape
        w = self.fc(self.pool(x).view(b, c)).view(b, c, 1, 1)
        return x * w

class GeM(nn.Module):
    """Generalized Mean Pooling；支援 mask 權重（數值穩定版）"""
    def __init__(self, p=3.0, eps=1e-6, p_min=1.0, p_max=6.0, learn_p=False):
        super().__init__()
        self.eps = eps
        self.p_min = p_min
        self.p_max = p_max
        self.learn_p = learn_p
        if learn_p:
            self.p = nn.Parameter(torch.tensor(float(p)))
        else:
            self.register_buffer("p", torch.tensor(float(p)))

    def _get_p(self, x):
        p = self.p
        if isinstance(p, torch.Tensor):
            p = p.to(device=x.device, dtype=x.dtype)
        else:
            p = torch.tensor(float(p), device=x.device, dtype=x.dtype)
        return p.clamp(min=self.p_min, max=self.p_max)

    def forward(self, x, mask=None):
        p = self._get_p(x)
        x = x.clamp(min=self.eps).pow(p)
        if mask is None:
            out = F.adaptive_avg_pool2d(x, 1).clamp(min=self.eps).pow(1.0 / p)
        else:
            w = mask.clamp(0, 1)
            num = (x * w).sum(dim=(2, 3), keepdim=True)
            den = w.sum(dim=(2, 3), keepdim=True).clamp(min=self.eps)
            out = (num / den).clamp(min=self.eps).pow(1.0 / p)
        return out

class ASPP(nn.Module):
    def __init__(self, in_ch, out_ch, rates=(1, 6, 12, 18)):
        super().__init__()
        brs = []
        brs.append(nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(True)
        ))
        for r in rates[1:]:
            brs.append(nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=r, dilation=r, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(True)
            ))
        self.branches = nn.ModuleList(brs)
        self.proj = nn.Sequential(
            nn.Conv2d(out_ch * len(brs), out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(True)
        )

    def forward(self, x):
        xs = [b(x) for b in self.branches]
        return self.proj(torch.cat(xs, dim=1))

class UNetEncoderClassifier(nn.Module):
    def __init__(self, in_ch=4, base_in=4, num_classes=3, drop=0.2, width=64,
                 use_se=True, use_aspp=True, gem_p_init=3.0, learn_gem_p=False):
        super().__init__()
        self.in_ch = in_ch
        self.base_in = base_in
        self.num_classes = num_classes
        self.use_se = use_se
        self.use_aspp = use_aspp

        c1, c2, c3, c4, c5 = width, width*2, width*4, width*8, width*16

        self.enc1 = DoubleConv(in_ch, c1)
        self.enc2 = Down(c1, c2)
        self.enc3 = Down(c2, c3)
        self.enc4 = Down(c3, c4)
        self.enc5 = Down(c4, c5)

        self.proj3 = nn.Conv2d(c3, 128, 1, bias=False)
        self.proj4 = nn.Conv2d(c4, 128, 1, bias=False)
        self.proj5 = nn.Conv2d(c5, 128, 1, bias=False)
        self.fuse_bn = nn.BatchNorm2d(128)

        if use_se:
            self.se = SEBlock(128, r=16)
        if use_aspp:
            self.aspp = ASPP(128, 128, rates=(1, 6, 12, 18))

        self.gem = GeM(p=gem_p_init, learn_p=learn_gem_p)
        self.head = nn.Sequential(
            nn.Dropout(drop),
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(drop),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        mask_idx = self.base_in - 1
        mask = x[:, mask_idx:mask_idx+1, ...].clamp(0, 1)

        f1 = self.enc1(x)
        f2 = self.enc2(f1)
        f3 = self.enc3(f2)
        f4 = self.enc4(f3)
        f5 = self.enc5(f4)

        H, W = x.shape[2:]
        u3 = F.interpolate(self.proj3(f3), size=(H, W), mode="bilinear", align_corners=False)
        u4 = F.interpolate(self.proj4(f4), size=(H, W), mode="bilinear", align_corners=False)
        u5 = F.interpolate(self.proj5(f5), size=(H, W), mode="bilinear", align_corners=False)

        feat = self.fuse_bn(u3 + u4 + u5)
        feat = F.relu(feat, inplace=True)

        if self.use_se:
            feat = self.se(feat)
        if self.use_aspp:
            feat = self.aspp(feat)

        g = self.gem(feat, mask=mask).flatten(1)
        return self.head(g)

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
            if num_classes > 1:
                true_dist.fill_(self.label_smoothing / (num_classes - 1))
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

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss

def class_balanced_weights(counts, beta=0.9999, device=None):
    counts = np.array([max(1, c) for c in counts], dtype=np.float32)
    eff_num = 1.0 - np.power(beta, counts)
    weights = (1.0 - beta) / eff_num
    weights = weights / weights.sum() * len(counts)
    t = torch.tensor(weights, dtype=torch.float32)
    return t if device is None else t.to(device)

def precision_recall_from_cm(cm):
    cm = np.asarray(cm, dtype=np.int64)
    tp = np.diag(cm).astype(np.float64)
    fp = cm.sum(axis=0).astype(np.float64) - tp
    fn = cm.sum(axis=1).astype(np.float64) - tp

    precision = tp / np.clip(tp + fp, 1e-12, None)
    recall = tp / np.clip(tp + fn, 1e-12, None)
    macro_precision = float(np.mean(precision))
    macro_recall = float(np.mean(recall))
    return macro_precision, macro_recall, precision.tolist(), recall.tolist()

def denorm_for_vis(x, rgb=False):
    x = x.detach().float().cpu().clone()
    if rgb and x.size(0) >= 3:
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
        img = x[:3] * std + mean
        img = img.clamp(0, 1).permute(1,2,0).numpy()
    else:
        img = x[0].clamp(0, 1).numpy()
    return img

def apply_colormap_on_image(img, cam, alpha=0.35):
    cam = np.clip(cam, 0.0, 1.0)
    cmap = plt.get_cmap('jet')
    heat = cmap(cam)[..., :3]
    if img.ndim == 2:
        base = np.repeat(img[..., None], 3, axis=2)
    else:
        base = img[..., :3]
    out = (1 - alpha) * base + alpha * heat
    return np.clip(out, 0.0, 1.0)

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self.fwd_handle = target_layer.register_forward_hook(self._forward_hook)
        self.bwd_handle = target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, inputs, output):
        self.activations = output

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def remove(self):
        self.fwd_handle.remove()
        self.bwd_handle.remove()

    def __call__(self, x, class_idx=None):
        self.model.zero_grad(set_to_none=True)
        logits = self.model(x)
        if class_idx is None:
            class_idx = logits.argmax(dim=1)
        elif isinstance(class_idx, int):
            class_idx = torch.full((x.size(0),), class_idx, device=logits.device, dtype=torch.long)
        score = logits.gather(1, class_idx.view(-1, 1)).sum()
        score.backward()

        grads = self.gradients
        acts = self.activations
        weights = grads.mean(dim=(2,3), keepdim=True)
        cam = (weights * acts).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=x.shape[2:], mode='bilinear', align_corners=False)
        cam = cam.squeeze(1)
        cam_min = cam.amin(dim=(1,2), keepdim=True)
        cam_max = cam.amax(dim=(1,2), keepdim=True)
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)
        return logits.detach(), cam.detach(), class_idx.detach()

def save_gradcam_examples(model_for_eval, dl_va, device, use_cuda, prep_input_fn, args, out_dir, class_names=None, max_samples=24):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    target_layer = model_for_eval.aspp.proj if getattr(model_for_eval, 'use_aspp', False) else model_for_eval.fuse_bn
    gradcam = GradCAM(model_for_eval, target_layer)
    saved = 0
    model_for_eval.eval()

    for xb, yb in dl_va:
        xb = xb.to(device, non_blocking=use_cuda).to(memory_format=torch.channels_last)
        yb = yb.to(device, non_blocking=use_cuda)
        x_in = prep_input_fn(xb)

        bs = x_in.size(0)
        for i in range(bs):
            if saved >= max_samples:
                gradcam.remove()
                return

            xi = x_in[i:i+1].clone().detach().requires_grad_(True)
            yi = int(yb[i].item())
            logits, cam, pred_idx = gradcam(xi)
            probs = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()
            pi = int(pred_idx.item())

            img = denorm_for_vis(xi[0], rgb=bool(args.rgb))
            cam_np = cam[0].cpu().numpy()
            overlay = apply_colormap_on_image(img, cam_np, alpha=0.35)

            mask_idx = model_for_eval.base_in - 1
            mask = xi[0, mask_idx].detach().cpu().numpy() if xi.size(1) > mask_idx else None

            fig, axes = plt.subplots(1, 4 if mask is not None else 3, figsize=(14, 4))
            axes = np.atleast_1d(axes)
            if img.ndim == 2:
                axes[0].imshow(img, cmap='gray')
            else:
                axes[0].imshow(img)
            axes[0].set_title('input')
            axes[0].axis('off')

            if mask is not None:
                axes[1].imshow(mask, cmap='gray')
                axes[1].set_title('mask')
                axes[1].axis('off')
                cam_ax = axes[2]
                overlay_ax = axes[3]
            else:
                cam_ax = axes[1]
                overlay_ax = axes[2]

            cam_ax.imshow(cam_np, cmap='jet')
            cam_ax.set_title('grad-cam')
            cam_ax.axis('off')

            overlay_ax.imshow(overlay)
            gt_name = class_names[yi] if class_names else str(yi)
            pred_name = class_names[pi] if class_names else str(pi)
            overlay_ax.set_title(f'GT={gt_name} Pred={pred_name}\nprob={probs[pi]:.3f}')
            overlay_ax.axis('off')

            fig.suptitle(' | '.join([f'{class_names[k] if class_names else k}:{probs[k]:.3f}' for k in range(len(probs))]), fontsize=10)
            fig.tight_layout()
            fig.savefig(out_dir / f'gradcam_{saved:03d}_gt{yi}_pred{pi}.png', dpi=150, bbox_inches='tight')
            plt.close(fig)
            saved += 1

    gradcam.remove()

class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        for name, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[name] = p.data.clone()

    def update(self, model):
        with torch.no_grad():
            for name, p in model.named_parameters():
                if p.requires_grad:
                    self.shadow[name].mul_(self.decay).add_(p.data, alpha=1.0 - self.decay)

    def apply_shadow(self, model):
        self.backup = {}
        for name, p in model.named_parameters():
            if name in self.shadow:
                self.backup[name] = p.data.clone()
                p.data.copy_(self.shadow[name])

    def restore(self, model):
        for name, p in model.named_parameters():
            if name in self.backup:
                p.data.copy_(self.backup[name])
        self.backup = {}

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
        if k == 1:
            labels.append(0)
        elif k == 2:
            labels.append(1)
        else:
            labels.append(2)
    return labels

def _class_counts(ds, num_classes=NUM_CLASSES):
    ys = _labels_of_items(ds.items)
    cnt = [0]*num_classes
    for y in ys:
        cnt[y] += 1
    return cnt, ys

class BalancedBatchSampler(Sampler):
    def __init__(self, labels, batch_size, num_classes=NUM_CLASSES):
        self.labels = np.array(labels, dtype=np.int64)
        self.batch_size = int(batch_size)
        self.num_classes = num_classes
        self.m = max(1, self.batch_size // self.num_classes)
        self.class_indices = {c: np.where(self.labels == c)[0].tolist()
                              for c in range(num_classes)}
        for c in self.class_indices:
            if len(self.class_indices[c]) == 0:
                raise ValueError(f"Class {c} has zero samples; cannot build balanced batches.")
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

def make_edge_from_mask(mask):
    dil = F.max_pool2d(mask, kernel_size=3, stride=1, padding=1)
    ero = -F.max_pool2d(-mask, kernel_size=3, stride=1, padding=1)
    edge = (dil - ero).clamp_(0, 1)
    edge = (edge > 0).float()
    edge = (edge - (mask > 0).float()).abs().clamp(0, 1)
    return edge

@torch.no_grad()
def make_pseudodist_from_mask(mask, iters=4):
    x = mask.float()
    acc = torch.zeros_like(x)
    cur = x
    for _ in range(iters):
        cur = F.avg_pool2d(cur, kernel_size=3, stride=1, padding=1)
        acc = acc + cur
    acc = acc / acc.amax(dim=[2, 3], keepdim=True).clamp(min=1e-6)
    return acc

def add_geom_channels(x, mask_index=-1, rgb_mode=True, use_edge=True, use_dist=False):
    mask = x[:, mask_index:mask_index+1, ...]
    extra = []
    if use_edge:
        extra.append(make_edge_from_mask(mask))
    if use_dist:
        extra.append(make_pseudodist_from_mask(mask))
    if len(extra):
        x = torch.cat([x, *extra], dim=1)
    return x

def random_augment(x, p_flip=0.5, p_vflip=0.5, p_rot=0.5, p_bc=0.5, p_erase=0.25):
    B, C, H, W = x.shape
    img_ch = min(3, C)
    img = x[:, :img_ch]
    if torch.rand(1).item() < p_flip:
        img = torch.flip(img, dims=[3])
        x = torch.flip(x, dims=[3])
        x[:, :img_ch] = img
    if torch.rand(1).item() < p_vflip:
        img = torch.flip(img, dims=[2])
        x = torch.flip(x, dims=[2])
        x[:, :img_ch] = img
    if torch.rand(1).item() < p_rot:
        k = torch.randint(0, 4, (1,)).item()
        img = torch.rot90(img, k, dims=[2, 3])
        x = torch.rot90(x, k, dims=[2, 3])
        x[:, :img_ch] = img
    if torch.rand(1).item() < p_bc:
        a = 0.8 + 0.4 * torch.rand(1, device=x.device)
        b = (torch.rand(1, device=x.device) - 0.5) * 0.2
        img = (img * a + b).clamp(0, 1)
        x[:, :img_ch] = img
    if torch.rand(1).item() < p_erase:
        h = int(H * (0.05 + 0.2 * torch.rand(1).item()))
        w = int(W * (0.05 + 0.2 * torch.rand(1).item()))
        y0 = torch.randint(0, max(1, H - h), (1,)).item()
        x0 = torch.randint(0, max(1, W - w), (1,)).item()
        val = torch.rand((img_ch, 1, 1), device=x.device)
        x[:, :img_ch, y0:y0+h, x0:x0+w] = val
    return x



def _sample_loguniform(rng, low, high):
    return float(math.exp(rng.uniform(math.log(low), math.log(high))))

def sample_hparams_random(base_args, trial_idx):
    rng = random.Random(int(base_args.hpo_seed) + int(trial_idx))
    return {
        "lr": _sample_loguniform(rng, 5e-5, 5e-4),
        "la_tau": rng.choice([0.0, 0.3, 0.5, 0.8, 1.0]),
        "gamma": rng.choice([1.0, 1.5, 2.0, 2.5]),
        "label_smoothing": rng.choice([0.0, 0.02, 0.05]),
        "balanced_batch": rng.choice([0, 1]),
        "geom_edge": rng.choice([0, 1]),
        "geom_dist": rng.choice([0, 1]),
        "se": rng.choice([0, 1]),
        "aspp": rng.choice([0, 1]),
        "gem_p": rng.choice([2.0, 3.0, 4.0]),
    }

def _extract_best_f1_from_log(log_path):
    txt = Path(log_path).read_text(encoding='utf-8', errors='ignore')
    ms = re.findall(r"\[DONE\] best macroF1=([0-9.]+)", txt)
    if ms:
        return float(ms[-1])
    ms = re.findall(r"macroF1=([0-9.]+)", txt)
    return float(ms[-1]) if ms else None

def _build_subprocess_cmd(base_args, trial_dir, trial_params):
    cmd = [sys.executable, os.path.abspath(sys.argv[0]),
           "--merged_root", str(base_args.merged_root),
           "--save_dir", str(trial_dir),
           "--img_size", str(base_args.img_size),
           "--rgb", str(base_args.rgb),
           "--bs", str(base_args.bs),
           "--epochs", str(base_args.hpo_epochs),
           "--seed", str(base_args.seed + 1000 + int(trial_dir.name.split('_')[-1])),
           "--workers", str(base_args.workers),
           "--device", str(base_args.device) if base_args.device is not None else "cuda",
           "--amp", str(base_args.amp),
           "--class_weight", str(base_args.class_weight),
           "--balance_sampler", str(base_args.balance_sampler),
           "--freeze_until", str(base_args.freeze_until),
           "--loss", str(base_args.loss),
           "--alpha", str(base_args.alpha) if base_args.alpha is not None else "None",
           "--early_stop", str(min(base_args.early_stop, max(5, base_args.hpo_epochs // 2))),
           "--ema", str(base_args.ema),
           "--ema_decay", str(base_args.ema_decay),
           "--aug", str(base_args.aug),
           "--tta_val", str(base_args.tta_val),
           "--accum_steps", str(base_args.accum_steps),
           "--gradcam", "0",
           "--delay_reweight_epochs", str(min(base_args.delay_reweight_epochs, max(1, base_args.hpo_epochs // 3))),
           "--learn_gem_p", str(base_args.learn_gem_p),
           "--hpo", "0"]
    if base_args.alpha is None:
        # remove placeholder for argparse float None issue
        idx = cmd.index("--alpha")
        del cmd[idx:idx+2]
    for k, v in trial_params.items():
        cmd.extend([f"--{k}", str(v)])
    return cmd

def run_hpo_random(args):
    hpo_root = Path(args.save_dir) / "hpo_runs"
    hpo_root.mkdir(parents=True, exist_ok=True)
    results = []
    print(f"[HPO] start random search: trials={args.hpo_trials}, epochs_per_trial={args.hpo_epochs}")
    for trial_idx in range(args.hpo_trials):
        trial_params = sample_hparams_random(args, trial_idx)
        trial_dir = hpo_root / f"trial_{trial_idx:03d}"
        trial_dir.mkdir(parents=True, exist_ok=True)
        log_path = trial_dir / "train.log"
        cmd = _build_subprocess_cmd(args, trial_dir, trial_params)
        print(f"[HPO] trial {trial_idx+1}/{args.hpo_trials} params={trial_params}")
        with open(log_path, 'w', encoding='utf-8') as f:
            proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, text=True)
        best_f1 = _extract_best_f1_from_log(log_path)
        row = {
            'trial': trial_idx,
            'returncode': int(proc.returncode),
            'best_macroF1': None if best_f1 is None else float(best_f1),
            'trial_dir': str(trial_dir),
            **trial_params,
        }
        results.append(row)
        print(f"[HPO] trial {trial_idx+1} done returncode={proc.returncode} best_macroF1={best_f1}")

    valid = [r for r in results if r['returncode'] == 0 and r['best_macroF1'] is not None]
    results_sorted = sorted(valid, key=lambda x: x['best_macroF1'], reverse=True)

    summary_json = hpo_root / 'hpo_summary.json'
    summary_csv = hpo_root / 'hpo_summary.csv'
    with open(summary_json, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    csv_cols = ['trial', 'best_macroF1', 'returncode', 'lr', 'la_tau', 'gamma', 'label_smoothing',
                'balanced_batch', 'geom_edge', 'geom_dist', 'se', 'aspp', 'gem_p', 'trial_dir']
    with open(summary_csv, 'w', encoding='utf-8') as f:
        f.write(','.join(csv_cols) + '\n')
        for r in results:
            vals = [str(r.get(c, '')) for c in csv_cols]
            f.write(','.join(vals) + '\n')

    if not results_sorted:
        print(f"[HPO] no successful trial. summary saved to {summary_json}")
        return

    best = results_sorted[0]
    best_json = hpo_root / 'best_hparams.json'
    with open(best_json, 'w', encoding='utf-8') as f:
        json.dump(best, f, indent=2, ensure_ascii=False)

    print(f"[HPO] best_macroF1={best['best_macroF1']:.4f} at trial={best['trial']}")
    print(f"[HPO] best params: lr={best['lr']}, la_tau={best['la_tau']}, gamma={best['gamma']}, label_smoothing={best['label_smoothing']}, balanced_batch={best['balanced_batch']}, geom_edge={best['geom_edge']}, geom_dist={best['geom_dist']}, se={best['se']}, aspp={best['aspp']}, gem_p={best['gem_p']}")
    print(f"[HPO] summary csv: {summary_csv}")
    print(f"[HPO] best json: {best_json}")

    if args.hpo_refit_best == 1:
        refit_dir = Path(args.save_dir) / 'best_refit'
        refit_dir.mkdir(parents=True, exist_ok=True)
        refit_params = {k: best[k] for k in ['lr', 'la_tau', 'gamma', 'label_smoothing', 'balanced_batch', 'geom_edge', 'geom_dist', 'se', 'aspp', 'gem_p']}
        refit_cmd = [sys.executable, os.path.abspath(sys.argv[0]),
                     '--merged_root', str(args.merged_root),
                     '--save_dir', str(refit_dir),
                     '--img_size', str(args.img_size),
                     '--rgb', str(args.rgb),
                     '--bs', str(args.bs),
                     '--epochs', str(args.epochs),
                     '--lr', str(refit_params['lr']),
                     '--seed', str(args.seed),
                     '--workers', str(args.workers),
                     '--amp', str(args.amp),
                     '--class_weight', str(args.class_weight),
                     '--balance_sampler', str(args.balance_sampler),
                     '--balanced_batch', str(refit_params['balanced_batch']),
                     '--freeze_until', str(args.freeze_until),
                     '--label_smoothing', str(refit_params['label_smoothing']),
                     '--loss', str(args.loss),
                     '--gamma', str(refit_params['gamma']),
                     '--early_stop', str(args.early_stop),
                     '--ema', str(args.ema),
                     '--ema_decay', str(args.ema_decay),
                     '--aug', str(args.aug),
                     '--tta_val', str(args.tta_val),
                     '--geom_edge', str(refit_params['geom_edge']),
                     '--geom_dist', str(refit_params['geom_dist']),
                     '--accum_steps', str(args.accum_steps),
                     '--la_tau', str(refit_params['la_tau']),
                     '--gradcam', str(args.gradcam),
                     '--gradcam_samples', str(args.gradcam_samples),
                     '--gradcam_dir', str(args.gradcam_dir),
                     '--delay_reweight_epochs', str(args.delay_reweight_epochs),
                     '--aspp', str(refit_params['aspp']),
                     '--se', str(refit_params['se']),
                     '--gem_p', str(refit_params['gem_p']),
                     '--learn_gem_p', str(args.learn_gem_p),
                     '--hpo', '0']
        if args.device is not None:
            refit_cmd.extend(['--device', str(args.device)])
        if args.alpha is not None:
            refit_cmd.extend(['--alpha', str(args.alpha)])
        print(f"[HPO] refit best config to: {refit_dir}")
        subprocess.run(refit_cmd)

def plot_loss_curves(train_losses, val_losses, save_path):
    epochs = np.arange(1, len(train_losses) + 1)
    plt.figure(figsize=(7, 5))
    plt.plot(epochs, train_losses, label='train loss')
    plt.plot(epochs, val_losses, label='val loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Curves')
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()

def main():
    num_classes = 3
    class_names = ["class0", "class1", "class2"]
    ap = argparse.ArgumentParser()
    ap.add_argument("--merged_root", required=True)
    ap.add_argument("--save_dir", required=True)
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--rgb", type=int, default=1)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=2025)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--amp", type=int, default=1)
    ap.add_argument("--class_weight", type=str, default="auto")
    ap.add_argument("--balance_sampler", type=int, default=1)
    ap.add_argument("--balanced_batch", type=int, default=1)
    ap.add_argument("--freeze_until", type=int, default=0)
    ap.add_argument("--label_smoothing", type=float, default=0.05)
    ap.add_argument("--loss", type=str, default="cb_focal", choices=["ce", "focal", "cb_focal"])
    ap.add_argument("--gamma", type=float, default=2.0)
    ap.add_argument("--alpha", type=float, default=None)
    ap.add_argument("--early_stop", type=int, default=50)
    ap.add_argument("--ema", type=int, default=1)
    ap.add_argument("--ema_decay", type=float, default=0.999)
    ap.add_argument("--aug", type=int, default=1)
    ap.add_argument("--tta_val", type=int, default=1)
    ap.add_argument("--geom_edge", type=int, default=1)
    ap.add_argument("--geom_dist", type=int, default=1)
    ap.add_argument("--accum_steps", type=int, default=1)
    ap.add_argument("--la_tau", type=float, default=1.0)
    ap.add_argument("--gradcam", type=int, default=0)
    ap.add_argument("--gradcam_samples", type=int, default=24)
    ap.add_argument("--gradcam_dir", type=str, default="")
    ap.add_argument("--delay_reweight_epochs", type=int, default=8)
    ap.add_argument("--aspp", type=int, default=1)
    ap.add_argument("--se", type=int, default=1)
    ap.add_argument("--gem_p", type=float, default=3.0)
    ap.add_argument("--learn_gem_p", type=int, default=0)
    ap.add_argument("--hpo", type=int, default=0, help="1=啟用自動 hyperparameter optimization")
    ap.add_argument("--hpo_trials", type=int, default=12, help="HPO trial 數")
    ap.add_argument("--hpo_epochs", type=int, default=12, help="每個 HPO trial 的 epoch 數")
    ap.add_argument("--hpo_seed", type=int, default=2026, help="HPO 取樣 seed")
    ap.add_argument("--hpo_refit_best", type=int, default=0, help="HPO 結束後是否用最佳參數重跑完整訓練")
    args = ap.parse_args()

    if args.hpo == 1:
        os.makedirs(args.save_dir, exist_ok=True)
        run_hpo_random(args)
        return

    set_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)
    device = _infer_device(args.device)
    use_cuda = device.type == "cuda"
    if use_cuda:
        torch.backends.cudnn.benchmark = True

    ds_tr = SingleCellFromMerged(args.merged_root, split="train", img_size=args.img_size, use_rgb=bool(args.rgb))
    ds_va = SingleCellFromMerged(args.merged_root, split="val", img_size=args.img_size, use_rgb=bool(args.rgb))

    sample_x, _ = ds_tr[0]
    if isinstance(sample_x, np.ndarray):
        sample_x = torch.from_numpy(sample_x)
    if sample_x.ndim != 3:
        raise ValueError(f"Expect sample_x as (C,H,W), but got shape={sample_x.shape}")
    orig_ch = sample_x.shape[0]
    if orig_ch < 2:
        raise ValueError(f"Expect at least 2 channels (image + mask), but got {orig_ch}")

    base_in = orig_ch
    added_geom = (1 if args.geom_edge == 1 else 0) + (1 if args.geom_dist == 1 else 0)
    in_ch = base_in + added_geom

    print(f"[INFO] orig_ch={orig_ch}, base_in={base_in}, added_geom={added_geom}, in_ch={in_ch}")

    model = UNetEncoderClassifier(
        in_ch=in_ch,
        base_in=base_in,
        num_classes=NUM_CLASSES,
        drop=0.2,
        width=64,
        use_se=bool(args.se),
        use_aspp=bool(args.aspp),
        gem_p_init=args.gem_p,
        learn_gem_p=bool(args.learn_gem_p)
    ).to(device)
    model = model.to(memory_format=torch.channels_last)

    counts, ys_tr = _class_counts(ds_tr, NUM_CLASSES)
    if args.class_weight == "auto":
        cls_w = torch.tensor([sum(counts)/max(1, c) for c in counts], dtype=torch.float32, device=device)
        cls_w = cls_w / (cls_w.mean().clamp(min=1e-6))
    else:
        cls_w = None

    if args.loss == "cb_focal":
        cb_w = class_balanced_weights(counts, beta=0.9999, device=device)
        criterion_main = FocalLoss(weight=cb_w, gamma=args.gamma, alpha=args.alpha, label_smoothing=0.0)
    elif args.loss == "focal":
        criterion_main = FocalLoss(weight=cls_w, gamma=args.gamma, alpha=args.alpha, label_smoothing=args.label_smoothing)
    else:
        criterion_main = nn.CrossEntropyLoss(weight=cls_w, label_smoothing=args.label_smoothing)

    criterion_warm = nn.CrossEntropyLoss(weight=None, label_smoothing=min(args.label_smoothing, 0.02))

    total_n = max(1, sum(counts))
    priors = torch.tensor([c/total_n for c in counts], dtype=torch.float32, device=device)

    loader_kwargs = dict(num_workers=max(0, args.workers), pin_memory=use_cuda)
    if args.workers > 0:
        loader_kwargs.update(persistent_workers=True, prefetch_factor=4)

    if args.balanced_batch == 1:
        batch_sampler = BalancedBatchSampler(ys_tr, batch_size=args.bs, num_classes=NUM_CLASSES)
        dl_tr = DataLoader(ds_tr, batch_sampler=batch_sampler, **loader_kwargs)
    else:
        dl_tr = DataLoader(ds_tr, batch_size=args.bs, shuffle=True, **loader_kwargs)
    dl_va = DataLoader(ds_va, batch_size=args.bs, shuffle=False, **loader_kwargs)

    base_lr = args.lr
    decay, nodecay, head_decay, head_nodecay = [], [], [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        is_head = ("head" in n)
        is_nodecay = (p.dim() == 1) or n.endswith(".bias") or ("bn" in n) or ("norm" in n)
        if is_head:
            (head_nodecay if is_nodecay else head_decay).append(p)
        else:
            (nodecay if is_nodecay else decay).append(p)
    param_groups = [
        {"params": decay,        "lr": base_lr,     "weight_decay": 1e-4},
        {"params": nodecay,      "lr": base_lr,     "weight_decay": 0.0},
        {"params": head_decay,   "lr": base_lr*2.0, "weight_decay": 1e-4},
        {"params": head_nodecay, "lr": base_lr*2.0, "weight_decay": 0.0},
    ]
    opt = optim.AdamW(param_groups)

    steps_per_epoch = len(dl_tr)
    accum = max(1, args.accum_steps)
    opt_steps_per_epoch = math.ceil(steps_per_epoch / accum)
    total_steps = opt_steps_per_epoch * args.epochs
    warmup = max(1, int(0.10 * total_steps))

    def lr_lambda(step):
        if step < warmup:
            return step / warmup
        progress = (step - warmup) / max(1, (total_steps - warmup))
        return 0.5 * (1 + math.cos(math.pi * progress))

    sched = optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_lambda)

    scaler = torch.amp.GradScaler('cuda', enabled=(args.amp == 1 and use_cuda))
    ema = EMA(model, decay=args.ema_decay) if args.ema == 1 else None

    best_f1, best_ep = -1.0, -1
    no_improve = 0
    global_step = 0
    train_loss_history = []
    val_loss_history = []

    def prep_input(xb):
        xb = add_geom_channels(
            xb,
            mask_index=base_in-1,
            rgb_mode=bool(args.rgb),
            use_edge=bool(args.geom_edge),
            use_dist=bool(args.geom_dist)
        )
        xb = xb[:, :in_ch, ...]
        if args.rgb == 1 and xb.size(1) >= 3:
            mean = torch.tensor([0.485, 0.456, 0.406], device=xb.device, dtype=xb.dtype).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], device=xb.device, dtype=xb.dtype).view(1, 3, 1, 1)
            xb[:, :3] = (xb[:, :3] - mean) / std
        xb = torch.nan_to_num(xb, nan=0.0, posinf=1.0, neginf=0.0)
        return xb

    def has_bad_tensor(x):
        return (not torch.isfinite(x).all().item())

    print(f"[INFO] steps_per_epoch={steps_per_epoch}, opt_steps_per_epoch={opt_steps_per_epoch}, total_steps={total_steps}, warmup={warmup}")

    for ep in range(1, args.epochs+1):
        model.train()
        run_loss = 0.0
        valid_train_batches = 0
        t0 = time.time()
        opt.zero_grad(set_to_none=True)

        criterion = criterion_warm if ep <= args.delay_reweight_epochs else criterion_main

        for bi, (xb, yb) in enumerate(dl_tr, 1):
            xb = xb.to(device, non_blocking=use_cuda).to(memory_format=torch.channels_last)
            yb = yb.to(device, non_blocking=use_cuda)

            xb = prep_input(xb)
            if args.aug == 1:
                xb = random_augment(xb)

            with torch.autocast('cuda', enabled=(args.amp == 1 and use_cuda)):
                logits = model(xb)
                if args.la_tau > 0:
                    logits = logits - args.la_tau * priors.clamp(min=1e-6).log().to(dtype=logits.dtype, device=logits.device)
                loss = criterion(logits, yb) / accum

            if has_bad_tensor(logits) or has_bad_tensor(loss):
                print(f"[WARN] skip bad batch at epoch={ep} iter={bi}")
                opt.zero_grad(set_to_none=True)
                continue

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
            valid_train_batches += 1
            if bi % 200 == 0:
                cur_lr = sched.get_last_lr()[0]
                print(f"[E{ep:02d}] {bi}/{steps_per_epoch} loss={run_loss/valid_train_batches:.4f} lr={cur_lr:.2e} gs={global_step}")

        if (steps_per_epoch % accum) != 0:
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            if ema is not None:
                ema.update(model)
            sched.step()
            global_step += 1

        train_epoch_loss = run_loss / max(1, valid_train_batches)
        train_loss_history.append(float(train_epoch_loss))

        def eval_with(model_for_eval):
            model_for_eval.eval()
            y_true, y_pred = [], []
            val_loss_sum = 0.0
            val_count = 0
            with torch.no_grad():
                for xb, yb in dl_va:
                    xb = xb.to(device, non_blocking=use_cuda).to(memory_format=torch.channels_last)
                    yb = yb.to(device, non_blocking=use_cuda)

                    def forward_once(x):
                        x = prep_input(x)
                        return model_for_eval(x)

                    x0 = xb
                    if args.tta_val == 2:
                        logits_list = []
                        for k in [0, 1, 2, 3]:
                            xk = torch.rot90(x0, k, dims=[2, 3])
                            logits_list.append(forward_once(xk))
                            logits_list.append(forward_once(torch.flip(xk, dims=[3])))
                        logits = torch.stack(logits_list, dim=0).mean(0)
                    elif args.tta_val == 1:
                        logits_list = [
                            forward_once(x0),
                            forward_once(torch.flip(x0, dims=[3])),
                            forward_once(torch.flip(x0, dims=[2])),
                            forward_once(torch.flip(torch.flip(x0, dims=[2]), dims=[3])),
                        ]
                        logits = torch.stack(logits_list, dim=0).mean(0)
                    else:
                        logits = forward_once(x0)

                    val_logits_for_loss = logits
                    if args.la_tau > 0:
                        logits = logits - args.la_tau * priors.clamp(min=1e-6).log().to(dtype=logits.dtype, device=logits.device)
                    val_loss = criterion(val_logits_for_loss, yb)
                    val_loss_sum += float(val_loss.item()) * yb.size(0)
                    val_count += int(yb.size(0))

                    pred = logits.argmax(1)
                    y_true += yb.tolist()
                    y_pred += pred.tolist()

            cm = confusion_matrix(y_true, y_pred, num_classes=NUM_CLASSES)
            macro_f1, per_cls = macro_f1_from_cm(cm)
            macro_precision, macro_recall, precision_per_cls, recall_per_cls = precision_recall_from_cm(cm)
            avg_val_loss = val_loss_sum / max(1, val_count)
            return {
                "macro_f1": macro_f1,
                "per_f1": per_cls,
                "cm": cm,
                "macro_precision": macro_precision,
                "macro_recall": macro_recall,
                "per_precision": precision_per_cls,
                "per_recall": recall_per_cls,
                "val_loss": avg_val_loss,
            }

        if ema is not None:
            ema.apply_shadow(model)
            eval_res = eval_with(model)
            ema.restore(model)
        else:
            eval_res = eval_with(model)

        macro_f1 = eval_res["macro_f1"]
        per_cls = eval_res["per_f1"]
        cm = eval_res["cm"]
        macro_precision = eval_res["macro_precision"]
        macro_recall = eval_res["macro_recall"]
        per_precision = eval_res["per_precision"]
        per_recall = eval_res["per_recall"]
        val_epoch_loss = eval_res["val_loss"]
        val_loss_history.append(float(val_epoch_loss))

        print(f"[VAL] E{ep:02d} loss={val_epoch_loss:.4f} macroF1={macro_f1:.3f} macroP={macro_precision:.3f} macroR={macro_recall:.3f}")
        print(f"[VAL] F1 per_class={[f'{x:.3f}' for x in per_cls]}")
        print(f"[VAL] P  per_class={[f'{x:.3f}' for x in per_precision]}")
        print(f"[VAL] R  per_class={[f'{x:.3f}' for x in per_recall]}")
        print("CM rows=gt cols=pred:")
        for r in cm:
            print(r.tolist())

        improved = macro_f1 > best_f1 + 1e-6
        if improved:
            best_f1, best_ep = macro_f1, ep
            no_improve = 0
            save_p = os.path.join(args.save_dir, "best_unet_cell_cls_3cls.pt")

            state_to_save = model.state_dict()
            if ema is not None:
                with torch.no_grad():
                    for name, _ in model.named_parameters():
                        if name in ema.shadow:
                            state_to_save[name] = ema.shadow[name].detach().clone()

            payload = {
                "state_dict": state_to_save,
                "best_f1": best_f1,
                "epoch": ep,
                "ema": (ema is not None),
                "in_ch": in_ch,
                "num_classes": NUM_CLASSES,
                "priors": [c/total_n for c in counts],
                "class_names": class_names,
                "args": vars(args),
                "train_loss_history": train_loss_history,
                "val_loss_history": val_loss_history,
            }
            torch.save(payload, save_p)
            print(f"[SAVE] best at epoch {ep} (macroF1={best_f1:.3f})")
        else:
            no_improve += 1

        curves_png = os.path.join(args.save_dir, "loss_curve.png")
        plot_loss_curves(train_loss_history, val_loss_history, curves_png)

        loss_csv = os.path.join(args.save_dir, "loss_history.csv")
        with open(loss_csv, "w", encoding="utf-8") as f:
            f.write("epoch,train_loss,val_loss\n")
            for i, (tr_l, va_l) in enumerate(zip(train_loss_history, val_loss_history), start=1):
                f.write(f"{i},{tr_l:.6f},{va_l:.6f}\n")

        dt = time.time() - t0
        print(f"[E{ep:02d}] epoch_time={dt/60:.1f} min  (no_improve={no_improve}/{args.early_stop})")
        print(f"[CURVE] saved to {curves_png}")

        if args.early_stop > 0 and no_improve >= args.early_stop:
            print("[EARLY STOP] no improvement, stop training.")
            break

    if args.gradcam == 1:
        gc_dir = args.gradcam_dir if args.gradcam_dir else os.path.join(args.save_dir, "gradcam")
        print(f"[Grad-CAM] saving examples to: {gc_dir}")
        if ema is not None:
            ema.apply_shadow(model)
            save_gradcam_examples(model, dl_va, device, use_cuda, prep_input, args, gc_dir, class_names=class_names, max_samples=args.gradcam_samples)
            ema.restore(model)
        else:
            save_gradcam_examples(model, dl_va, device, use_cuda, prep_input, args, gc_dir, class_names=class_names, max_samples=args.gradcam_samples)

    print(f"[DONE] best macroF1={best_f1:.3f} @ epoch {best_ep}")

if __name__ == "__main__":
    main()


