import os, sys, argparse, math, time, random, json, csv, subprocess, shutil
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Sampler

from data_from_merged_2 import SingleCellFromMerged
from metrics import confusion_matrix, macro_f1_from_cm
from utils import set_seed
from torchvision.models import densenet161

# 放在檔案頂端的 import（與 DenseNet 並存即可）
try:
    from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
    _HAS_CNX_WEIGHTS = True
except Exception:
    from torchvision.models import convnext_tiny
    _HAS_CNX_WEIGHTS = False


class ConvNeXtCellClassifier(nn.Module):
    """
    - 完全不同於 DenseNet/ViT 的 ConvNeXt 架構（Conv-only、Large-kernel DW Conv）
    - in_ch 可不是 3：自動改第一層 patchify conv 的輸入通道
    - 輸出接口與你的 DenseNetCellClassifier 相同：forward(x)->(B, num_classes)
    - 保留屬性名稱：features / head（你的 optimizer 用 "head" 區分學習率會照樣運作）
    - freeze_until：>0 會凍結前兩個 stage（含 stem）
    """
    def __init__(self, in_ch=4, num_classes=3, pretrained=True, freeze_until=0, drop=0.2):
        super().__init__()

        if _HAS_CNX_WEIGHTS and pretrained:
            m = convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        else:
            m = convnext_tiny(weights=("IMAGENET1K_V1" if pretrained else None))

        stem_conv = m.features[0][0]
        if in_ch != stem_conv.in_channels:
            new = nn.Conv2d(
                in_ch, stem_conv.out_channels,
                kernel_size=stem_conv.kernel_size,
                stride=stem_conv.stride,
                padding=stem_conv.padding,
                bias=(stem_conv.bias is not None)
            )
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
            for _, p in self.features.named_parameters():
                p.requires_grad = False

    def forward(self, x):
        f = self.features(x)
        g = self.pool(f).flatten(1)
        g = self.norm(g)
        return self.head(g)


# ==================== 全域設定：3 類 ====================
NUM_CLASSES = 3  # 0:1to1, 1:1to2, 2:1to4(含k=3或4)


def build_ema_full_state_dict(model: nn.Module, ema):
    sd = model.state_dict()
    if ema is None:
        return sd
    with torch.no_grad():
        for name, _ in model.named_parameters():
            if name in ema.shadow:
                sd[name] = ema.shadow[name].detach().clone()
    return sd


# -------------------- DenseNet-161 classifier --------------------
class DenseNetCellClassifier(nn.Module):
    def __init__(self, in_ch=4, num_classes=NUM_CLASSES, pretrained=True, freeze_until=0, drop=0.2):
        super().__init__()
        m = densenet161(weights="IMAGENET1K_V1" if pretrained else None)

        conv0 = m.features.conv0
        if in_ch != 3:
            w = conv0.weight
            new = nn.Conv2d(in_ch, conv0.out_channels, kernel_size=conv0.kernel_size,
                            stride=conv0.stride, padding=conv0.padding, bias=False)
            with torch.no_grad():
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
            nn.Linear(256, num_classes)
        )

        if freeze_until > 0:
            for name, p in self.features.named_parameters():
                p.requires_grad = False
                if "denseblock2" in name:
                    break

    def forward(self, x):
        f = self.features(x)
        f = self.norm(f)
        f = self.relu(f)
        g = self.pool(f).flatten(1)
        return self.head(g)


# -------------------- Losses --------------------
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
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


def class_balanced_weights(counts, beta=0.9999, device=None):
    counts = np.array([max(1, c) for c in counts], dtype=np.float32)
    eff_num = 1.0 - np.power(beta, counts)
    weights = (1.0 - beta) / eff_num
    weights = weights / weights.sum() * len(counts)
    t = torch.tensor(weights, dtype=torch.float32)
    return t if device is None else t.to(device)


# -------------------- EMA --------------------
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


# -------------------- misc utils --------------------
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
    cnt = [0] * num_classes
    for y in ys:
        cnt[y] += 1
    return cnt, ys


class BalancedBatchSampler(Sampler):
    def __init__(self, labels, batch_size, num_classes=NUM_CLASSES):
        self.labels = np.array(labels, dtype=np.int64)
        self.batch_size = int(batch_size)
        self.num_classes = num_classes
        self.m = max(1, self.batch_size // self.num_classes)
        self.class_indices = {c: np.where(self.labels == c)[0].tolist() for c in range(num_classes)}
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


# -------------------- simple GPU aug --------------------
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
        x[:, :img_ch, y0:y0 + h, x0:x0 + w] = val
    return x


# -------------------- geometry channels from mask --------------------
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


def add_geom_channels(x, rgb_mode=True, use_edge=True, use_dist=False):
    mask = x[:, -1:, ...]
    extra = []
    if use_edge:
        extra.append(make_edge_from_mask(mask))
    if use_dist:
        extra.append(make_pseudodist_from_mask(mask))
    if len(extra):
        x = torch.cat([x, *extra], dim=1)
    return x


# -------------------- plotting & history --------------------
def save_history(history, save_dir):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    with open(save_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

    epochs = history["epoch"]

    plt.figure(figsize=(7, 5))
    plt.plot(epochs, history["train_loss"], label="train_loss")
    plt.plot(epochs, history["val_loss"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Train / Val Loss Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_dir / "loss_curve.png", dpi=200)
    plt.close()

    plt.figure(figsize=(7, 5))
    plt.plot(epochs, history["train_acc"], label="train_acc")
    plt.plot(epochs, history["val_acc"], label="val_acc")
    plt.plot(epochs, history["val_macro_f1"], label="val_macro_f1")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.title("Accuracy / Macro-F1 Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_dir / "accuracy_curve.png", dpi=200)
    plt.close()



# -------------------- HPO --------------------
def _sample_hparams(rng):
    return {
        "lr": rng.choice([1e-4, 2e-4, 3e-4, 5e-4]),
        "gamma": rng.choice([1.5, 2.0, 2.5]),
        "label_smoothing": rng.choice([0.0, 0.03, 0.05, 0.1]),
        "balanced_batch": rng.choice([0, 1]),
        "geom_edge": rng.choice([0, 1]),
        "geom_dist": rng.choice([0, 1]),
        "aug": rng.choice([0, 1]),
    }

def run_hpo(args):
    save_dir = Path(args.save_dir)
    hpo_dir = save_dir / "hpo_runs"
    hpo_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    tried = []
    best_row = None

    for trial in range(1, args.hpo_trials + 1):
        hp = _sample_hparams(rng)
        trial_dir = hpo_dir / f"trial_{trial:03d}"
        trial_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            sys.executable, os.path.abspath(__file__),
            "--merged_root", args.merged_root,
            "--save_dir", str(trial_dir),
            "--img_size", str(args.img_size),
            "--rgb", str(args.rgb),
            "--bs", str(args.bs),
            "--epochs", str(args.hpo_epochs),
            "--lr", str(hp["lr"]),
            "--seed", str(args.seed + trial),
            "--workers", str(args.workers),
            "--amp", str(args.amp),
            "--class_weight", str(args.class_weight),
            "--balance_sampler", str(args.balance_sampler),
            "--balanced_batch", str(hp["balanced_batch"]),
            "--freeze_until", str(args.freeze_until),
            "--label_smoothing", str(hp["label_smoothing"]),
            "--loss", str(args.loss),
            "--gamma", str(hp["gamma"]),
            "--early_stop", str(min(args.early_stop, max(3, args.hpo_epochs // 2))),
            "--ema", str(args.ema),
            "--ema_decay", str(args.ema_decay),
            "--aug", str(hp["aug"]),
            "--tta_val", str(args.tta_val),
            "--geom_edge", str(hp["geom_edge"]),
            "--geom_dist", str(hp["geom_dist"]),
            "--accum_steps", str(args.accum_steps),
            "--gradcam_n", str(args.gradcam_n),
            "--save_gradcam", "0",
            "--hpo", "0",
        ]
        if args.device is not None:
            cmd.extend(["--device", str(args.device)])
        if args.alpha is not None:
            cmd.extend(["--alpha", str(args.alpha)])

        env = os.environ.copy()
        print(f"[HPO] trial {trial}/{args.hpo_trials} -> {hp}")
        log_path = trial_dir / "train.log"
        with open(log_path, "w", encoding="utf-8") as logf:
            proc = subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT, env=env)
        if proc.returncode != 0:
            row = {"trial": trial, **hp, "best_macroF1": None, "status": f"failed:{proc.returncode}"}
            tried.append(row)
            print(f"[HPO] trial {trial} failed, see {log_path}")
            continue

        hist_path = trial_dir / "history.json"
        if hist_path.exists():
            hist = json.loads(hist_path.read_text(encoding="utf-8"))
            best_f1 = max(hist.get("val_macro_f1", [float("-inf")]))
            best_ep = int(np.argmax(hist.get("val_macro_f1", [0.0])) + 1)
        else:
            best_f1 = float("-inf")
            best_ep = -1

        row = {"trial": trial, **hp, "best_macroF1": float(best_f1), "best_epoch": best_ep, "status": "ok"}
        tried.append(row)
        if best_row is None or row["best_macroF1"] > best_row["best_macroF1"]:
            best_row = row

    csv_path = save_dir / "hpo_summary.csv"
    json_path = save_dir / "hpo_summary.json"
    fields = ["trial", "lr", "gamma", "label_smoothing", "balanced_batch", "geom_edge", "geom_dist", "aug", "best_macroF1", "best_epoch", "status"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in tried:
            writer.writerow(row)
    json_path.write_text(json.dumps(tried, ensure_ascii=False, indent=2), encoding="utf-8")

    if best_row is None:
        print("[HPO] no successful trials.")
        return

    best_hp = {k: best_row[k] for k in ["lr", "gamma", "label_smoothing", "balanced_batch", "geom_edge", "geom_dist", "aug"]}
    (save_dir / "best_hparams.json").write_text(json.dumps(best_hp, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[HPO] best trial={best_row['trial']} macroF1={best_row['best_macroF1']:.4f} hp={best_hp}")

    if args.hpo_refit_best == 1:
        refit_dir = save_dir / "refit_best"
        refit_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable, os.path.abspath(__file__),
            "--merged_root", args.merged_root,
            "--save_dir", str(refit_dir),
            "--img_size", str(args.img_size),
            "--rgb", str(args.rgb),
            "--bs", str(args.bs),
            "--epochs", str(args.epochs),
            "--lr", str(best_hp["lr"]),
            "--seed", str(args.seed),
            "--workers", str(args.workers),
            "--amp", str(args.amp),
            "--class_weight", str(args.class_weight),
            "--balance_sampler", str(args.balance_sampler),
            "--balanced_batch", str(best_hp["balanced_batch"]),
            "--freeze_until", str(args.freeze_until),
            "--label_smoothing", str(best_hp["label_smoothing"]),
            "--loss", str(args.loss),
            "--gamma", str(best_hp["gamma"]),
            "--early_stop", str(args.early_stop),
            "--ema", str(args.ema),
            "--ema_decay", str(args.ema_decay),
            "--aug", str(best_hp["aug"]),
            "--tta_val", str(args.tta_val),
            "--geom_edge", str(best_hp["geom_edge"]),
            "--geom_dist", str(best_hp["geom_dist"]),
            "--accum_steps", str(args.accum_steps),
            "--gradcam_n", str(args.gradcam_n),
            "--save_gradcam", str(args.save_gradcam),
            "--hpo", "0",
        ]
        if args.device is not None:
            cmd.extend(["--device", str(args.device)])
        if args.alpha is not None:
            cmd.extend(["--alpha", str(args.alpha)])
        print(f"[HPO] refit best -> {refit_dir}")
        subprocess.run(cmd, env=os.environ.copy(), check=True)
# -------------------- Grad-CAM --------------------
class GradCAM:
    def __init__(self, model, target_module):
        self.model = model
        self.target_module = target_module
        self.activations = None
        self.gradients = None
        self.fwd_handle = target_module.register_forward_hook(self._forward_hook)
        self.bwd_handle = target_module.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, inputs, output):
        self.activations = output.detach()

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def remove(self):
        self.fwd_handle.remove()
        self.bwd_handle.remove()

    def generate(self, x, class_idx=None):
        self.model.zero_grad(set_to_none=True)
        logits = self.model(x)

        if class_idx is None:
            class_idx = logits.argmax(dim=1)
        elif isinstance(class_idx, int):
            class_idx = torch.full((x.size(0),), class_idx, device=x.device, dtype=torch.long)

        score = logits.gather(1, class_idx.view(-1, 1)).sum()
        score.backward(retain_graph=False)

        grads = self.gradients
        acts = self.activations
        weights = grads.mean(dim=(2, 3), keepdim=True)
        cam = (weights * acts).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)

        cam_min = cam.amin(dim=(2, 3), keepdim=True)
        cam_max = cam.amax(dim=(2, 3), keepdim=True)
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)
        return logits.detach(), cam.detach(), class_idx.detach()


def tensor_to_vis_image(x, use_rgb=True):
    x = x.detach().cpu().float()
    if use_rgb and x.size(0) >= 3:
        img = x[:3].permute(1, 2, 0).numpy()
        img = np.clip(img, 0.0, 1.0)
    else:
        g = x[0].numpy()
        g = np.clip(g, 0.0, 1.0)
        img = np.stack([g, g, g], axis=-1)
    return img


def save_gradcam_examples(model, gradcam, raw_batch, labels, prep_input_fn, save_path, use_rgb=True, class_names=None):
    model.eval()
    xb = raw_batch.clone()
    xb = prep_input_fn(xb)
    xb = xb.to(next(model.parameters()).device, non_blocking=True).to(memory_format=torch.channels_last)

    logits, cams, preds = gradcam.generate(xb)
    probs = torch.softmax(logits, dim=1).detach().cpu().numpy()
    cams = cams.cpu()

    n = xb.size(0)
    fig, axes = plt.subplots(n, 3, figsize=(12, 4 * n))
    if n == 1:
        axes = np.expand_dims(axes, axis=0)

    for i in range(n):
        raw_img = tensor_to_vis_image(raw_batch[i], use_rgb=use_rgb)
        cam = cams[i, 0].numpy()

        axes[i, 0].imshow(raw_img)
        axes[i, 0].set_title("Original")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(cam, cmap="jet")
        axes[i, 1].set_title("Grad-CAM")
        axes[i, 1].axis("off")

        axes[i, 2].imshow(raw_img)
        axes[i, 2].imshow(cam, cmap="jet", alpha=0.45)
        gt = int(labels[i])
        pred = int(preds[i].cpu())
        conf = float(probs[i, pred])

        if class_names is None:
            gt_name = str(gt)
            pred_name = str(pred)
        else:
            gt_name = class_names[gt]
            pred_name = class_names[pred]

        axes[i, 2].set_title(f"Overlay\nGT={gt_name}  Pred={pred_name}  p={conf:.3f}")
        axes[i, 2].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close(fig)


# -------------------- eval --------------------
def evaluate(model_for_eval, dl_va, criterion, device, use_cuda, args, in_ch, prep_input):
    model_for_eval.eval()
    y_true, y_pred = [], []
    val_loss_sum = 0.0
    val_total = 0
    val_correct = 0

    vis_raw = None
    vis_labels = None

    with torch.no_grad():
        for step_idx, (xb, yb) in enumerate(dl_va, 1):
            if vis_raw is None:
                take_n = min(args.gradcam_n, xb.size(0))
                vis_raw = xb[:take_n].clone()
                vis_labels = yb[:take_n].clone()

            xb = xb.to(device, non_blocking=use_cuda).to(memory_format=torch.channels_last)
            yb = yb.to(device, non_blocking=use_cuda)

            def forward_once(x):
                x = prep_input(x)
                x = x[:, :in_ch, ...]
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
                logits_list = []
                logits_list.append(forward_once(x0))
                logits_list.append(forward_once(torch.flip(x0, dims=[3])))
                logits_list.append(forward_once(torch.flip(x0, dims=[2])))
                logits_list.append(forward_once(torch.flip(torch.flip(x0, dims=[2]), dims=[3])))
                logits = torch.stack(logits_list, dim=0).mean(0)
            else:
                logits = forward_once(x0)

            loss = criterion(logits, yb)
            pred = logits.argmax(1)

            bs = yb.size(0)
            val_loss_sum += loss.item() * bs
            val_total += bs
            val_correct += (pred == yb).sum().item()

            y_true += yb.tolist()
            y_pred += pred.tolist()

    cm = confusion_matrix(y_true, y_pred, num_classes=NUM_CLASSES)
    macro_f1, per_cls = macro_f1_from_cm(cm)
    val_loss = val_loss_sum / max(1, val_total)
    val_acc = val_correct / max(1, val_total)
    return val_loss, val_acc, macro_f1, per_cls, cm, vis_raw, vis_labels


# -------------------- train --------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--merged_root", required=True)
    ap.add_argument("--save_dir", required=True)
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--rgb", type=int, default=1)               # 1->RGB+mask, 0->Gray+mask
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
    ap.add_argument("--tta_val", type=int, default=1)           # 1:4-TTA, 2:8-TTA
    ap.add_argument("--geom_edge", type=int, default=1)
    ap.add_argument("--geom_dist", type=int, default=1)
    ap.add_argument("--accum_steps", type=int, default=1)

    # 新增功能
    ap.add_argument("--gradcam_n", type=int, default=6, help="每次最佳模型更新時，輸出幾張 Grad-CAM 樣本")
    ap.add_argument("--save_gradcam", type=int, default=1, help="是否儲存 Grad-CAM 視覺化")
    ap.add_argument("--hpo", type=int, default=0, help="1: 啟動自動 hyperparameter optimization")
    ap.add_argument("--hpo_trials", type=int, default=8, help="HPO 要試幾組參數")
    ap.add_argument("--hpo_epochs", type=int, default=8, help="每組 trial 先跑幾個 epoch")
    ap.add_argument("--hpo_refit_best", type=int, default=1, help="搜尋完後是否用最佳參數再正式重跑")
    args = ap.parse_args()

    if args.hpo == 1:
        run_hpo(args)
        return

    set_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)
    device = _infer_device(args.device)
    use_cuda = device.type == "cuda"
    if use_cuda:
        torch.backends.cudnn.benchmark = True

    # ---- datasets ----
    ds_tr = SingleCellFromMerged(args.merged_root, split="train",
                                 img_size=args.img_size, use_rgb=bool(args.rgb))
    ds_va = SingleCellFromMerged(args.merged_root, split="val",
                                 img_size=args.img_size, use_rgb=bool(args.rgb))

    x0, _ = ds_tr[0]
    orig_ch = x0.shape[0]
    print(f"[INFO] orig_ch={orig_ch}")

    base_in = orig_ch
    added = (1 if args.geom_edge == 1 else 0) + (1 if args.geom_dist == 1 else 0)
    in_ch = base_in + added
    print(f"[INFO] base_in={base_in}, added_geom={added}, in_ch={in_ch}")

    model = DenseNetCellClassifier(
        in_ch=in_ch,
        num_classes=NUM_CLASSES,
        pretrained=True,
        freeze_until=args.freeze_until
    ).to(device)
    model = model.to(memory_format=torch.channels_last)

    counts, ys_tr = _class_counts(ds_tr, NUM_CLASSES)
    if args.class_weight == "auto":
        cls_w = torch.tensor([sum(counts) / max(1, c) for c in counts], dtype=torch.float32, device=device)
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
        {"params": head_decay,   "lr": base_lr * 5.0, "weight_decay": 1e-4},
        {"params": head_nodecay, "lr": base_lr * 5.0, "weight_decay": 0.0},
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

    history = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
        "val_macro_f1": [],
        "lr": [],
    }

    class_names = ["1to1", "1to2", "1to4"]

    def prep_input(xb):
        xb = add_geom_channels(
            xb,
            rgb_mode=bool(args.rgb),
            use_edge=bool(args.geom_edge),
            use_dist=bool(args.geom_dist)
        )
        xb = xb[:, :in_ch, ...]
        if args.rgb == 1 and xb.size(1) >= 3:
            mean = torch.tensor([0.485, 0.456, 0.406], device=xb.device, dtype=xb.dtype).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], device=xb.device, dtype=xb.dtype).view(1, 3, 1, 1)
            xb[:, :3] = (xb[:, :3] - mean) / std
        return xb

    gradcam = GradCAM(model, model.features)

    print(f"[INFO] steps_per_epoch={steps_per_epoch}, opt_steps_per_epoch={opt_steps_per_epoch}, total_steps={total_steps}, warmup={warmup}")

    for ep in range(1, args.epochs + 1):
        model.train()
        run_loss = 0.0
        train_correct = 0
        train_total = 0
        t0 = time.time()
        opt.zero_grad(set_to_none=True)

        for bi, (xb, yb) in enumerate(dl_tr, 1):
            xb = xb.to(device, non_blocking=use_cuda).to(memory_format=torch.channels_last)
            yb = yb.to(device, non_blocking=use_cuda)

            xb = prep_input(xb)
            if args.aug == 1:
                xb = random_augment(xb)

            with torch.autocast('cuda', enabled=(args.amp == 1 and use_cuda)):
                logits = model(xb)
                loss_raw = criterion(logits, yb)
                loss = loss_raw / accum

            scaler.scale(loss).backward()

            pred = logits.argmax(1)
            bs = yb.size(0)
            run_loss += loss_raw.item() * bs
            train_total += bs
            train_correct += (pred == yb).sum().item()

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

            if bi % 200 == 0:
                cur_lr = sched.get_last_lr()[0]
                print(
                    f"[E{ep:02d}] {bi}/{steps_per_epoch} "
                    f"loss={run_loss / max(1, train_total):.4f} "
                    f"acc={train_correct / max(1, train_total):.4f} "
                    f"lr={cur_lr:.2e} gs={global_step}"
                )

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

        train_loss = run_loss / max(1, train_total)
        train_acc = train_correct / max(1, train_total)

        if ema is not None:
            ema.apply_shadow(model)
            val_loss, val_acc, macro_f1, per_cls, cm, vis_raw, vis_labels = evaluate(
                model, dl_va, criterion, device, use_cuda, args, in_ch, prep_input
            )
            ema.restore(model)
        else:
            val_loss, val_acc, macro_f1, per_cls, cm, vis_raw, vis_labels = evaluate(
                model, dl_va, criterion, device, use_cuda, args, in_ch, prep_input
            )

        history["epoch"].append(ep)
        history["train_loss"].append(float(train_loss))
        history["val_loss"].append(float(val_loss))
        history["train_acc"].append(float(train_acc))
        history["val_acc"].append(float(val_acc))
        history["val_macro_f1"].append(float(macro_f1))
        history["lr"].append(float(sched.get_last_lr()[0]))
        save_history(history, args.save_dir)

        print(
            f"[VAL] E{ep:02d} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} macroF1={macro_f1:.4f} "
            f"per_class={[f'{x:.3f}' for x in per_cls]}"
        )
        print("CM rows=gt cols=pred:")
        for r in cm:
            print(r.tolist())

        improved = macro_f1 > best_f1 + 1e-6
        if improved:
            best_f1, best_ep = macro_f1, ep
            no_improve = 0
            save_p = os.path.join(args.save_dir, "best_densenet_cell_cls_3cls.pt")

            state_to_save = build_ema_full_state_dict(model, ema)
            payload = {
                "state_dict": state_to_save,
                "best_f1": best_f1,
                "epoch": ep,
                "ema": (ema is not None),
                "in_ch": in_ch,
                "num_classes": NUM_CLASSES,
                "args": vars(args),
                "history": history,
            }
            torch.save(payload, save_p)
            print(f"[SAVE] best at epoch {ep} (macroF1={best_f1:.3f})")

            if args.save_gradcam == 1 and vis_raw is not None:
                gradcam_dir = Path(args.save_dir) / "gradcam"
                gradcam_dir.mkdir(parents=True, exist_ok=True)

                if ema is not None:
                    ema.apply_shadow(model)
                save_gradcam_examples(
                    model=model,
                    gradcam=gradcam,
                    raw_batch=vis_raw,
                    labels=vis_labels,
                    prep_input_fn=prep_input,
                    save_path=gradcam_dir / f"gradcam_epoch_{ep:03d}.png",
                    use_rgb=bool(args.rgb),
                    class_names=class_names
                )
                if ema is not None:
                    ema.restore(model)
                print(f"[SAVE] Grad-CAM -> {gradcam_dir / f'gradcam_epoch_{ep:03d}.png'}")
        else:
            no_improve += 1

        dt = time.time() - t0
        print(f"[E{ep:02d}] epoch_time={dt / 60:.1f} min  (no_improve={no_improve}/{args.early_stop})")

        if args.early_stop > 0 and no_improve >= args.early_stop:
            print("[EARLY STOP] no improvement, stop training.")
            break

    gradcam.remove()
    print(f"[DONE] best macroF1={best_f1:.3f} @ epoch {best_ep}")
    print(f"[DONE] curves saved to: {args.save_dir}/loss_curve.png and {args.save_dir}/accuracy_curve.png")


if __name__ == "__main__":
    main()