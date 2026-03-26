import os
import numpy as np
from scipy.optimize import linear_sum_assignment
from PIL import Image
import matplotlib.pyplot as plt

def compute_instance_overlap(gt_mask, pr_mask, iou_thresh=0.5):
    gt_labels = np.unique(gt_mask)
    pr_labels = np.unique(pr_mask)

    gt_labels = gt_labels[gt_labels != 0]
    pr_labels = pr_labels[pr_labels != 0]

    iou_matrix = np.zeros((len(gt_labels), len(pr_labels)))

    for i, gt_label in enumerate(gt_labels):
        gt_bin = (gt_mask == gt_label)
        for j, pr_label in enumerate(pr_labels):
            pr_bin = (pr_mask == pr_label)
            intersection = np.logical_and(gt_bin, pr_bin).sum()
            union = np.logical_or(gt_bin, pr_bin).sum()
            iou = intersection / union if union > 0 else 0
            iou_matrix[i, j] = iou

    matched_gt = set()
    matched_pr = set()

    gt_idx, pr_idx = linear_sum_assignment(-iou_matrix)

    for i, j in zip(gt_idx, pr_idx):
        if iou_matrix[i, j] >= iou_thresh:
            matched_gt.add(gt_labels[i])
            matched_pr.add(pr_labels[j])

    return {
        "gt_total": len(gt_labels),
        "pr_total": len(pr_labels),
        "matched": len(matched_gt),
        "miss": len(gt_labels) - len(matched_gt),
        "false_positive": len(pr_labels) - len(matched_pr)
    }

def compare_all_instance_masks(gt_dir, pr_dir, output_txt):
    pr_files = sorted([f for f in os.listdir(pr_dir) if f.endswith('_seg.npy')])
    results = []
    error_rates = []
    total_gt, total_pr, total_match, total_miss, total_fp = 0, 0, 0, 0, 0

    for pr_fname in pr_files:
        base = pr_fname.replace('_merged_seg.npy', '')
        gt_fname = base + '_seg.npy'
        

        
        pr_path = os.path.join(pr_dir, pr_fname)
        gt_path = os.path.join(gt_dir, gt_fname)

        if not os.path.exists(gt_path):
            print(f"[⚠️] 找不到 Ground Truth: {gt_fname}")
            continue

        try:
            pr_data = np.load(pr_path, allow_pickle=True)
            gt_data = np.load(gt_path, allow_pickle=True)
        except Exception as e:
            print(f" 讀取失敗：{pr_fname} or {gt_fname}：{e}")
            continue

        pr_mask = pr_data.item().get('masks') if pr_data.dtype == object else pr_data
        gt_mask = gt_data.item().get('masks') if gt_data.dtype == object else gt_data

        pr_mask = np.array(pr_mask)
        gt_mask = np.array(gt_mask)

        if pr_mask.shape != gt_mask.shape:
            pr_mask = np.array(Image.fromarray(pr_mask.astype(np.uint8)).resize(
                (gt_mask.shape[1], gt_mask.shape[0]), Image.NEAREST))

        result = compute_instance_overlap(gt_mask, pr_mask)
        # === 新增錯誤率計算 ===
        gt_count = result["gt_total"]
        err = (result["miss"] + result["false_positive"])
        error_rate = err / gt_count if gt_count > 0 else 0
        result["error_rate"] = error_rate
        error_rates.append(error_rate)

        
        results.append((pr_fname, result))

        total_gt += result["gt_total"]
        total_pr += result["pr_total"]
        total_match += result["matched"]
        total_miss += result["miss"]
        total_fp += result["false_positive"]

        print(f"[✓] {pr_fname}: GT={result['gt_total']} PR={result['pr_total']} "
              f"Match={result['matched']} Miss={result['miss']} FP={result['false_positive']}")

    # 輸出統計文字檔
    with open(output_txt, 'w') as f:
        for fname, res in results:
            f.write(f"{fname}:\n")
            f.write(f"  GT: {res['gt_total']}\n")
            f.write(f"  PR: {res['pr_total']}\n")
            f.write(f"  Matched: {res['matched']}\n")
            f.write(f"  Miss: {res['miss']}\n")
            f.write(f"  False Positive: {res['false_positive']}\n\n")
            f.write(f"  Error Rate: {res['error_rate']:.3f}\n\n")
        f.write("======== OVERALL ========\n")
        f.write(f"Total GT: {total_gt}\n")
        f.write(f"Total PR: {total_pr}\n")
        f.write(f"Total Matched: {total_match}\n")
        f.write(f"Total Missed: {total_miss}\n")
        f.write(f"Total False Positives: {total_fp}\n")
        f.write(f"Average Error Rate: {np.mean(error_rates):.3f}\n")
    print(f" 所有統計結果已儲存至：{output_txt}")
      # === 畫圖統計 ===
    image_names = [fname.replace('_seg.npy', '') for fname, _ in results]
    matched = np.array([res["matched"] for _, res in results], dtype=float)
    missed = np.array([res["miss"] for _, res in results], dtype=float)
    false_positive = np.array([res["false_positive"] for _, res in results], dtype=float)

    total = matched + missed + false_positive
    total[total == 0] = 1  # 避免除以 0

    matched_norm = matched / total
    missed_norm = missed / total
    fp_norm = false_positive / total

    x = np.arange(len(image_names))
    width = 0.6

    plt.figure(figsize=(max(10, len(image_names) * 0.5), 6))
    plt.bar(x, matched_norm, width, label='Matched', color='green')
    plt.bar(x, missed_norm, width, bottom=matched_norm, label='Missed', color='red')
    plt.bar(x, fp_norm, width, bottom=matched_norm + missed_norm, label='False Positive', color='blue')

    plt.xticks(x, image_names, rotation=90)
    plt.ylim(0, 1)
    plt.ylabel("Normalized Proportion")
    plt.title("Instance Matching Summary per Image (Normalized)")
    plt.legend()
    plt.tight_layout()

    plot_path = os.path.join(os.path.dirname(output_txt), "instance_comparison_plot_normalized_new_6.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()

    print(f" Normalized 匹配結果圖已儲存至：{plot_path}")
def main():
    print(" Instance Mask Comparison Tool")
    gt_dir = input("請輸入 Ground Truth 資料夾路徑（含 _group_seg.npy）: ").strip()
    pr_dir = input("請輸入 Prediction 資料夾路徑（含 _seg.npy）: ").strip()
    output_txt = os.path.join(pr_dir, "instance_comparison_summary_old_6.txt")

    if not os.path.isdir(gt_dir) or not os.path.isdir(pr_dir):
        print("錯誤：請確認路徑正確且為資料夾")
        return

    compare_all_instance_masks(gt_dir, pr_dir, output_txt)

if __name__ == "__main__":
    main()
