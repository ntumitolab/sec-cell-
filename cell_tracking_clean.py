import numpy as np
import os
import re
from scipy.spatial import distance
import cv2
import csv
from skimage.measure import label, regionprops

def natural_key(string):
    """用於自然排序的 key function"""
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string)]

def calculate_iou(mask1, mask2):
    """計算兩個遮罩的 IoU (Intersection over Union)"""
    intersection = np.logical_and(mask1, mask2)
    union = np.logical_or(mask1, mask2)
    iou = np.sum(intersection) / np.sum(union) if np.sum(union) > 0 else 0
    return iou

def process_tracking(npy_folder):
    """處理細胞追蹤並返回追蹤結果"""
    # 初始化追蹤結果
    global_color_assignments = {}  # 儲存每個時間點的細胞對應的全域 ID
    parent_map = {}  # 儲存每個細胞的母細胞 ID
    generation_map = {}  # 儲存每個細胞的世代
    current_global_id = 1  # 全域 ID 計數器
    
    # 取得所有 .npy 文件並排序
    npy_files = [f for f in os.listdir(npy_folder) 
                 if (f.endswith('_seg.npy') or f.endswith('_seg_updated.npy'))
                 and not f.endswith('_merged_seg.npy')]
    npy_files.sort(key=natural_key)
    
    # 載入 merged mask 資料
    merged_data = {}
    for npy_file in npy_files:
        base_name = re.sub(r'_seg(_updated)?$', '', os.path.splitext(npy_file)[0])
        merged_file = f"{base_name}_merged_seg.npy"
        merged_path = os.path.join(npy_folder, merged_file)
        
        if os.path.exists(merged_path):
            data = np.load(merged_path, allow_pickle=True).item()
            if 'masks' in data:
                time_match = re.search(r'T(\d+)', npy_file)
                time_point = int(time_match.group(1)) if time_match else -1
                merged_data[time_point] = data['masks']
    
    # 處理每個時間點
    prev_masks = None
    prev_time = None
    
    for npy_file in npy_files:
        # 載入當前幀的遮罩
        data = np.load(os.path.join(npy_folder, npy_file), allow_pickle=True).item()
        if 'masks' not in data:
            continue
            
        current_masks = data['masks']
        time_match = re.search(r'T(\d+)', npy_file)
        current_time = int(time_match.group(1)) if time_match else -1
        
        # 取得當前幀的 merged mask
        current_merged_mask = merged_data.get(current_time, None)
        
        # 處理每個細胞標籤
        for label in np.unique(current_masks)[1:]:  # 跳過背景 0
            if current_time == 0:
                # T0：使用 merged mask 分組
                if current_merged_mask is not None:
                    label_region = (current_masks == label)
                    for ml in np.unique(current_merged_mask)[1:]:
                        merged_region = (current_merged_mask == ml)
                        if np.sum(np.logical_and(label_region, merged_region)) > 0:
                            global_color_assignments[(current_time, label)] = str(ml)
                            generation_map[(current_time, label)] = 0
                            break
                    else:
                        global_color_assignments[(current_time, label)] = f"cell_{label}"
                        generation_map[(current_time, label)] = 0
            else:
                # Tn：尋找最佳配對
                current_cell_mask = (current_masks == label)
                best_iou = 0.7  # IoU 閾值
                best_prev_label = None
                
                if prev_masks is not None:
                    # 與前一幀的所有細胞比較
                    for prev_label in np.unique(prev_masks)[1:]:
                        prev_cell_mask = (prev_masks == prev_label)
                        iou = calculate_iou(current_cell_mask, prev_cell_mask)
                        if iou > best_iou:
                            best_iou = iou
                            best_prev_label = prev_label
                
                if best_prev_label is not None:
                    # 找到對應的母細胞
                    parent_id = global_color_assignments.get((prev_time, best_prev_label))
                    global_color_assignments[(current_time, label)] = parent_id
                    parent_map[parent_id] = parent_id
                    generation_map[(current_time, label)] = generation_map.get((prev_time, best_prev_label), 0)
                else:
                    # 檢查是否為分裂事件
                    if current_merged_mask is not None:
                        label_region = (current_masks == label)
                        for ml in np.unique(current_merged_mask)[1:]:
                            merged_region = (current_merged_mask == ml)
                            if np.sum(np.logical_and(label_region, merged_region)) > 0:
                                # 計算同一 merged group 的細胞數
                                count = len([k for k, v in global_color_assignments.items() 
                                         if k[0] == current_time and 
                                         v.startswith(f"merged_{ml}")])
                                new_id = f"merged_{ml}_{count + 1}"
                                global_color_assignments[(current_time, label)] = str(ml)
                                parent_map[new_id] = str(ml)

                                # 自動取得上一代 generation 並 +1
                                parent_generation = 0
                                for (t, l), g in global_color_assignments.items():
                                    if t == current_time - 1 and g == str(ml):
                                        parent_generation = generation_map.get((t, l), 0)
                                        break
                                generation_map[(current_time, label)] = parent_generation + 1
                                break
                        else:
                            # 新細胞
                            new_id = f"cell_{current_global_id}"
                            global_color_assignments[(current_time, label)] = new_id
                            current_global_id += 1
                            generation_map[(current_time, label)] = 0
                    else:
                        # 無 merged mask 資訊
                        new_id = f"cell_{current_global_id}"
                        global_color_assignments[(current_time, label)] = new_id
                        current_global_id += 1
                        generation_map[(current_time, label)] = 0
        
        # 更新前一幀資訊
        prev_masks = current_masks
        prev_time = current_time
    
    return {
        'global_color_assignments': global_color_assignments,
        'parent_map': parent_map,
        'generation_map': generation_map
    }


# def extract_and_classify_labels(
#     npy_folder,
#     output_folder,
#     save_merged_crops=True,
#     image_root="/home/kevinwu/Desktop/output_folder/out/"
# ):
#     """
#     只做分類與（可選）輸出 merged 區域，且「每個 merged 區塊都同時儲存 .tiff 與 .npy」。
#     - 依 merged 區塊內與 seg 的重疊 label 數 k 分到 1to1/1to2/1to3/1to4
#     - 若有原圖、尺寸相符則輸出原圖裁切；否則輸出該區域的二值遮罩影像（tiff）
#     - .npy 內容為 dict，含：base, label_id, category, k, overlap_labels, bbox, merged_crop_mask, seg_crop_mask, has_original_image
#     - summary_all.txt 紀錄每個 merged 區塊的分類結果與輸出檔名
#     """
#     os.makedirs(output_folder, exist_ok=True)

#     # 建立分類子資料夾
#     categories = ["1to1", "1to2", "1to3", "1to4"]
#     for cat in categories:
#         os.makedirs(os.path.join(output_folder, cat), exist_ok=True)

#     summary_path = os.path.join(output_folder, "summary_all.txt")
#     summary_lines = []

#     # 只處理 *_merged_seg.npy
#     npy_files = [f for f in os.listdir(npy_folder) if f.endswith('_merged_seg.npy')]
#     npy_files.sort()

#     for merged_fname in npy_files:
#         base_name = re.sub(r'_merged_seg.npy$', '', merged_fname)
#         seg_fname = base_name + '_seg.npy'
#         merged_path = os.path.join(npy_folder, merged_fname)
#         seg_path = os.path.join(npy_folder, seg_fname)

#         if not os.path.exists(seg_path):
#             print(f" 找不到對應的原始檔案：{seg_fname}")
#             continue

#         merged_data = np.load(merged_path, allow_pickle=True).item()
#         seg_data = np.load(seg_path, allow_pickle=True).item()

#         merged_mask = merged_data.get("masks")
#         seg_mask = seg_data.get("masks")
#         if merged_mask is None or seg_mask is None:
#             print(f" {merged_fname} 或 {seg_fname} 缺少 'masks'")
#             continue

#         # 原圖路徑
#         img_path = os.path.join(image_root, base_name + ".tiff")
#         if os.path.exists(img_path):
#             original_img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
#         else:
#             original_img = None

#         # 尺寸檢查
#         H, W = merged_mask.shape[:2]
#         if original_img is not None:
#             if original_img.ndim == 2:
#                 h_img, w_img = original_img.shape
#             else:
#                 h_img, w_img = original_img.shape[:2]
#             if (H, W) != (h_img, w_img):
#                 print(f"⚠️ 尺寸不符：mask=({H},{W}) vs image=({h_img},{w_img})，改用遮罩影像")
#                 original_img = None

#         assert merged_mask.shape == seg_mask.shape, "mask 尺寸不一致"

#         # 逐個 merged label
#         merged_labels = np.unique(merged_mask)
#         merged_labels = merged_labels[merged_labels != 0]

#         for label_id in merged_labels:
#             region_mask = (merged_mask == label_id)
#             coords = np.argwhere(region_mask)
#             if coords.size == 0:
#                 continue

#             y_min, x_min = coords.min(axis=0)
#             y_max, x_max = coords.max(axis=0)

#             # 分類：該 merged 區內有幾個 seg label
#             overlap_labels = np.unique(seg_mask[region_mask])
#             overlap_labels = overlap_labels[overlap_labels != 0]
#             k = len(overlap_labels)
#             if   k == 1: category = "1to1"
#             elif k == 2: category = "1to2"
#             elif k == 3: category = "1to3"
#             else:        category = "1to4"

#             # 準備輸出檔名（同名 .tiff 與 .npy）
#             stem = f"{base_name}_label_{label_id}"
#             tiff_path = os.path.join(output_folder, category, stem + ".tiff")
#             npy_out_path = os.path.join(output_folder, category, stem + ".npy")

#             # 產生裁切用資料
#             merged_region_mask_crop = region_mask[y_min:y_max+1, x_min:x_max+1].astype(np.uint8)
#             seg_region_mask_crop = (seg_mask[y_min:y_max+1, x_min:x_max+1] > 0).astype(np.uint8)

#             # 影像輸出（tiff）：原圖裁切 or 遮罩影像
#             has_original = False
#             if save_merged_crops:
#                 if original_img is not None:
#                     crop = original_img[y_min:y_max+1, x_min:x_max+1]
#                     has_original = True
#                 else:
#                     # 用 merged 的二值遮罩做單通道影像
#                     crop = (merged_region_mask_crop * 255).astype(np.uint8)

#                 # 確保可寫入 tiff
#                 cv2.imwrite(tiff_path, crop)

#             # .npy 記錄完整資訊（方便之後訓練/重建）
#             out_dict = {
#                 "base": base_name,
#                 "label_id": int(label_id),
#                 "category": category,
#                 "k": int(k),
#                 "overlap_labels": [int(v) for v in overlap_labels.tolist()],
#                 "bbox": [int(y_min), int(x_min), int(y_max), int(x_max)],  # [ymin, xmin, ymax, xmax]
#                 "merged_crop_mask": merged_region_mask_crop,              # 該 merged 區域裁切的二值遮罩
#                 "seg_crop_mask": seg_region_mask_crop,                    # 該區域內是否屬於任一 seg 的二值遮罩
#                 "has_original_image": bool(has_original),
#                 "source": {
#                     "merged_path": merged_path,
#                     "seg_path": seg_path,
#                     "image_path": (img_path if os.path.exists(img_path) else None),
#                 },
#             }
#             # 小心：numpy 不能直接存 bool/np.ndarray 以外的複雜物件? -> 我們用 allow_pickle=True 讀回
#             np.save(npy_out_path, out_dict, allow_pickle=True)

#             # 摘要
#             summary_lines.append(
#                 f"[{base_name}] MergedLabel={label_id}  "
#                 f"overlap={list(map(int, overlap_labels))}  "
#                 f"count={k}  category={category}  "
#                 f"tiff={os.path.basename(tiff_path) if save_merged_crops else 'N/A'}  "
#                 f"npy={os.path.basename(npy_out_path)}\n"
#             )

#             print(f" {base_name} label {label_id} → {category}（已輸出 .tiff 與 .npy 到 {category}/）")

#     # 寫 summary
#     with open(summary_path, "w") as f:
#         f.writelines(summary_lines)
#     print(f" 已寫入總結：{summary_path}")

# #單顆加多顆的存檔
# def extract_and_classify_labels(
#     npy_folder,
#     output_folder,
#     image_root="/home/kevinwu/Desktop/output_folder/out/",
#     save_merged_crops=True,
#     save_single_cells=True
# ):
#     """
#     每個 merged 區塊：
#       - 存 .tiff (原圖裁切或遮罩) + .npy (meta)
#       - 若 k>=2 並且 save_single_cells=True，則再把裡面每顆 seg 單細胞切出來，各自存 .tiff + .npy
#     """
#     os.makedirs(output_folder, exist_ok=True)
#     categories = ["1to1", "1to2", "1to3", "1to4"]
#     for cat in categories:
#         os.makedirs(os.path.join(output_folder, cat), exist_ok=True)

#     summary_path = os.path.join(output_folder, "summary_all.txt")
#     summary_lines = []

#     npy_files = sorted([f for f in os.listdir(npy_folder) if f.endswith("_merged_seg.npy")])

#     for merged_fname in npy_files:
#         base = re.sub(r"_merged_seg\.npy$", "", merged_fname)
#         seg_fname = base + "_seg.npy"
#         merged_path = os.path.join(npy_folder, merged_fname)
#         seg_path = os.path.join(npy_folder, seg_fname)

#         if not os.path.exists(seg_path):
#             print(f"[跳過] 找不到 seg：{seg_fname}")
#             continue

#         merged_data = np.load(merged_path, allow_pickle=True).item()
#         seg_data = np.load(seg_path, allow_pickle=True).item()
#         merged_mask = merged_data.get("masks")
#         seg_mask = seg_data.get("masks")

#         if merged_mask is None or seg_mask is None:
#             print(f"[跳過] {base}: seg/merged 缺少 'masks'")
#             continue

#         assert merged_mask.shape == seg_mask.shape, f"{base}: seg/merged 尺寸不一致"
#         H, W = merged_mask.shape[:2]

#         # 原圖
#         img_path = os.path.join(image_root, base + ".tiff")
#         original_img = None
#         if os.path.exists(img_path):
#             original_img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
#             if original_img is not None:
#                 if original_img.ndim == 2:
#                     h_img, w_img = original_img.shape
#                 else:
#                     h_img, w_img = original_img.shape[:2]
#                 if (H, W) != (h_img, w_img):
#                     print(f"⚠️ 尺寸不符：mask=({H},{W}) vs image=({h_img},{w_img})，改用遮罩影像")
#                     original_img = None

#         merged_labels = np.unique(merged_mask)
#         merged_labels = merged_labels[merged_labels != 0]

#         for label_id in merged_labels:
#             region_mask = (merged_mask == label_id)
#             coords = np.argwhere(region_mask)
#             if coords.size == 0: continue

#             y_min, x_min = coords.min(axis=0)
#             y_max, x_max = coords.max(axis=0)

#             # 分類 k
#             overlap_labels = np.unique(seg_mask[region_mask])
#             overlap_labels = overlap_labels[overlap_labels != 0]
#             k = len(overlap_labels)
#             if   k == 1: category = "1to1"
#             elif k == 2: category = "1to2"
#             elif k == 3: category = "1to3"
#             else:        category = "1to4"

#             stem = f"{base}_merged_{label_id}"
#             tiff_out = os.path.join(output_folder, category, stem + ".tiff")
#             npy_out  = os.path.join(output_folder, category, stem + ".npy")

#             merged_crop_mask = region_mask[y_min:y_max+1, x_min:x_max+1].astype(np.uint8)
#             seg_crop_mask = (seg_mask[y_min:y_max+1, x_min:x_max+1] > 0).astype(np.uint8)

#             # 輸出 merged 影像
#             has_original = False
#             if save_merged_crops:
#                 if original_img is not None:
#                     crop = original_img[y_min:y_max+1, x_min:x_max+1]
#                     has_original = True
#                 else:
#                     crop = (merged_crop_mask * 255).astype(np.uint8)
#                 cv2.imwrite(tiff_out, crop)

#             out_dict = {
#                 "base": base,
#                 "merged_label": int(label_id),
#                 "category": category,
#                 "k": int(k),
#                 "overlap_labels": [int(v) for v in overlap_labels.tolist()],
#                 "bbox": [int(y_min), int(x_min), int(y_max), int(x_max)],
#                 "merged_crop_mask": merged_crop_mask,
#                 "seg_crop_mask": seg_crop_mask,
#                 "has_original_image": bool(has_original),
#                 "source": {
#                     "merged_path": merged_path,
#                     "seg_path": seg_path,
#                     "image_path": (img_path if os.path.exists(img_path) else None),
#                 },
#             }
#             np.save(npy_out, out_dict, allow_pickle=True)

#             # 如果要逐顆切
#             if save_single_cells and k >= 2:
#                 for s in overlap_labels:
#                     single_mask = (seg_mask == s) & region_mask
#                     if not np.any(single_mask): continue
#                     syx = np.argwhere(single_mask)
#                     sy_min, sx_min = syx.min(axis=0)
#                     sy_max, sx_max = syx.max(axis=0)

#                     # 裁切
#                     if original_img is not None:
#                         cell_crop = original_img[sy_min:sy_max+1, sx_min:sx_max+1].copy()
#                         local_mask = single_mask[sy_min:sy_max+1, sx_min:sx_max+1]
#                         if cell_crop.ndim == 2:
#                             cell_crop[~local_mask] = 0
#                         else:
#                             cell_crop[~local_mask] = 0
#                     else:
#                         cell_crop = (single_mask[sy_min:sy_max+1, sx_min:sx_max+1].astype(np.uint8) * 255)

#                     cell_name = f"{base}_merged_{label_id}_seg{s}"
#                     tiff_cell = os.path.join(output_folder, category, cell_name + ".tiff")
#                     npy_cell  = os.path.join(output_folder, category, cell_name + ".npy")

#                     cv2.imwrite(tiff_cell, cell_crop)

#                     cell_dict = {
#                         "base": base,
#                         "merged_label": int(label_id),
#                         "seg_label": int(s),
#                         "category": category,
#                         "bbox": [int(sy_min), int(sx_min), int(sy_max), int(sx_max)],
#                         "mask": single_mask[sy_min:sy_max+1, sx_min:sx_max+1].astype(np.uint8),
#                         "has_original_image": original_img is not None,
#                     }
#                     np.save(npy_cell, cell_dict, allow_pickle=True)

#             summary_lines.append(
#                 f"[{base}] merged={label_id} k={k} category={category} "
#                 f"tiff={os.path.basename(tiff_out)} npy={os.path.basename(npy_out)}\n"
#             )
#             print(f"{base} merged={label_id} → {category}（已輸出 merged .tiff/.npy，單細胞 {('有' if (save_single_cells and k>=2) else '無')}）")

#     with open(summary_path, "w") as f:
#         f.writelines(summary_lines)
#     print(f"已寫入總結：{summary_path}")




# def extract_and_classify_labels(npy_folder, output_folder, save_merged_crops=True):
#     """
#     只做分類與（可選）輸出 merged 區域裁切，不再對 seg 做逐顆細胞裁切。
#     - 依 merged 區塊內與 seg 的重疊 label 數 k 分到 1to1/1to2/1to3/1to4
#     - 若有原圖、尺寸相符則輸出 merged 區域裁切；否則輸出該區域的二值遮罩
#     - 輸出 summary_all.txt 紀錄每個 merged 區塊的分類結果
#     """
#     os.makedirs(output_folder, exist_ok=True)

#     # 建立分類子資料夾（統一使用 1to4）
#     categories = ["1to1", "1to2", "1to3", "1to4"]
#     for cat in categories:
#         os.makedirs(os.path.join(output_folder, cat), exist_ok=True)

#     summary_path = os.path.join(output_folder, "summary_all.txt")
#     summary_lines = []

#     # 僅找 *_merged_seg.npy
#     npy_files = [f for f in os.listdir(npy_folder) if f.endswith('_merged_seg.npy')]
#     npy_files.sort()

#     # 固定原圖資料夾（可按需修改）
#     image_root = "/home/kevinwu/Desktop/output_folder/out/"

#     for merged_fname in npy_files:
#         base_name = re.sub(r'_merged_seg.npy$', '', merged_fname)
#         seg_fname = base_name + '_seg.npy'
#         merged_path = os.path.join(npy_folder, merged_fname)
#         seg_path = os.path.join(npy_folder, seg_fname)

#         if not os.path.exists(seg_path):
#             print(f" 找不到對應的原始檔案：{seg_fname}")
#             continue

#         merged_data = np.load(merged_path, allow_pickle=True).item()
#         seg_data = np.load(seg_path, allow_pickle=True).item()

#         merged_mask = merged_data.get("masks")
#         seg_mask = seg_data.get("masks")
#         if merged_mask is None or seg_mask is None:
#             print(f" {merged_fname} 或 {seg_fname} 缺少 'masks'")
#             continue

#         # --- 從固定資料夾讀原圖（若存在且尺寸相符才用來裁切） ---
#         img_path = os.path.join(image_root, base_name + ".tiff")
#         if not os.path.exists(img_path):
#             original_img = None
#         else:
#             original_img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

#         # 尺寸檢查
#         H, W = merged_mask.shape[:2]
#         if original_img is not None:
#             if original_img.ndim == 2:
#                 h_img, w_img = original_img.shape
#             else:
#                 h_img, w_img = original_img.shape[:2]
#             if (H, W) != (h_img, w_img):
#                 print(f"⚠️ 尺寸不符：mask=({H},{W}) vs image=({h_img},{w_img})，改用遮罩輸出")
#                 original_img = None

#         assert merged_mask.shape == seg_mask.shape, "mask 尺寸不一致"

#         # 逐個 merged label
#         merged_labels = np.unique(merged_mask)
#         merged_labels = merged_labels[merged_labels != 0]

#         for label_id in merged_labels:
#             region_mask = (merged_mask == label_id)
#             coords = np.argwhere(region_mask)
#             if coords.size == 0:
#                 continue

#             y_min, x_min = coords.min(axis=0)
#             y_max, x_max = coords.max(axis=0)

#             # 分類（看 merged 區內 seg 有幾個不同 label）
#             overlap_labels = np.unique(seg_mask[region_mask])
#             overlap_labels = overlap_labels[overlap_labels != 0]
#             k = len(overlap_labels)
#             if   k == 1: category = "1to1"
#             elif k == 2: category = "1to2"
#             elif k == 3: category = "1to3"
#             else:        category = "1to4"

#             merged_name = f"{base_name}_label_{label_id}.png"

#             # 只輸出 merged 區域（原圖裁切或遮罩），不再逐顆 seg 裁切
#             if save_merged_crops:
#                 if original_img is not None:
#                     merged_crop = original_img[y_min:y_max+1, x_min:x_max+1]
#                     cv2.imwrite(os.path.join(output_folder, category, merged_name), merged_crop)
#                 else:
#                     merged_crop = (region_mask[y_min:y_max+1, x_min:x_max+1].astype(np.uint8) * 255)
#                     cv2.imwrite(os.path.join(output_folder, category, merged_name), merged_crop)

#             # 累積摘要
#             summary_lines.append(
#                 f"[{base_name}] MergedLabel={label_id}  "
#                 f"overlap={list(map(int, overlap_labels))}  "
#                 f"count={k}  category={category}  "
#                 f"merged_crop={merged_name if save_merged_crops else 'N/A'}\n"
#             )

#             print(f" {base_name} label {label_id} → {category}（已輸出到 {category}/，不做逐顆切割）")

#     with open(summary_path, "w") as f:
#         f.writelines(summary_lines)

#     print(f" 已寫入總結：{summary_path}")





def extract_and_classify_labels(
    npy_folder,
    output_folder,
    image_root="/home/kevinwu/Desktop/output_folder/out/",
    save_single_cells=True
):
    """
    只輸出「已切開的單細胞」，不存 merged block。
    """
    os.makedirs(output_folder, exist_ok=True)
    categories = ["1to1", "1to2", "1to3", "1to4"]
    for cat in categories:
        os.makedirs(os.path.join(output_folder, cat), exist_ok=True)

    summary_path = os.path.join(output_folder, "summary_all.txt")
    summary_lines = []

    npy_files = sorted([f for f in os.listdir(npy_folder) if f.endswith("_merged_seg.npy")])

    for merged_fname in npy_files:
        base = re.sub(r"_merged_seg\.npy$", "", merged_fname)
        seg_fname = base + "_seg.npy"
        merged_path = os.path.join(npy_folder, merged_fname)
        seg_path = os.path.join(npy_folder, seg_fname)

        if not os.path.exists(seg_path):
            print(f"[跳過] 找不到 seg：{seg_fname}")
            continue

        merged_data = np.load(merged_path, allow_pickle=True).item()
        seg_data = np.load(seg_path, allow_pickle=True).item()
        merged_mask = merged_data.get("masks")
        seg_mask = seg_data.get("masks")

        if merged_mask is None or seg_mask is None:
            print(f"[跳過] {base}: seg/merged 缺少 'masks'")
            continue

        assert merged_mask.shape == seg_mask.shape, f"{base}: seg/merged 尺寸不一致"
        H, W = merged_mask.shape[:2]

        # 原圖
        img_path = os.path.join(image_root, base + ".tiff")
        original_img = None
        if os.path.exists(img_path):
            original_img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            if original_img is not None:
                if original_img.ndim == 2:
                    h_img, w_img = original_img.shape
                else:
                    h_img, w_img = original_img.shape[:2]
                if (H, W) != (h_img, w_img):
                    print(f"⚠️ 尺寸不符：mask=({H},{W}) vs image=({h_img},{w_img})，改用遮罩影像")
                    original_img = None

        merged_labels = np.unique(merged_mask)
        merged_labels = merged_labels[merged_labels != 0]

        for label_id in merged_labels:
            region_mask = (merged_mask == label_id)
            coords = np.argwhere(region_mask)
            if coords.size == 0: continue

            # 找出這個 merged block 包含幾顆 seg
            overlap_labels = np.unique(seg_mask[region_mask])
            overlap_labels = overlap_labels[overlap_labels != 0]
            k = len(overlap_labels)
            if   k == 1: category = "1to1"
            elif k == 2: category = "1to2"
            elif k == 3: category = "1to3"
            else:        category = "1to4"

            # === 只存單顆 ===
            if save_single_cells and k >= 1:   # k>=1 代表至少有一顆細胞
                for s in overlap_labels:
                    single_mask = (seg_mask == s) & region_mask
                    if not np.any(single_mask): continue
                    syx = np.argwhere(single_mask)
                    sy_min, sx_min = syx.min(axis=0)
                    sy_max, sx_max = syx.max(axis=0)

                    # 裁切
                    if original_img is not None:
                        cell_crop = original_img[sy_min:sy_max+1, sx_min:sx_max+1].copy()
                        local_mask = single_mask[sy_min:sy_max+1, sx_min:sx_max+1]
                        if cell_crop.ndim == 2:
                            cell_crop[~local_mask] = 0
                        else:
                            cell_crop[~local_mask] = 0
                    else:
                        cell_crop = (single_mask[sy_min:sy_max+1, sx_min:sx_max+1].astype(np.uint8) * 255)

                    cell_name = f"{base}_merged_{label_id}_seg{s}"
                    tiff_cell = os.path.join(output_folder, category, cell_name + ".tiff")
                    npy_cell  = os.path.join(output_folder, category, cell_name + ".npy")

                    cv2.imwrite(tiff_cell, cell_crop)

                    cell_dict = {
                        "base": base,
                        "merged_label": int(label_id),
                        "seg_label": int(s),
                        "category": category,
                        "bbox": [int(sy_min), int(sx_min), int(sy_max), int(sx_max)],
                        "mask": single_mask[sy_min:sy_max+1, sx_min:sx_max+1].astype(np.uint8),
                        "has_original_image": original_img is not None,
                    }
                    np.save(npy_cell, cell_dict, allow_pickle=True)

                    summary_lines.append(
                        f"[{base}] merged={label_id} seg={s} → {category} "
                        f"tiff={os.path.basename(tiff_cell)} npy={os.path.basename(npy_cell)}\n"
                    )
                print(f"{base} merged={label_id} → {category}（已輸出 {len(overlap_labels)} 顆單細胞）")

    with open(summary_path, "w") as f:
        f.writelines(summary_lines)
    print(f"已寫入總結：{summary_path}")

    #############################################################
   
 
def main():
    # 設定你的資料夾路徑
    npy_folder = "/home/kevinwu/Desktop/output_folder/out_old/L/"

    # 執行細胞追蹤
    tracking_result = process_tracking(npy_folder)

    global_ids = tracking_result['global_color_assignments']
    parent_map = tracking_result['parent_map']
    generation_map = tracking_result['generation_map']

    # 印出每個細胞的資訊
    print("=== 細胞追蹤結果 ===")
    for (time, label), global_id in sorted(global_ids.items()):
        generation = generation_map.get((time, label), 0)
        parent = parent_map.get(global_id, "N/A")
        print(f"T{time} - Label {label} → Global ID: {global_id}, Generation: {generation}, Parent: {parent}")

    # 可選：儲存成 CSV
    save_csv = True
    if save_csv:
        csv_path = "tracking_output.csv"
        with open(csv_path, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Time", "Label", "Global_ID", "Generation", "Parent_ID"])
            for (time, label), global_id in sorted(global_ids.items()):
                generation = generation_map.get((time, label), 0)
                parent = parent_map.get(global_id, "N/A")
                writer.writerow([time, label, global_id, generation, parent])
        print(f"\n 已儲存追蹤結果至：{csv_path}")

    output_folder = "/home/kevinwu/Desktop/1to4_LMS/"

    extract_and_classify_labels(npy_folder, output_folder)


if __name__ == "__main__":
    main()
