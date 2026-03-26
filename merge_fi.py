import os
import numpy as np
from skimage import io as sk_io
from cellpose import utils
from seg_generate import merge_similar_rgb
import matplotlib.pyplot as plt
import copy
def main():
    # === 1. 使用者在程式執行後輸入路徑 ===
    # default_in = '/home/kevinwu/Desktop/merge/out/00'
    # default_out = '/home/kevinwu/Desktop/merge/out/00'
    
    input_folder = input(f"請輸入輸入檔所在資料夾路徑 : ")
    # if not input_folder.strip():
    #     input_folder = default_in  # 如果沒有輸入，就用預設值
    
    output_folder = input(f"請輸入輸出檔要儲存的資料夾路徑 : ")
    # if not output_folder.strip():
    #     output_folder = default_out
    
    # 其它可設定的參數 (例如合併閾值、debug_mode、迴圈範圍等)
    merge_threshold = 8
    # merge_threshold = 0.1
    debug_mode = False
    start_index = 1
    end_index = 421
    texture_thresh = 0.6
    # === 2. 確保輸出資料夾存在 ===
    os.makedirs(output_folder, exist_ok=True)
    
    # === 3. 進行批次處理 ===
    for i in range(start_index, end_index + 1):
        # 同時處理 P 以及 A
        for suffix in ['P', 'A']:
            base_name = f"fish_{i}_{suffix}"

            seg_file_path = os.path.join(input_folder, f"{base_name}_seg.npy")
            rgb_file_path = os.path.join(input_folder, f"{base_name}.tiff")

            # 檢查檔案是否存在
            if not os.path.exists(seg_file_path) or not os.path.exists(rgb_file_path):
                print(f"{base_name} 的檔案不存在，跳過...")
                continue

            # 嘗試載入 segmentation 結果
            try:
                seg_data = np.load(seg_file_path, allow_pickle=True).item()
                print(f"{base_name}_seg.npy 分割數據加載成功！")
            except Exception as e:
                print(f"{base_name}_seg.npy 加載分割數據失敗：{e}")
                continue

            # 嘗試載入對應的 RGB 圖像
            try:
                rgb_img = sk_io.imread(rgb_file_path)
                print(f"{base_name}.tiff 圖像加載成功！")
            except Exception as e:
                print(f"{base_name}.tiff 加載圖像失敗：{e}")
                continue
            # 如果是灰階圖，轉成 3-channel
            if len(rgb_img.shape) == 2:
                rgb_img = np.stack([rgb_img] * 3, axis=-1)
                print("偵測到灰階影像，已轉成 3-channel")

            # 如果是 RGBA，去掉 alpha channel
            if len(rgb_img.shape) == 3 and rgb_img.shape[2] == 4:
                rgb_img = rgb_img[:, :, :3]
                print("偵測到 RGBA 影像，已移除 alpha channel")

            # 確保圖像為 uint8 類型
            if rgb_img.dtype != np.uint8:
                rgb_img = (rgb_img * 255).astype(np.uint8)

            masks = seg_data.get('masks', None)
            if masks is None:
                print(f"{base_name}_seg.npy 分割遮罩未找到！")
                continue

            # 呼叫自訂函式進行合併
            seg_merged, relabeling = merge_similar_rgb(
                masks, rgb_img, merge=merge_threshold, debug_mode=debug_mode
            )

            # 更新 seg_data，並存回檔案
            
            new_seg_data = copy.deepcopy(seg_data)
            new_seg_data["masks"] = seg_merged
            new_seg_data["filename"] = f"{base_name}.tiff"
            
            #new_seg_data["filename"] = f"{base_name}_merge_seg.npy"
            merged_seg_path = os.path.join(output_folder, f"{base_name}_merged_seg.npy")
            np.save(merged_seg_path, new_seg_data)
            
            print(f"{base_name} 的分割數據已保存至 {merged_seg_path}")

            # 儲存 overlay 圖像
            # output_overlay_path = os.path.join(output_folder, f"{base_name}_overlay.png")

            # changed_mask = (masks != seg_merged)
            # merged_labels = list(relabeling.keys())
            # target_labels = list(relabeling.values())
            # merged_from_mask = np.isin(masks, merged_labels) & changed_mask
            # merged_into_mask = np.isin(seg_merged, target_labels)

            # overlay = rgb_img.copy().astype(float) / 255.0
            # overlay[merged_from_mask] = [1, 0, 0]  # 紅色
            # overlay[merged_into_mask & ~merged_from_mask] = [0, 0, 1]  # 藍色

            # plt.imsave(output_overlay_path, overlay)
            # print(f"{base_name} 的 Overlay 結果已儲存至 {output_overlay_path}")
            # plt.close()

            # # 計算合併比例並寫入文字檔
            # ratio_file_path = os.path.join(output_folder, 'merge_ratios.txt')
            # mask_area = np.count_nonzero(masks)
            # overlay_merged_area = np.count_nonzero(merged_from_mask | merged_into_mask)

            # if mask_area == 0:
            #     print(f"{base_name} 中無非背景像素，無法計算比例。")
            #     ratio_overlay = None
            # else:
            #     ratio_overlay = overlay_merged_area / mask_area
            #     print(f"{base_name} Overlay merged area ratio = {ratio_overlay:.2%}")

            # with open(ratio_file_path, 'a') as f:
            #     if ratio_overlay is None:
            #         f.write(f"{base_name}: 無法計算比例（無非背景像素）\n")
            #     else:
            #         f.write(f"{base_name}: {ratio_overlay:.2%}\n")

if __name__ == "__main__":
    main()
