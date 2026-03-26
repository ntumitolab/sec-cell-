import os
import numpy as np
import tifffile as tiff
from albumentations import (
    HorizontalFlip, VerticalFlip, Rotate, RandomBrightnessContrast, Compose
)
import cv2
import random
import copy
# 資料夾設定
image_dir = "/home/kevinwu/Desktop/output_folder/out_600/"
mask_dir = "/home/kevinwu/Desktop/output_folder/out_600/"
aug_img_dir = "/home/kevinwu/Desktop/output_folder/out_600_extend/"
aug_mask_dir = "/home/kevinwu/Desktop/output_folder/out_600_extend/"
os.makedirs(aug_img_dir, exist_ok=True)
os.makedirs(aug_mask_dir, exist_ok=True)

# 定義同步變形的 augment 組合
augment = Compose([
    HorizontalFlip(p=0.5),
    VerticalFlip(p=0.5),
    Rotate(limit=90, p=0.5),
    RandomBrightnessContrast(p=0.3)
], additional_targets={'mask': 'mask'})  # 讓 mask 跟著 image 一起變形

n_aug = 15  # 每張圖片額外生成幾張

# 執行同步變形
for fname in os.listdir(image_dir):
    if fname.endswith(".tif") or fname.endswith(".tiff"):
        base_name = os.path.splitext(fname)[0]
        img_path = os.path.join(image_dir, fname)
        mask_path = os.path.join(mask_dir, f"{base_name}_seg.npy")
        if not os.path.exists(mask_path):
            print(f"[跳過] 找不到 mask：{mask_path}")
            continue
        # 讀圖與對應 mask
        image = tiff.imread(img_path)
        seg_data = np.load(mask_path, allow_pickle=True).item() # 假設是 cellpose 結構
        # for key, val in seg_data.items():
        #     print(f"{key} => {val}")


        mask = seg_data["masks"]
        
        if image.shape[:2] != mask.shape[:2]:
            print(f"[尺寸不一致] image: {image.shape}, mask: {mask.shape}，自動 resize 中...")
            mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
        for i in range(n_aug):
            augmented = augment(image=image, mask=mask)
            
            aug_img = augmented['image']
            aug_mask = augmented['mask']
            # 儲存
            
            
            aug_name = f"{base_name}_aug{i+1}"
            
            
            tiff.imwrite(os.path.join(aug_img_dir, f"{aug_name}.tiff"), aug_img.astype(np.uint8))
            imge_path = os.path.join(aug_img_dir, f"{aug_name}.tiff")
            ima = tiff.imread(imge_path)
            
            seg_data['masks'] = aug_mask
            # if 'filename' in seg_data:
            #     del seg_data['filename']
            seg_data['filename'] = f"/home/kevinwu/Desktop/extracted_single/1to3/{aug_name}.tiff"

            # print(seg_data.keys())
            # for key, val in seg_data.items():
            #     print(f"{key} => {val}")
            
                
            np.save(os.path.join(aug_mask_dir, f"{aug_name}_seg.npy"), seg_data)
            # for k, v in seg_data.items():
            #     print(f"\n【{k}】")
            #     print(f"  型別: {type(v)}")
            #     if isinstance(v, np.ndarray):
            #         print(f"  shape: {v.shape}, dtype: {v.dtype}")
            #         print(f"  範例數值 (前10): {v.flat[:10]}")
            #     elif isinstance(v, (list, tuple)):
            #         print(f"  長度: {len(v)}，前幾個值: {v[:5]}")
            #     else:
            #         print(f"  值: {v}")


print("影像與 mask 同步變形完成")

