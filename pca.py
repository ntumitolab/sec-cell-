import os
import numpy as np
import cv2

# 原始資料夾
input_folder = "/home/kevinwu/Desktop/output_folder/out_old_test/"

# 輸出資料夾
output_folder = "/home/kevinwu/Desktop/output_folder/out_old_test_gray/"
os.makedirs(output_folder, exist_ok=True)


def pca_grayscale(image):

    # 轉成 float
    img = image.astype(np.float32)

    # reshape 成 N x 3
    pixels = img.reshape(-1, 3)

    # PCA
    mean = np.mean(pixels, axis=0)
    centered = pixels - mean

    cov = np.cov(centered, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)

    principal_component = eigvecs[:, np.argmax(eigvals)]

    gray = centered @ principal_component

    # normalize
    gray = gray.reshape(image.shape[:2])
    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)

    return gray.astype(np.uint8)


for fname in os.listdir(input_folder):

    if fname.lower().endswith(".tiff") or fname.lower().endswith(".tif"):

        path = os.path.join(input_folder, fname)

        img = cv2.imread(path)

        if img is None:
            print("讀取失敗:", fname)
            continue

        gray = pca_grayscale(img)

        save_path = os.path.join(output_folder, fname)

        cv2.imwrite(save_path, gray)

        print("完成:", fname)

print("全部轉換完成")