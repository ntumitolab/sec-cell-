# -*- coding: utf-8 -*-
"""
"""
import os
import pickle
from copy import deepcopy
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import skimage as ski
from PIL import Image, ImageDraw, ImageFont
from rich import print
from rich.console import Console
from rich.pretty import Pretty
from rich.traceback import install
from scipy.ndimage import binary_dilation as dila
from skimage import measure
from skimage.color import deltaE_ciede94, rgb2lab
from skimage.segmentation import mark_boundaries, slic
import random

# from ..shared.config import load_config
# from ..shared.utils import create_new_dir
# from .calc_seg_feat import count_element, update_seg_analysis_dict
# from .utils import get_seg_desc, get_slic_param_name

install()
# -----------------------------------------------------------------------------/


def bwRGB(bw,im):
    """ (deprecated) channel order of `im` is `BGR` (default to `cv2.imread()`)
    """
    A = np.sum(bw)
    B = np.sum(im[bw,0])/A
    G = np.sum(im[bw,1])/A
    R = np.sum(im[bw,2])/A
    return [R,G,B]
    # -------------------------------------------------------------------------/


def simple_col_dis(color1, color2):
    """
    """
    sum = 0
    for i in range(3):
        ds = (float(color1[i]) - float(color2[i]))**2
        sum =sum+ds
    delta_e = np.sqrt(sum)
    return delta_e

    # return np.sqrt(np.sum((np.array(color1) - np.array(color2))**2))
    # -------------------------------------------------------------------------/


def save_segment_result(save_path:Path, seg:np.ndarray):
    """
    """
    with open(save_path, mode="wb") as f_writer:
        pickle.dump(seg, f_writer)
    # -------------------------------------------------------------------------/


def save_seg_on_img(save_path:Path, img:np.ndarray, seg:np.ndarray):
    """ (deprecated)
    """
    seg_on_img = np.uint8(mark_boundaries(img, seg)*255)
    cv2.imwrite(str(save_path), seg_on_img)
    # -------------------------------------------------------------------------/


def get_average_rgb(mask: np.ndarray, rgb_img: np.ndarray,
                    avg_ratio: float):
    """
    """
    assert rgb_img.dtype == np.uint8, "rgb_img.dtype != np.uint8"
    assert isinstance(avg_ratio, float)
    
    # vars
    ch_order = {"R":0, "G":1, "B":2}
    avg_rgb_dict = {"R":float, "G":float, "B":float}
    
    for k, v in ch_order.items():
        pixels = np.sort(rgb_img[mask, v])[::-1]
        area = len(pixels)
        # average with partial pixels
        pixel_avg = np.average(pixels[:int(area*avg_ratio)])
        # update `avg_rgb`
        avg_rgb_dict[k] = pixel_avg
    
    return avg_rgb_dict, np.array(list(avg_rgb_dict.values()))
    # -------------------------------------------------------------------------/


def get_average_rgb_v2(mask: np.ndarray, rgb_img: np.ndarray,
                       qL: float, qR: float):
    """
    """
    assert rgb_img.dtype == np.uint8, "rgb_img.dtype != np.uint8"
    assert isinstance(qL, float)
    assert isinstance(qR, float)
    
    # vars
    ch_order = {"R":0, "G":1, "B":2}
    avg_rgb_dict = {"R":float, "G":float, "B":float}
    
    for k, v in ch_order.items():
        pixels = np.sort(rgb_img[mask, v])[::-1]
        val_qL = np.quantile(pixels, qL)
        val_qR = np.quantile(pixels, qR)
        # average with custom quantile interval
        pixels = pixels[(pixels >= val_qL) & (pixels <= val_qR)]
        pixel_avg = np.average(pixels)
        # update `avg_rgb`
        avg_rgb_dict[k] = pixel_avg
    
    return avg_rgb_dict, np.array(list(avg_rgb_dict.values()))


    # -------------------------------------------------------------------------/


def average_rgb_coloring(seg: np.ndarray, rgb_img: np.ndarray):
    """ channel order of `img` is `RGB`
    """
    assert rgb_img.dtype == np.uint8, "rgb_img.dtype != np.uint8"
    
    # vars
    avgcolor_img = np.zeros((*seg.shape, 3), dtype=rgb_img.dtype)
    
    labels = np.unique(seg)
    for label in labels:
        if label == 0: continue # skip background
        mask = (seg == label)
        # _, avg_rgb = get_average_rgb(mask, rgb_img, avg_ratio=0.5)
        _, avg_rgb = get_average_rgb_v2(mask, rgb_img, qL=0.5, qR=0.9)
        avgcolor_img[mask] = np.uint8(avg_rgb)
    
    # check and return
    assert id(avgcolor_img) != id(rgb_img)
    return avgcolor_img
    # -------------------------------------------------------------------------/

# #new
def merge_similar_rgb(seg: np.ndarray, rgb_img: np.ndarray,
                      merge: float, debug_mode: bool):
    assert rgb_img.dtype == np.uint8, "rgb_img.dtype != np.uint8"

    from copy import deepcopy
    import random

    merge_seg = deepcopy(seg)
    relabeling: dict[int, int] = {}
    labels = list(np.unique(merge_seg))
    if 0 in labels:
        labels.remove(0)
    random.shuffle(labels)

    merge_groups = {label: {label} for label in labels}

    for label in labels:
        #  加這行：確保目前的 label group不超過4人
        if len(merge_groups.get(label, {label})) > 4:
            continue

        mask = (merge_seg == label)
        if np.sum(mask) > 0:
            color = get_average_rgb_v2(mask, rgb_img, qL=0.5, qR=0.9)[1]
            mask_dila = dila(mask, iterations=2)
            nlabels = np.unique(merge_seg[mask_dila])

            for nlabel in nlabels:
                if nlabel == 0 or nlabel == label:
                    continue

                merged_set = merge_groups[label].union(merge_groups.get(nlabel, {nlabel}))
                if len(merged_set) > 4:
                    continue

                nmask = (merge_seg == nlabel)
                ncolor = get_average_rgb_v2(nmask, rgb_img, qL=0.5, qR=0.9)[1]
                delta_e = deltaE_ciede94(rgb2lab(color / 255.0), rgb2lab(ncolor / 255.0))

                if delta_e <= merge:
                    # 合併
                    merge_seg[nmask] = label
                    relabeling[nlabel] = label

                    # 更新 group
                    for sublabel in merged_set:
                        merge_groups[sublabel] = merged_set

                    if debug_mode:
                        print(f"Merge: {label} and {nlabel}, Group size: {len(merged_set)}")

                if len(merge_groups[label]) > 4:
                    break  # 達到4個就break

    # ✅ 加這段：二次全域檢查
    for label, group in merge_groups.items():
        if len(group) > 4:
            raise RuntimeError(f"❌ 找到超過4個的群組！label={label}, group={group}")

    return merge_seg, relabeling




# def merge_similar_rgb(seg: np.ndarray, rgb_img: np.ndarray,
#                                    merge: float, debug_mode: bool):
#     """
#     使用全局追蹤限制每個區域的合併總數不超過4。
#     """
#     assert rgb_img.dtype == np.uint8, "rgb_img.dtype != np.uint8"
    
#     # vars
#     merge_seg = deepcopy(seg)
#     relabeling: dict[int, int] = {}
#     delta_e_dict: dict[str, float] = {}
    
#     labels = list(np.unique(merge_seg))
#     if 0 in labels:
#         labels.remove(0)  # 移除背景標籤
#     random.shuffle(labels)  # 隨機順序處理標籤
    
#     # 全局追蹤合併集合
#     merge_groups = {label: {label} for label in labels}

#     for label in labels:
#         if len(merge_groups[label]) > 4:
#             continue  # 超過限制的區域跳過

#         mask = (merge_seg == label)
#         if np.sum(mask) > 0:
#             color = get_average_rgb_v2(mask, rgb_img, qL=0.5, qR=0.9)[1]
#             mask_dila = dila(mask, iterations=2)
#             nlabels = np.unique(merge_seg[mask_dila])

#             for nlabel in nlabels:
#                 if nlabel == 0 or nlabel == label:
#                     continue  # 跳過背景或自身

#                 if len(merge_groups[label].union(merge_groups[nlabel])) > 4:
#                     continue  # 若合併後超過4個則跳過

#                 nmask = (merge_seg == nlabel)
#                 ncolor = get_average_rgb_v2(nmask, rgb_img, qL=0.5, qR=0.9)[1]
#                 delta_e = deltaE_ciede94(rgb2lab(color / 255.0), rgb2lab(ncolor / 255.0))

#                 if delta_e <= merge:
#                     # 合併操作
#                     merge_seg[nmask] = label
#                     relabeling[nlabel] = label

#                     # 更新全局合併集合
#                     merge_groups[label] = merge_groups[label].union(merge_groups[nlabel])
#                     for sublabel in merge_groups[nlabel]:
#                         merge_groups[sublabel] = merge_groups[label]  # 同步更新所有成員

#                     if debug_mode:
#                         print(f"Merge: {label} and {nlabel}, Total: {len(merge_groups[label])}")

#                 if len(merge_groups[label]) > 4:
#                     break  # 若達到限制則停止進一步合併

#     return merge_seg, relabeling

#origin
# def merge_similar_rgb(seg: np.ndarray, rgb_img: np.ndarray,
#                       merge: float, debug_mode: bool):
#     """
#     """
#     assert rgb_img.dtype == np.uint8, "rgb_img.dtype != np.uint8"
    
#     # vars
#     merge_seg = deepcopy(seg)
#     relabeling: dict[int, int] = {}
#     delta_e_dict: dict[str, float] = {} # for debugger
    
#     labels = np.unique(merge_seg)
#     for label in labels:
#         if label == 0: continue # skip background
#         mask = (merge_seg == label)
#         if np.sum(mask) > 0: # merge 後會跳號， mask 可能會沒東西
#             # color = get_average_rgb(mask, rgb_img, avg_ratio=0.5)[1] # get self color
#             color = get_average_rgb_v2(mask, rgb_img, qL=0.5, qR=0.9)[1] # get self color
#             mask_dila = dila(mask, iterations=2) # find neighbor
#             nlabels = np.unique(merge_seg[mask_dila]) # self + neighbor's labels
#             for nlabel in nlabels:
#                 if nlabel == 0: continue # skip background
#                 elif nlabel > label: # avoid repeated merging
#                     nmask = (merge_seg == nlabel)
#                     # ncolor = get_average_rgb(nmask, rgb_img, avg_ratio=0.5)[1] # neighbor's color
#                     ncolor = get_average_rgb_v2(nmask, rgb_img, qL=0.5, qR=0.9)[1] # neighbor's color
#                     delta_e = deltaE_ciede94(rgb2lab(color/255.0), rgb2lab(ncolor/255.0))
#                     delta_e_dict[f"{label}_cmp_{nlabel}"] = delta_e # for debugger
#                     if delta_e <= merge:
#                         merge_seg[nmask] = label
#                         relabeling[nlabel] = label
#         else:
#             if debug_mode:
#                 print(f"'{label}' has been merged before dealing with")
    
#     # check and return
#     assert id(merge_seg) != id(seg)
#     return merge_seg, relabeling





# def merge_similar_rgb(seg: np.ndarray, rgb_img: np.ndarray,
#                       merge: float, debug_mode: bool):
#     """
#     將相鄰且顏色相近的區域合併並返回合併後的標籤影像與標籤對應字典。
#     """

#     assert rgb_img.dtype == np.uint8, "rgb_img.dtype != np.uint8"
    
#     merge_seg = deepcopy(seg)
#     relabeling: dict[int, int] = {}
#     delta_e_dict: dict[str, float] = {} # for debugging
    
#     labels = np.unique(merge_seg)
#     for label in labels:
#         if label == 0:
#             continue # skip background
#         mask = (merge_seg == label)
#         if np.sum(mask) > 0:
#             color = get_average_rgb_v2(mask, rgb_img, qL=0.5, qR=0.9)[1] # get self color
#             mask_dila = dila(mask, iterations=2) # find neighbors
#             nlabels = np.unique(merge_seg[mask_dila]) # self + neighbors
#             for nlabel in nlabels:
#                 if nlabel == 0:
#                     continue # skip background
#                 elif nlabel > label: # avoid repeated merging
#                     nmask = (merge_seg == nlabel)
#                     if np.sum(nmask) == 0:
#                         continue
#                     ncolor = get_average_rgb_v2(nmask, rgb_img, qL=0.5, qR=0.9)[1] # neighbor's color
#                     delta_e = deltaE_ciede94(rgb2lab(color/255.0), rgb2lab(ncolor/255.0))
#                     delta_e_dict[f"{label}_cmp_{nlabel}"] = delta_e # for debugger
#                     if delta_e <= merge:
#                         merge_seg[nmask] = label
#                         relabeling[nlabel] = label
#         else:
#             if debug_mode:
#                 print(f"'{label}' has been merged before dealing with")
    
#     assert id(merge_seg) != id(seg)
#     return merge_seg, relabeling





# def merge_similar_rgb(seg: np.ndarray, rgb_img: np.ndarray,
#                              merge: float, debug_mode: bool):
#     """
#     嚴格限制每個區域的合併總數不超過4，包括直接和間接合併的所有區域。
#     """
#     assert rgb_img.dtype == np.uint8, "rgb_img.dtype != np.uint8"
    
#     # vars
#     merge_seg = deepcopy(seg)
#     relabeling: dict[int, int] = {}
#     delta_e_dict: dict[str, float] = {}
    
#     labels = list(np.unique(merge_seg))
#     if 0 in labels:
#         labels.remove(0)  # 移除背景標籤
#     random.shuffle(labels)  # 隨機順序處理標籤
    
#     # 全局合併集合
#     merge_groups = {label: {label} for label in labels}

#     for label in labels:
#         if len(merge_groups[label]) > 4:
#             continue  # 超過限制的區域跳過

#         mask = (merge_seg == label)
#         if np.sum(mask) > 0:
#             # 計算當前區域的平均 RGB 顏色
#             color = get_average_rgb_v2(mask, rgb_img, qL=0.5, qR=0.9)[1]
#             mask_dila = dila(mask, iterations=2)
#             nlabels = np.unique(merge_seg[mask_dila])

#             for nlabel in nlabels:
#                 if nlabel == 0 or nlabel == label:
#                     continue

#                 # 檢查合併後的集合大小是否超過限制
#                 combined_size = len(merge_groups[label].union(merge_groups[nlabel]))
#                 if combined_size > 4:
#                     if debug_mode:
#                         print(f"Skipping merge of {label} and {nlabel}: combined size {combined_size} exceeds limit")
#                     continue  # 跳過合併

#                 nmask = (merge_seg == nlabel)
#                 ncolor = get_average_rgb_v2(nmask, rgb_img, qL=0.5, qR=0.9)[1]

#                 # 計算顏色差異（DeltaE）
#                 delta_e = deltaE_ciede94(rgb2lab(color / 255.0), rgb2lab(ncolor / 255.0))

#                 if delta_e <= merge:
#                     # 合併操作
#                     merge_seg[nmask] = label
#                     relabeling[nlabel] = label

#                     # 更新全局合併集合
#                     merge_groups[label] = merge_groups[label].union(merge_groups[nlabel])
#                     for sublabel in merge_groups[nlabel]:
#                         merge_groups[sublabel] = merge_groups[label]  # 同步更新所有成員

#                     if debug_mode:
#                         print(f"Merge: {label} and {nlabel}, New Group Size: {len(merge_groups[label])}")

#                 # 如果合併後的集合大小達到限制，停止進一步合併
#                 if len(merge_groups[label]) > 4:
#                     if debug_mode:
#                         print(f"Stopping merges for {label}: group size {len(merge_groups[label])} reached limit")
#                     break

#     return merge_seg, relabeling


# def merge_similar_rgb(seg: np.ndarray, rgb_img: np.ndarray,
#                              merge: float, debug_mode: bool):
#     """
#     嚴格限制每個區域的合併總數不超過4，包括直接和間接合併的所有區域。
#     """
#     assert rgb_img.dtype == np.uint8, "rgb_img.dtype != np.uint8"
    
#     # vars
#     merge_seg = deepcopy(seg)
#     relabeling: dict[int, int] = {}
#     delta_e_dict: dict[str, float] = {}
    
#     labels = list(np.unique(merge_seg))
#     if 0 in labels:
#         labels.remove(0)  # 移除背景標籤
#     random.shuffle(labels)  # 隨機順序處理標籤
    
#     # 全局合併集合
#     merge_groups = {label: {label} for label in labels}

#     for label in labels:
#         if len(merge_groups[label]) > 4:
#             continue  # 超過限制的區域跳過

#         mask = (merge_seg == label)
#         if np.sum(mask) > 0:
#             # 計算當前區域的平均 RGB 強度
#             color = get_average_rgb_v2(mask, rgb_img, qL=0.5, qR=0.9)[1]
#             r_mean, g_mean, b_mean = color
            
#             mask_dila = dila(mask, iterations=2)
#             nlabels = np.unique(merge_seg[mask_dila])

#             for nlabel in nlabels:
#                 if nlabel == 0 or nlabel == label:
#                     continue

#                 # 檢查合併後的集合大小是否超過限制
#                 combined_size = len(merge_groups[label].union(merge_groups[nlabel]))
#                 if combined_size > 4:
#                     if debug_mode:
#                         print(f"Skipping merge of {label} and {nlabel}: combined size {combined_size} exceeds limit")
#                     continue  # 跳過合併

#                 nmask = (merge_seg == nlabel)
#                 ncolor = get_average_rgb_v2(nmask, rgb_img, qL=0.5, qR=0.9)[1]
#                 nr_mean, ng_mean, nb_mean = ncolor

#                 # 計算顏色差異（DeltaE）
#                 delta_e = deltaE_ciede94(rgb2lab(color / 255.0), rgb2lab(ncolor / 255.0))

#                 # 計算 RGB 強度差異
#                 rgb_diff = np.sqrt((r_mean - nr_mean) ** 2 +
#                                    (g_mean - ng_mean) ** 2 +
#                                    (b_mean - nb_mean) ** 2)

#                 # 加權總差異：70% DeltaE + 30% RGB 差異
#                 total_diff = 0.7 * delta_e + 0.3 * rgb_diff

#                 # 判斷是否合併
#                 if total_diff <= merge:
#                     # 嘗試合併
#                     merge_seg[nmask] = label
#                     merge_groups[label] = merge_groups[label].union(merge_groups[nlabel])

#                     # 更新全局合併集合
#                     for sublabel in merge_groups[nlabel]:
#                         merge_groups[sublabel] = merge_groups[label]

#                     if debug_mode:
#                         print(f"Merge: {label} and {nlabel}, Total Diff: {total_diff:.2f}")

#                 # 合併後再次檢查集合大小，回滾合併
#                 if len(merge_groups[label]) > 4:
#                     # 回滾操作
#                     merge_seg[nmask] = nlabel  # 恢復標籤
#                     merge_groups[label] -= merge_groups[nlabel]  # 恢復集合
#                     if debug_mode:
#                         print(f"Rolling back merge of {label} and {nlabel}: group size exceeds limit")
#                     break

#     return merge_seg, relabeling

    # -------------------------------------------------------------------------/


def draw_label_on_image(seg: np.ndarray, rgb_img: np.ndarray,
                        relabeling: dict[str, str] = {}):
    """
    """
    assert rgb_img.dtype == np.uint8, "rgb_img.dtype != np.uint8"
    
    # 計算每個 label 區域的屬性
    props = measure.regionprops(seg)
    
    # 使用 PIL 在影像上寫入文字
    pil_img = Image.fromarray(rgb_img)
    draw = ImageDraw.Draw(pil_img)
    font = ImageFont.load_default()  # 使用預設字體，也可以指定自定義字體
    
    # 對每個區域的屬性進行迭代，並在重心位置標上 label 值
    for prop in props:
        
        cY, cX = prop.centroid
        
        if prop.label in relabeling:
            label_text = str(relabeling[prop.label])
        else:
            label_text = str(prop.label)
        
        # 計算文字的大小
        text_size = draw.textsize(label_text, font=font)
        text_width, text_height = text_size
        
        # 計算置中後的文字位置
        text_position = (cX - text_width / 2, cY - text_height / 2)
        
        # 畫陰影
        shadow_offset = 2 # pixels
        shadow_position = (text_position[0] + shadow_offset, text_position[1] + shadow_offset)
        draw.text(shadow_position, label_text, fill="#000000", font=font)
        
        # 畫主要文字
        if prop.label in relabeling:
            draw.text(text_position, label_text, fill="#FF0000", font=font)
        else:
            draw.text(text_position, label_text, fill="#FFFFFF", font=font)
    
    # check and return
    assert id(pil_img) != id(rgb_img)
    return np.array(pil_img)
    # -------------------------------------------------------------------------/


def single_slic_labeling(dst_dir:Path, img_path:Path,
                         n_segments:int, dark:int, merge:int,
                         debug_mode:bool=False):
    """
    """
    img_name = dst_dir.name
    # read image
    img = ski.io.imread(img_path)

    """ SLIC (seg0) """
    seg0 = slic(img,
                n_segments = n_segments,
                channel_axis=-1,
                convert2lab=True,
                enforce_connectivity=True,
                slic_zero=False, compactness=30,
                max_num_iter=100,
                sigma = [1.7,1.7],
                spacing=[1,1], # 3D: z, y, x; 2D: y, x
                min_size_factor=0.4,
                max_size_factor=3,
                start_label=0)
        # parameters can refer to https://scikit-image.org/docs/stable/api/skimage.segmentation.html#skimage.segmentation.slic

    """ Save 'SLIC' result (seg0, without any merge) """
    # save segmentation as pkl file
    save_path = dst_dir.joinpath(f"{img_name}.seg0.pkl")
    save_segment_result(save_path, seg0)
    # Mark `seg0` on `img`
    seg0_on_img = np.uint8(mark_boundaries(img, seg0, color=(0, 1, 1))*255)
    save_path = dst_dir.joinpath(f"{img_name}.seg0.png")
    ski.io.imsave(save_path, seg0_on_img)

    """ Merge background (seg1) """
    seg1 = deepcopy(seg0)
    labels = np.unique(seg0)
    new_label = np.max(labels) + 1 # new (re-index) label start
    for label in labels:
        mask = (seg1 == label)
        if np.sum(mask) > 0: # SLIC 生成的 labels 會跳號， mask 可能會沒東西
            color = get_average_rgb(mask, img, avg_ratio=1.0)[1]
            color_dist = simple_col_dis(color, (0, 0, 0)) # compare with 'background'
            if color_dist <= dark:
                seg1[mask] = 0 # background on `seg1` is set to 0
            else:
                seg1[mask] = new_label # re-index
                new_label +=1
        else:
            print(f"'{label}' is missing")

    """ Save 'Merge background' result (seg1) """
    # save segmentation as pkl file
    save_path = dst_dir.joinpath(f"{img_name}.seg1.pkl")
    save_segment_result(save_path, seg1)
    # Generate average 'RGB' of `img` and mark `seg1` labels on it
    avg_rgb = average_rgb_coloring(seg1, img)
    seg1_on_img = np.uint8(mark_boundaries(avg_rgb, seg1, color=(0, 1, 1))*255)
    seg1_on_img = draw_label_on_image(seg1, seg1_on_img)
    save_path = dst_dir.joinpath(f"{img_name}.seg1a.png")
    ski.io.imsave(save_path, seg1_on_img)

    """ Merge similar RGB (seg2) """
    seg2, relabeling = merge_similar_rgb(seg1, img,
                                         merge=merge, debug_mode=debug_mode)

    """ Save 'Merge similar RGB' result (seg2) """
    # save segmentation as pkl file
    save_path = dst_dir.joinpath(f"{img_name}.seg2.pkl")
    save_segment_result(save_path, seg2)
    # Mark `seg2` labels and merged regions on `img` and `avg_rgb`
    for k, v in {"o": img, "a": avg_rgb}.items():
        seg2_on_img = np.uint8(mark_boundaries(v, seg2, color=(0, 1, 1))*255)
        seg2_on_img = draw_label_on_image(seg1, seg2_on_img, relabeling=relabeling)
        save_path = dst_dir.joinpath(f"{img_name}.seg2{k}.png")
        ski.io.imsave(save_path, seg2_on_img)
    
    return seg1, seg2
    # -------------------------------------------------------------------------/


def single_cellpose_prediction(dst_dir: Path, debug_mode: bool=False):
    """ Function name TBD
    Place holder for running Cellpose prediction
    """
    # npy_file = dst_dir.joinpath("[xxx_seg.npy]")
    # seg1 = np.load(npy_file, allow_pickle=True).item()['masks']
    # -------------------------------------------------------------------------/


if __name__ == '__main__':

    """ Init components """
    console = Console()
    console.print(f"\nPy Module: '{Path(__file__)}'\n")
    
    # colloct image file names
    img_dir = Path(r"") # directory of input images, images extension: .tif / .tiff

    # scan TIFF files
    img_paths = list(img_dir.glob("*.tif*"))
    console.print(f"Found {len(img_paths)} TIFF files.")
    console.rule()

    """ Load config """
    config_name: str = "ml_analysis.toml"
    config = load_config(config_name)
    # [SLIC]
    n_segments: int  = config["SLIC"]["n_segments"]
    dark: int        = config["SLIC"]["dark"]
    merge: int       = config["SLIC"]["merge"]
    debug_mode: bool = config["SLIC"]["debug_mode"]
    # [seg_results]
    seg_desc = get_seg_desc(config)
    console.print(f"Config : '{config_name}'\n",
                    Pretty(config, expand_all=True))
    console.rule()
    
    for img_path in img_paths:
        
        # get `seg_dirname`
        if seg_desc == "SLIC":
            seg_param_name = get_slic_param_name(config)
        elif seg_desc == "Cellpose":
            seg_param_name = "model_id" # TBD
        seg_dirname = f"{img_path.stem}.{seg_param_name}"
        
        seg_dir = img_path.parent.joinpath(f"{seg_desc}/{seg_dirname}")
        create_new_dir(seg_dir)
        
        # generate cell segmentation
        if seg_desc == "SLIC":
            seg1, seg2 = single_slic_labeling(seg_dir, img_path,
                                              n_segments, dark, merge,
                                              debug_mode)
        elif seg_desc == "Cellpose":
            seg1, seg2 = single_cellpose_prediction()
        # Note : `seg1` is 1st merge (background), `seg2` is 2nd merge (color)
        
        # save cell segmentation feature
        analysis_dict = {}
        analysis_dict = update_seg_analysis_dict(analysis_dict, *count_element(seg1, "cell"))
        save_path = seg_dir.joinpath(f"cell_count_{analysis_dict['cell_count']}")
        with open(save_path, mode="w") as f_writer: pass # empty file
        
        console.print(f"'{seg_desc}' of '{img_path.name}' save to : '{save_path.parent}'")
    
    console.line()
    console.print("[green]Done! \n")
    # -------------------------------------------------------------------------/