# -*- coding: utf-8 -*-
"""
Crack removal evaluation:
1) 改动集中度：geom_score（掩码内变化 / 掩码外变化）
2) 背景保持度：mask 外的 MSE / PSNR / (可选) SSIM
"""

import os
import re
from collections import defaultdict

import numpy as np
import cv2
import matplotlib.pyplot as plt

# ========== 可选：SSIM，如果没有安装 skimage 就自动忽略 ==========
try:
    from skimage.metrics import structural_similarity as ssim
    HAS_SSIM = True
except ImportError:
    HAS_SSIM = False
    print("[INFO] 未安装 scikit-image，将不计算 SSIM。"
          "如需 SSIM：pip install scikit-image")


# =====================
# 路径设置（按需修改）
# =====================
background_folder = r"E:/AppData/PythonProjects/Daily_Practice/Crack_generation/task_2/base"
mask_folder       = r"E:/AppData/PythonProjects/Daily_Practice/Crack_generation/task_2/mask"
synth_folder      = r"E:/AppData/PythonProjects/Daily_Practice/Crack_generation/task_2/result/CrackRemove"

# 生成图命名格式示例：
# base1_00002_.png  -> base{base_id}_{sample_id}_.png
synth_pattern = re.compile(
    r"base(\d+)_([0-9]+)_\.png$",
    re.IGNORECASE
)

# 是否做少量可视化（调试用）
DEBUG_VIS = False           # True 则会画几张对比图
DEBUG_LIMIT = 5             # 最多显示多少张


# =====================
# 工具函数
# =====================

def imread_chs(path: str, flag=cv2.IMREAD_COLOR):
    """
    支持中文路径的 imread：np.fromfile + cv2.imdecode
    """
    data = np.fromfile(path, dtype=np.uint8)
    if data.size == 0:
        return None
    img = cv2.imdecode(data, flag)
    return img


def load_gt_mask(mask_path: str) -> np.ndarray:
    """
    读取裂纹掩码，假设黑底白裂纹，返回 0/255 二值图
    """
    gt = imread_chs(mask_path, cv2.IMREAD_GRAYSCALE)
    if gt is None:
        raise FileNotFoundError(f"无法读取掩码：{mask_path}")
    _, gt_bin = cv2.threshold(gt, 127, 255, cv2.THRESH_BINARY)
    return gt_bin


def mse(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    均方误差（MSE）
    """
    diff = img1.astype(np.float32) - img2.astype(np.float32)
    return float(np.mean(diff ** 2))


def psnr(img1: np.ndarray, img2: np.ndarray, max_val: float = 255.0) -> float:
    """
    峰值信噪比（PSNR）
    """
    m = mse(img1, img2)
    if m == 0:
        return float("inf")
    return 10.0 * np.log10((max_val ** 2) / m)


def geom_and_background_metrics(base_path: str,
                                synth_path: str,
                                mask_path: str):
    """
    计算一张图的：
    1) geom_score：改动集中度（掩码内亮度变化 / 掩码外亮度变化）
    2) 背景区域（掩码外）的 MSE / PSNR / (可选) SSIM

    base  : 含裂纹原图
    synth : 去裂后的结果图
    mask  : 裂纹掩码（白 = 裂纹区域）

    返回:
        geom_score, mse_bg, psnr_bg, ssim_bg, base_bgr, synth_bgr, gt_mask
    """
    # 读 base / synth
    base_bgr  = imread_chs(base_path,  cv2.IMREAD_COLOR)
    synth_bgr = imread_chs(synth_path, cv2.IMREAD_COLOR)
    if base_bgr is None or synth_bgr is None:
        raise FileNotFoundError(f"无法读取 base 或 result：\n"
                                f"  base  : {base_path}\n"
                                f"  result: {synth_path}")

    h, w = synth_bgr.shape[:2]
    if base_bgr.shape[:2] != (h, w):
        base_bgr = cv2.resize(base_bgr, (w, h), interpolation=cv2.INTER_LINEAR)

    # 读取掩码
    gt_mask = load_gt_mask(mask_path)
    if gt_mask.shape != (h, w):
        gt_mask = cv2.resize(gt_mask, (w, h), interpolation=cv2.INTER_NEAREST)

    # 转 Lab，取 L 通道做亮度差分（更贴近人眼感知）
    base_lab  = cv2.cvtColor(base_bgr,  cv2.COLOR_BGR2LAB)
    synth_lab = cv2.cvtColor(synth_bgr, cv2.COLOR_BGR2LAB)
    L_base,  _, _ = cv2.split(base_lab)
    L_synth, _, _ = cv2.split(synth_lab)

    delta = cv2.absdiff(L_synth, L_base).astype(np.float32)

    m_in  = gt_mask > 0   # 裂纹区域
    m_out = ~m_in         # 非裂纹 / 背景

    eps = 1e-6
    if np.sum(m_in) == 0 or np.sum(m_out) == 0:
        # 没有掩码或全部都是掩码，无法区分内外，退化为 1
        geom_score = 1.0
    else:
        mean_in  = float(delta[m_in].mean())   # 裂纹区域改动
        mean_out = float(delta[m_out].mean())  # 背景区域改动
        geom_score = mean_in / (mean_out + eps)

    # 背景一致性：只在 mask 外算 MSE / PSNR / SSIM
    if np.sum(m_out) == 0:
        mse_bg = 0.0
        psnr_bg = float("inf")
        ssim_bg = None
    else:
        # 只保留 mask 外像素，这里在 RGB 上算 MSE/PSNR
        base_bg  = base_bgr.copy()
        synth_bg = synth_bgr.copy()
        base_bg[m_in]  = 0
        synth_bg[m_in] = 0

        mse_bg = mse(base_bg, synth_bg)
        psnr_bg = psnr(base_bg, synth_bg)

        if HAS_SSIM:
            # SSIM 用灰度图算即可
            base_gray  = cv2.cvtColor(base_bgr,  cv2.COLOR_BGR2GRAY)
            synth_gray = cv2.cvtColor(synth_bgr, cv2.COLOR_BGR2GRAY)
            base_gray  = base_gray.astype(np.float32)
            synth_gray = synth_gray.astype(np.float32)

            # 只在 mask 外区域算 SSIM：用 mask 乘起来再送进 ssim
            mask_out_float = m_out.astype(np.float32)
            base_gray_bg  = base_gray  * mask_out_float
            synth_gray_bg = synth_gray * mask_out_float

            ssim_bg = float(
                ssim(base_gray_bg, synth_gray_bg,
                     data_range=base_gray_bg.max() - base_gray_bg.min() + eps)
            )
        else:
            ssim_bg = None

    return geom_score, mse_bg, psnr_bg, ssim_bg, base_bgr, synth_bgr, gt_mask


# =====================
# 主流程
# =====================

def main():
    files = [f for f in os.listdir(synth_folder)
             if f.lower().endswith(".png")]
    files.sort()

    print(f"共发现结果图 {len(files)} 张，开始评估裂纹去除效果…\n")

    combo_geom   = defaultdict(list)
    combo_mse_bg = defaultdict(list)
    combo_psnr_bg = defaultdict(list)
    combo_ssim_bg = defaultdict(list)

    all_geom = []
    all_mse_bg = []
    all_psnr_bg = []
    all_ssim_bg = []

    debug_count = 0

    for fname in files:
        m = synth_pattern.fullmatch(fname)
        if not m:
            # 名字不符合 base{idx}_{id}_.png 就跳过
            # 例如 base1_00001_.png
            print(f"[WARN] 文件名不符合规则，已跳过：{fname}")
            continue

        base_id  = int(m.group(1))  # 对应 base 文件夹里的 1.jpg / 1.png ...
        crack_id = int(m.group(2))  # 这里把第二个数字当作“样本编号”

        synth_path   = os.path.join(synth_folder, fname)
        gt_mask_path = os.path.join(mask_folder, f"{base_id}.png")

        if not os.path.exists(gt_mask_path):
            print(f"[WARN] 找不到掩码：{gt_mask_path}，跳过 {fname}")
            continue

        # base 路径（可能是 jpg 或 png）
        base_path_jpg = os.path.join(background_folder, f"{base_id}.jpg")
        base_path_png = os.path.join(background_folder, f"{base_id}.png")
        if os.path.exists(base_path_jpg):
            base_path_used = base_path_jpg
        elif os.path.exists(base_path_png):
            base_path_used = base_path_png
        else:
            print(f"[WARN] 找不到 base 图：{base_id}.jpg / {base_id}.png，跳过 {fname}")
            continue

        try:
            geom_score, mse_bg, psnr_bg, ssim_bg, base_bgr, synth_bgr, gt_mask = \
                geom_and_background_metrics(
                    base_path_used,
                    synth_path,
                    gt_mask_path
                )
        except FileNotFoundError as e:
            print("[ERROR]", e)
            continue

        # (base_id, crack_id) 组合做统计
        combo_key = (base_id, crack_id)

        combo_geom[combo_key].append(geom_score)
        combo_mse_bg[combo_key].append(mse_bg)
        combo_psnr_bg[combo_key].append(psnr_bg)
        if ssim_bg is not None:
            combo_ssim_bg[combo_key].append(ssim_bg)

        all_geom.append(geom_score)
        all_mse_bg.append(mse_bg)
        all_psnr_bg.append(psnr_bg)
        if ssim_bg is not None:
            all_ssim_bg.append(ssim_bg)

        print(f"{fname:<35} -> "
              f"集中度 geom={geom_score:6.3f}, "
              f"MSE_bg={mse_bg:8.2f}, "
              f"PSNR_bg={psnr_bg:6.2f}"
              + (f", SSIM_bg={ssim_bg:5.3f}" if ssim_bg is not None else "")
              )

        # 调试可视化
        if DEBUG_VIS and debug_count < DEBUG_LIMIT:
            debug_count += 1

            plt.figure(figsize=(12, 4))
            plt.subplot(1, 3, 1)
            plt.title(f"Base {base_id} (有裂纹)")
            plt.imshow(cv2.cvtColor(base_bgr, cv2.COLOR_BGR2RGB))
            plt.axis("off")

            plt.subplot(1, 3, 2)
            plt.title(f"Result {fname} (去裂)")
            plt.imshow(cv2.cvtColor(synth_bgr, cv2.COLOR_BGR2RGB))
            plt.axis("off")

            plt.subplot(1, 3, 3)
            plt.title(f"Mask (裂纹区域)")
            plt.imshow(gt_mask, cmap="gray")
            plt.axis("off")

            plt.suptitle(
                f"geom={geom_score:.3f}, "
                f"MSE_bg={mse_bg:.1f}, "
                f"PSNR_bg={psnr_bg:.2f}"
                + (f", SSIM_bg={ssim_bg:.3f}" if ssim_bg is not None else "")
            )
            plt.tight_layout()
            plt.show()

    # ========== 统计 ==========
    print("\n==============================")
    print(" 每种 (base_i, sample_j) 的平均指标（裂纹去除）")
    print("==============================")

    keys_sorted = sorted(combo_geom.keys())

    for (b, c) in keys_sorted:
        geom_mean = float(np.mean(combo_geom[(b, c)]))
        mse_mean  = float(np.mean(combo_mse_bg[(b, c)]))
        psnr_mean = float(np.mean(combo_psnr_bg[(b, c)]))
        if combo_ssim_bg.get((b, c)):
            ssim_mean = float(np.mean(combo_ssim_bg[(b, c)]))
            ssim_str  = f", mean SSIM_bg={ssim_mean:.3f}"
        else:
            ssim_str  = ""

        print(f"base{b}_sample{c}: n={len(combo_geom[(b, c)]):3d}, "
              f"mean geom={geom_mean:.3f}, "
              f"mean MSE_bg={mse_mean:.1f}, "
              f"mean PSNR_bg={psnr_mean:.2f}"
              f"{ssim_str}"
              )

    print("\n==============================")
    print(" 整体平均指标（裂纹去除）")
    print("==============================")

    if all_geom:
        print(f"overall mean geom      = {float(np.mean(all_geom)):.3f}")
        print(f"overall mean MSE_bg    = {float(np.mean(all_mse_bg)):.1f}")
        print(f"overall mean PSNR_bg   = {float(np.mean(all_psnr_bg)):.2f}")
        if all_ssim_bg:
            print(f"overall mean SSIM_bg   = {float(np.mean(all_ssim_bg)):.3f}")
    else:
        print("没有成功评估任何样本，请检查路径和命名。")


if __name__ == "__main__":
    main()
