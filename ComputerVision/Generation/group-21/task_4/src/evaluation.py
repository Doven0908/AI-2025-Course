import os
import re
from collections import defaultdict
import matplotlib.pyplot as plt
import cv2
import numpy as np


# =====================
# 路径设置（按需修改）
# =====================
background_folder = r"E:/zwd/博一/课程/人工智能原理与应用/task_4/base"
mask_folder       = r"E:/zwd/博一/课程/人工智能原理与应用/task_4/mask"
synth_folder      = r"E:/zwd/博一/课程/人工智能原理与应用/task_4/CrackSynthesis"

# 生成图命名格式示例：
# base1_crack1_00002_.png
# base{i}_crack{j}_{任意数字}_ .png
synth_pattern = re.compile(
    r"base(\d+)_crack(\d+)_\d+_\.png",
    re.IGNORECASE
)


# =====================
# 掩码与指标函数
# =====================
def imread_chs(path: str, flag=cv2.IMREAD_GRAYSCALE):
    """
    兼容中文路径的 imread：用 np.fromfile + cv2.imdecode
    """
    data = np.fromfile(path, dtype=np.uint8)
    if data.size == 0:
        return None
    img = cv2.imdecode(data, flag)
    return img

def load_gt_mask(mask_path: str) -> np.ndarray:
    """
    读取 GT 裂纹掩码（mask 目录中 j.png），
    假设为黑底白裂纹。
    返回二值图：0 / 255
    """
    gt = imread_chs(mask_path, cv2.IMREAD_GRAYSCALE)
    if gt is None:
        raise FileNotFoundError(f"无法读取 GT 掩码：{mask_path}")
    _, gt_bin = cv2.threshold(gt, 127, 255, cv2.THRESH_BINARY)
    return gt_bin


def extract_crack_mask_from_synth_with_base(
    synth_path: str,
    base_id: int,
    background_folder: str,
    target_shape=None
) -> np.ndarray:
    """
    使用 合成图 - 原始 base 图 的差异来提取裂纹区域：
    1. 读取 base{i}.jpg 和合成图（支持中文路径）
    2. 转 Lab，取 L 通道
    3. 计算 |L_synth - L_base| 作为差分图
    4. 对差分图做阈值 + 形态学 + 去小连通域，得到裂纹掩码
    """

    # 1) 读取合成图
    synth_bgr = imread_chs(synth_path, cv2.IMREAD_COLOR)
    if synth_bgr is None:
        raise FileNotFoundError(f"无法读取生成图：{synth_path}")

    # 2) 读取对应的 base 图（既可能是 jpg 也可能是 png，这里都尝试一下）
    base_path_jpg = os.path.join(background_folder, f"{base_id}.jpg")
    base_path_png = os.path.join(background_folder, f"{base_id}.png")

    if os.path.exists(base_path_jpg):
        base_bgr = imread_chs(base_path_jpg, cv2.IMREAD_COLOR)
        base_path_used = base_path_jpg
    elif os.path.exists(base_path_png):
        base_bgr = imread_chs(base_path_png, cv2.IMREAD_COLOR)
        base_path_used = base_path_png
    else:
        raise FileNotFoundError(f"找不到 base 图：{base_path_jpg} 或 {base_path_png}")

    if base_bgr is None:
        raise FileNotFoundError(f"无法读取 base 图：{base_path_used}")

    # 3) 把 base resize 到和合成图相同大小（你的合成流程一般会保持大小一致，但这里保险一点）
    h, w = synth_bgr.shape[:2]
    if base_bgr.shape[:2] != (h, w):
        base_bgr = cv2.resize(base_bgr, (w, h), interpolation=cv2.INTER_LINEAR)

    # 4) 转 Lab，取 L 通道
    synth_lab = cv2.cvtColor(synth_bgr, cv2.COLOR_BGR2LAB)
    base_lab  = cv2.cvtColor(base_bgr,  cv2.COLOR_BGR2LAB)

    L_synth, _, _ = cv2.split(synth_lab)
    L_base,  _, _ = cv2.split(base_lab)

    # 5) 亮度差分：裂纹区域亮度变化会比较明显
    diff_L = cv2.absdiff(L_synth, L_base)

    # 可选：稍微平滑一下，去噪
    diff_blur = cv2.GaussianBlur(diff_L, (5, 5), 0)

    # 6) 归一化到 [0, 255]，便于阈值
    diff_norm = cv2.normalize(
        diff_blur, None, alpha=0, beta=255,
        norm_type=cv2.NORM_MINMAX
    ).astype(np.uint8)

    # 7) 用 Otsu 做二值化：差异大的区域 -> 255
    _, bin_mask = cv2.threshold(
        diff_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # 8) 形态学操作：开运算去小噪点，闭运算连通
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    bin_clean = cv2.morphologyEx(bin_mask, cv2.MORPH_OPEN, k, iterations=1)
    bin_clean = cv2.morphologyEx(bin_clean, cv2.MORPH_CLOSE, k, iterations=1)

    # 9) 去掉面积特别小的碎片（噪点）
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        bin_clean, connectivity=8
    )
    # 最小面积阈值可以按图像大小调，这里先给个经验值
    min_area = 80
    filtered = np.zeros_like(bin_clean)
    for i in range(1, num_labels):  # 0 是背景
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            filtered[labels == i] = 255

    # 10) resize 到 GT 大小
    if target_shape is not None and (filtered.shape != target_shape):
        filtered = cv2.resize(
            filtered,
            (target_shape[1], target_shape[0]),
            interpolation=cv2.INTER_NEAREST
        )

    return filtered, base_bgr, synth_bgr


def iou_from_masks(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """
    根据预测掩码与 GT 掩码计算 IoU。
    mask 为 0 / 255 图像。
    """
    pred = pred_mask > 0
    gt   = gt_mask > 0

    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()

    if union == 0:
        # 极端情况：两者全背景
        return 1.0
    return inter / union


def dice_from_masks(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """
    Dice 系数（F1）
    """
    pred = pred_mask > 0
    gt   = gt_mask > 0

    inter = np.logical_and(pred, gt).sum()
    s = pred.sum() + gt.sum()
    if s == 0:
        return 1.0
    return 2 * inter / s


# =====================
# 主流程
# =====================

def main():
    # (base_i, crack_j) -> 指标列表
    combo_ious = defaultdict(list)
    combo_dices = defaultdict(list)

    # 遍历所有生成图
    files = [f for f in os.listdir(synth_folder)
             if f.lower().endswith(".png")]

    if not files:
        print("Cracksynthesis 目录里没有找到 png 文件，请检查路径。")
        return

    files.sort()
    print(f"共发现生成图 {len(files)} 张，开始计算 IoU…\n")

    for fname in files:
        m = synth_pattern.fullmatch(fname)
        if not m:
            print('非 base{i}_crack{j}_xxxx_.png 格式的跳过')
            continue

        base_id = int(m.group(1))
        crack_id = int(m.group(2))

        synth_path = os.path.join(synth_folder, fname)
        gt_mask_path = os.path.join(mask_folder, f"{crack_id}.png")

        if not os.path.exists(gt_mask_path):
            print(f"[WARN] 找不到 GT mask：{gt_mask_path}，跳过 {fname}")
            continue

        try:
            gt_mask = load_gt_mask(gt_mask_path)
            pred_mask, base_bgr, synth_bgr = extract_crack_mask_from_synth_with_base(
                synth_path,
                base_id=base_id,
                background_folder=background_folder,
                target_shape=gt_mask.shape
            )

            # ==== 预览两种mask ====
            plt.figure(figsize=(12, 4))

            plt.subplot(1, 3, 1)
            plt.title(f"Base {base_id}")
            plt.imshow(cv2.cvtColor(base_bgr, cv2.COLOR_BGR2RGB))
            plt.axis("off")

            plt.subplot(1, 3, 2)
            plt.title(f"Synth {fname}")
            plt.imshow(cv2.cvtColor(synth_bgr, cv2.COLOR_BGR2RGB))
            plt.axis("off")

            plt.subplot(1, 3, 3)
            plt.title("Pred Mask (diff-based)")
            plt.imshow(pred_mask, cmap="gray")
            plt.axis("off")

            plt.show()
            # ==========================

        except FileNotFoundError as e:
            print("[ERROR]", e)
            continue

        iou = iou_from_masks(pred_mask, gt_mask)
        dice = dice_from_masks(pred_mask, gt_mask)

        combo_ious[(base_id, crack_id)].append(iou)
        combo_dices[(base_id, crack_id)].append(dice)

        print(
            f"处理 {fname:<30}  ->  "
            f"IoU={iou:.4f}, Dice={dice:.4f}"
        )

    print("\n==============================")
    print(" 每种 (base_i, crack_j) 组合的平均指标")
    print("==============================")

    all_ious = []
    all_dices = []

    for (b, c) in sorted(combo_ious.keys()):
        ious = combo_ious[(b, c)]
        dices = combo_dices[(b, c)]
        mean_iou = float(np.mean(ious)) if ious else 0.0
        mean_dice = float(np.mean(dices)) if dices else 0.0

        all_ious.extend(ious)
        all_dices.extend(dices)

        print(
            f"base{b}_crack{c}: "
            f"n={len(ious):3d},  "
            f"mean IoU={mean_iou:.4f},  "
            f"mean Dice={mean_dice:.4f}"
        )

    print("\n==============================")
    print(" 全部样本的整体平均指标")
    print("==============================")

    if all_ious:
        overall_iou = float(np.mean(all_ious))
        overall_dice = float(np.mean(all_dices))
        print(f"Overall mean IoU  = {overall_iou:.4f}")
        print(f"Overall mean Dice = {overall_dice:.4f}")
    else:
        print("没有计算到任何 IoU（可能匹配不到文件名或路径有误）。")


if __name__ == "__main__":
    main()
