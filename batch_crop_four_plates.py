#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import os
import sys

# 新增: HSV 饱和度检测透明边框
def detect_plate_by_hsv(img_full, sat_thresh=40, min_area_ratio=0.003):
    """
    使用 HSV 饱和度分量检测透明 MMM 平板边框。饱和度较低的边框区域成为 mask，进一步通过轮廓提取外接矩形。
    """
    H, W = img_full.shape[:2]
    image_area = H * W

    hsv = cv2.cvtColor(img_full, cv2.COLOR_BGR2HSV)
    # 饱和度低于阈值，可能为透明边框
    sat = hsv[:, :, 1]
    _, mask = cv2.threshold(sat, sat_thresh, 255, cv2.THRESH_BINARY_INV)

    # 形态学闭运算填充小孔
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (41, 41))
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rects = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < image_area * min_area_ratio:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        if w * h > image_area * 0.9:
            continue
        rects.append((x, y, x + w, y + h))

    # 按面积降序并保留最大四个
    rects = sorted(rects, key=lambda r: (r[2]-r[0])*(r[3]-r[1]), reverse=True)[:4]
    return rects

def merge_rects(rects, overlapThresh=0.3):
    """
    合并重叠或接近的矩形框，避免单个 plate 被分割成多个区域。
    矩形格式: (x1, y1, x2, y2)
    归一化成 numpy 数组，用简单的非最大抑制思想（NMS）合并重叠超过阈值的框。
    """
    if not rects:
        return []

    boxes = np.array(rects)
    x1 = boxes[:,0]; y1 = boxes[:,1]; x2 = boxes[:,2]; y2 = boxes[:,3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)
    pick = []

    while len(idxs) > 0:
        last = idxs[-1]
        pick.append(last)
        xx1 = np.maximum(x1[last], x1[idxs[:-1]])
        yy1 = np.maximum(y1[last], y1[idxs[:-1]])
        xx2 = np.minimum(x2[last], x2[idxs[:-1]])
        yy2 = np.minimum(y2[last], y2[idxs[:-1]])
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        inter = w * h
        iou = inter / (areas[last] + areas[idxs[:-1]] - inter)
        keep_idxs = np.where(iou <= overlapThresh)[0]
        idxs = idxs[keep_idxs]

    merged = boxes[pick].tolist()
    return merged

def detect_plate_by_hough(img_full,
                          canny_thresh1=50,
                          canny_thresh2=150,
                          hough_thresh=100,
                          min_line_len_ratio=0.5,
                          max_line_gap=20,
                          margin=10):
    """
    专门针对透明度高的 MMM 平板：使用 HoughLinesP 找四条最外侧长直线（水平和垂直），
    计算四边形边界后裁剪。新增 CLAHE 增强对比。
    """
    H, W = img_full.shape[:2]
    # --- 新增：使用 CLAHE 增强对比 ---
    gray = cv2.cvtColor(img_full, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, canny_thresh1, canny_thresh2)
    # -------------------------------

    min_line_len = int(min(W, H) * min_line_len_ratio)
    lines = cv2.HoughLinesP(
        edges, rho=1, theta=np.pi / 180,
        threshold=hough_thresh, minLineLength=min_line_len,
        maxLineGap=max_line_gap
    )
    if lines is None:
        return []

    lines = lines.reshape(-1, 4)
    horiz_ys = []
    vert_xs = []
    for x1, y1, x2, y2 in lines:
        dx = x2 - x1; dy = y2 - y1
        # 水平线
        if abs(dy) <= 5 and abs(dx) >= min_line_len:
            ys = (y1 + y2) // 2
            horiz_ys.append(ys)
        # 垂直线
        if abs(dx) <= 5 and abs(dy) >= min_line_len:
            xs = (x1 + x2) // 2
            vert_xs.append(xs)

    if len(horiz_ys) < 2 or len(vert_xs) < 2:
        return []

    top = min(horiz_ys)
    bottom = max(horiz_ys)
    left = min(vert_xs)
    right = max(vert_xs)

    # 增加 margin
    y1 = max(0, top - margin)
    y2 = min(H, bottom + margin)
    x1 = max(0, left - margin)
    x2 = min(W, right + margin)

    return [(x1, y1, x2, y2)]

def detect_plate_mmm_adaptive(img_full, min_area_ratio=0.02):
    """
    专门针对透明度高的 MMM 平板：使用自适应阈值 + 大核闭运算，将弱边框连成整体轮廓，提取外接矩形。
    """
    H, W = img_full.shape[:2]
    image_area = H * W

    # 1. 灰度 + CLAHE + 较小高斯模糊
    gray = cv2.cvtColor(img_full, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)
    blur = cv2.GaussianBlur(gray, (15, 15), 0)

    # 2. 自适应阈值 (倒相)，使板边框成为白色连通区域
    thresh = cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        81, 2
    )

    # 3. 大核闭运算，连接散碎边缘（核缩小）
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (31, 31))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # 4. 查找外部轮廓，提取外接矩形
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rects = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < image_area * 0.001:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        # 过滤掉过大的区域（可能是整张图）
        if w * h > image_area * 0.9:
            continue
        rects.append((x, y, x + w, y + h))

    # 按面积排序，并保留最大的前 4 个
    rects = sorted(rects, key=lambda r: (r[2]-r[0]) * (r[3]-r[1]), reverse=True)
    return rects[:4]

def detect_plate_by_border(img_full,
                           canny_thresh1=50,
                           canny_thresh2=150,
                           min_area_ratio=0.05):
    """
    当 colony-based 方法失败时，使用边缘检测 + 轮廓近似矩形来定位 plate 边框。
    返回一个或多个矩形区域 (x1, y1, x2, y2)。
    """
    H, W = img_full.shape[:2]
    image_area = H * W

    gray = cv2.cvtColor(img_full, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, canny_thresh1, canny_thresh2)

    # 进行形态学闭运算，填充边缘小断
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    # 查找轮廓
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rects = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < image_area * 0.001:
            continue
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        # 近似矩形 (顶点 >= 4)
        if len(approx) >= 4:
            x, y, w, h = cv2.boundingRect(approx)
            rects.append((x, y, x + w, y + h))

    if not rects:
        return []

    # 合并重叠，并过滤几乎覆盖全图的框
    merged = merge_rects(rects, overlapThresh=0.3)
    valid = []
    for (x1, y1, x2, y2) in merged:
        w = x2 - x1; h = y2 - y1
        if w * h >= image_area * 0.90:
            continue  # 太大，很可能是整张图
        valid.append((x1, y1, x2, y2))
    return valid

def detect_plate_regions(img_full,
                         blur_ksize=(5,5),
                         thresh_method=cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
                         open_kernel=(5,5),
                         dilate_kernel=(25,25),
                         min_area_ratio=0.05,
                         margin=20):
    """
    使用阈值 + 连通组件提取 plate 区域:
    1. 转灰度, 高斯模糊
    2. 使用 Otsu 二值化 (反向), 将菌落 (暗) 变为白
    3. 开运算去除单个小噪点 (open_kernel)
    4. 膨胀 (dilate_kernel) 将一个 plate 内的菌落连成一块, 但避免不同 plate 膨胀合并
    5. 计算连通组件, 对每个组件按面积过滤, 生成边界框
    6. 对所有边界框调用 merge_rects 合并, 并剔除几乎覆盖全图的框
    7. 若无有效框, 调用 detect_plate_by_border 作为回退方法
    8. 返回最终边界框列表 (x1,y1,x2,y2)
    """
    H, W = img_full.shape[:2]
    image_area = H * W

    gray = cv2.cvtColor(img_full, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, blur_ksize, 0)
    _, thresh = cv2.threshold(blur, 0, 255, thresh_method)
    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, open_kernel)
    opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_open)
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, dilate_kernel)
    dilated = cv2.dilate(opened, kernel_dilate, iterations=1)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(dilated, connectivity=8)
    raw_rects = []
    for label in range(1, num_labels):
        x = stats[label, cv2.CC_STAT_LEFT]
        y = stats[label, cv2.CC_STAT_TOP]
        w = stats[label, cv2.CC_STAT_WIDTH]
        h = stats[label, cv2.CC_STAT_HEIGHT]
        area = stats[label, cv2.CC_STAT_AREA]
        if area < image_area * 0.01:
            continue
        raw_rects.append((x, y, x + w, y + h))

    valid_rects = []
    if raw_rects:
        merged = merge_rects(raw_rects, overlapThresh=0.3)
        for (x1, y1, x2, y2) in merged:
            w = x2 - x1; h = y2 - y1
            # 排除几乎覆盖全图的框
            if w * h >= image_area * 0.90:
                continue
            valid_rects.append((x1, y1, x2, y2))

    # 如果 colony-based 没找到有效区域, 进行边框检测
    if not valid_rects:
        valid_rects = detect_plate_by_border(
            img_full,
            canny_thresh1=50,
            canny_thresh2=150,
            min_area_ratio=min_area_ratio
        )

    return valid_rects

def batch_crop_four_plates():
    # 1. Raw 目录
    raw_dir = os.path.join(
        os.getcwd(),
        "wetransfer_processed-zip_2025-06-05_1049",
        "Raw"
    )
    if not os.path.isdir(raw_dir):
        print(f"Error: 找不到 Raw 目录 {raw_dir}")
        sys.exit(1)

    # 2. 输出目录
    processed_root = os.path.join(os.path.dirname(raw_dir), "Processed")
    os.makedirs(processed_root, exist_ok=True)

    # 3. 列出所有 .jpg/.jpeg
    allowed_ext = {".jpg", ".jpeg", ".JPG", ".JPEG"}
    raw_files = [f for f in os.listdir(raw_dir)
                 if os.path.splitext(f)[1] in allowed_ext]
    if not raw_files:
        print("Error: Raw 目录下没有找到任何 .jpg/.jpeg 文件。")
        sys.exit(1)

    print(f"在 Raw 目录下检测到 {len(raw_files)} 张图片，开始批量处理：")

    for filename in sorted(raw_files):
        full_path = os.path.join(raw_dir, filename)
        base_name, _ = os.path.splitext(filename)
        print(f"\n-> 处理: {filename}")

        try:
            img_full = cv2.imread(full_path)
            if img_full is None:
                raise RuntimeError("cv2.imread 失败，无法加载图像。")

            H, W = img_full.shape[:2]

            if "MMM" in base_name:
                # *** MMM 图片检测逻辑（针对透明边框优化）：先尝试边框检测，再自适应阈值检测，最后使用 Hough/SAT 方法 ***
                # 1) 边框检测（宽松阈值）快速定位
                rects = detect_plate_by_border(
                    img_full,
                    canny_thresh1=20,
                    canny_thresh2=80,
                    min_area_ratio=0.001
                )
                if not rects:
                    # 2) 自适应阈值 + 大核闭运算
                    rects = detect_plate_mmm_adaptive(img_full, min_area_ratio=0.001)
                if not rects:
                    # 3) Hough 直线检测
                    rects = detect_plate_by_hough(
                        img_full,
                        canny_thresh1=50,
                        canny_thresh2=150,
                        hough_thresh=80,
                        min_line_len_ratio=0.4,
                        max_line_gap=30,
                        margin=15
                    )
                if not rects:
                    # 4) HSV 饱和度检测
                    rects = detect_plate_by_hsv(
                        img_full,
                        sat_thresh=40,
                        min_area_ratio=0.001
                    )
                # 保留面积最大的前 4 个区域
                if len(rects) > 4:
                    rects = sorted(rects, key=lambda r: (r[2]-r[0])*(r[3]-r[1]), reverse=True)[:4]
                # 应用动态 padding（5% 宽高）
                padded_rects = []
                for (x, y, x2, y2) in rects:
                    rw = x2 - x; rh = y2 - y
                    pad_x = int(rw * 0.05)
                    pad_y = int(rh * 0.05)
                    x1p = max(0, x - pad_x)
                    y1p = max(0, y - pad_y)
                    x2p = min(W, x2 + pad_x)
                    y2p = min(H, y2 + pad_y)
                    padded_rects.append((x1p, y1p, x2p, y2p))
                rects = padded_rects
            else:
                # *** R5 或其他：优先使用边框检测，再 fallback 菌落区域方法 ***
                rects = detect_plate_by_border(
                    img_full,
                    canny_thresh1=30,
                    canny_thresh2=120,
                    min_area_ratio=0.01
                )
                if not rects:
                    rects = detect_plate_regions(
                        img_full,
                        blur_ksize=(5,5),
                        thresh_method=cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
                        open_kernel=(3,3),
                        dilate_kernel=(15,15),
                        min_area_ratio=0.02,
                        margin=20
                    )
                if len(rects) > 4:
                    rects = sorted(rects, key=lambda r: (r[2]-r[0])*(r[3]-r[1]), reverse=True)[:4]
                # 动态 padding：以矩形宽高 5% 作为 padding
                padded_rects = []
                for (x, y, x2, y2) in rects:
                    rw = x2 - x; rh = y2 - y
                    pad_x = int(rw * 0.05)
                    pad_y = int(rh * 0.05)
                    x1p = max(0, x - pad_x)
                    y1p = max(0, y - pad_y)
                    x2p = min(W, x2 + pad_x)
                    y2p = min(H, y2 + pad_y)
                    padded_rects.append((x1p, y1p, x2p, y2p))
                rects = padded_rects

            # 多余区域已在各自分支中处理，这里不再额外截断（保留最多 4 个）

            if not rects:
                print("    [警告] 未检测到任何 plate，跳过此文件。")
                continue

            subdir = os.path.join(processed_root, base_name)
            os.makedirs(subdir, exist_ok=True)

            for idx, (x1, y1, x2, y2) in enumerate(rects, start=1):
                cropped = img_full[y1:y2, x1:x2]
                out_name = f"{base_name}_{idx}.jpeg"
                out_path = os.path.join(subdir, out_name)
                cv2.imwrite(out_path, cropped)
                print(f"    Saved: {out_path}")

            print(f"    共检测到 {len(rects)} 个 plate 并裁剪保存。")

        except Exception as e:
            print(f"    [错误] {filename} 处理异常: {e}")

    print("\n批量处理完成。")

if __name__ == "__main__":
    batch_crop_four_plates()
