#!/usr/bin/env python3
"""
检查特定孔位的菌落检测情况
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path


def check_specific_wells(image_path, result_dir, wells_to_check=['E6', 'H12']):
    """检查特定孔位的检测情况"""

    # 加载图像
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]

    # 计算孔板网格
    rows, cols = 8, 12
    margin_y = h * 0.05
    margin_x = w * 0.05

    usable_height = h - 2 * margin_y
    usable_width = w - 2 * margin_x

    cell_height = usable_height / rows
    cell_width = usable_width / cols

    # 创建可视化
    fig, axes = plt.subplots(1, len(wells_to_check) + 1, figsize=(15, 5))

    # 显示全图
    axes[0].imshow(img_rgb)
    axes[0].set_title('Full Image with Target Wells')

    # 标记目标孔位
    for well_id in wells_to_check:
        row_idx = ord(well_id[0]) - ord('A')
        col_idx = int(well_id[1:]) - 1

        center_y = margin_y + (row_idx + 0.5) * cell_height
        center_x = margin_x + (col_idx + 0.5) * cell_width

        # 在全图上标记
        rect = plt.Rectangle((center_x - cell_width/2, center_y - cell_height/2),
                             cell_width, cell_height,
                             fill=False, color='red', linewidth=3)
        axes[0].add_patch(rect)
        axes[0].text(center_x, center_y, well_id,
                     color='red', fontsize=16, ha='center', va='center',
                     bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.7))

    # 显示每个孔位的放大图
    for i, well_id in enumerate(wells_to_check):
        row_idx = ord(well_id[0]) - ord('A')
        col_idx = int(well_id[1:]) - 1

        center_y = int(margin_y + (row_idx + 0.5) * cell_height)
        center_x = int(margin_x + (col_idx + 0.5) * cell_width)

        # 提取孔位区域（稍微扩大一点）
        pad = int(min(cell_height, cell_width) * 0.2)
        y1 = max(0, int(center_y - cell_height/2 - pad))
        y2 = min(h, int(center_y + cell_height/2 + pad))
        x1 = max(0, int(center_x - cell_width/2 - pad))
        x2 = min(w, int(center_x + cell_width/2 + pad))

        well_img = img_rgb[y1:y2, x1:x2]

        axes[i+1].imshow(well_img)
        axes[i+1].set_title(f'Well {well_id}')

        # 标记中心点
        local_center_y = center_y - y1
        local_center_x = center_x - x1
        axes[i+1].plot(local_center_x, local_center_y, 'r+',
                       markersize=20, markeredgewidth=3)

        # 显示搜索半径
        search_radius = min(cell_height, cell_width) * 0.6
        circle = plt.Circle((local_center_x, local_center_y), search_radius,
                            fill=False, color='yellow', linewidth=2)
        axes[i+1].add_patch(circle)

        # 扩展搜索半径
        extended_radius = search_radius * 1.5
        circle2 = plt.Circle((local_center_x, local_center_y), extended_radius,
                             fill=False, color='orange', linewidth=2, linestyle='--')
        axes[i+1].add_patch(circle2)

    plt.tight_layout()

    # 保存图像
    output_path = Path(result_dir) / 'well_check.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ 检查结果保存到: {output_path}")

    # 分析这些位置的图像特征
    print("\n分析目标孔位的图像特征：")
    for well_id in wells_to_check:
        row_idx = ord(well_id[0]) - ord('A')
        col_idx = int(well_id[1:]) - 1

        center_y = int(margin_y + (row_idx + 0.5) * cell_height)
        center_x = int(margin_x + (col_idx + 0.5) * cell_width)

        # 提取小区域进行分析
        radius = int(min(cell_height, cell_width) * 0.3)
        y1 = max(0, center_y - radius)
        y2 = min(h, center_y + radius)
        x1 = max(0, center_x - radius)
        x2 = min(w, center_x + radius)

        region = img_rgb[y1:y2, x1:x2]

        # 计算一些基本特征
        mean_intensity = np.mean(region)
        std_intensity = np.std(region)

        # 检查是否有菌落（简单的阈值检测）
        gray_region = cv2.cvtColor(region, cv2.COLOR_RGB2GRAY)
        _, binary = cv2.threshold(
            gray_region, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        white_ratio = np.sum(binary == 255) / binary.size

        print(f"\n{well_id}:")
        print(f"  位置: ({center_x}, {center_y})")
        print(f"  平均强度: {mean_intensity:.1f}")
        print(f"  强度标准差: {std_intensity:.1f}")
        print(f"  白色区域比例: {white_ratio:.2%}")

        if white_ratio > 0.3:
            print(f"  → 可能有菌落")
        else:
            print(f"  → 可能是空孔位或暗菌落")


def main():
    parser = argparse.ArgumentParser(description='检查特定孔位的菌落检测情况')
    parser.add_argument('--image', '-i', required=True, help='输入图像路径')
    parser.add_argument(
        '--output', '-o', default='well_check_output', help='输出目录')
    parser.add_argument('--wells', '-w', nargs='+', default=['E6', 'H12'],
                        help='要检查的孔位列表（默认: E6 H12）')

    args = parser.parse_args()

    # 创建输出目录
    Path(args.output).mkdir(exist_ok=True)

    # 检查孔位
    check_specific_wells(args.image, args.output, args.wells)

    print(f"\n💡 提示：")
    print("- 黄色圆圈：标准搜索半径")
    print("- 橙色虚线圆圈：扩展搜索半径")
    print("- 红色十字：孔位中心")


if __name__ == "__main__":
    main()
