# debug_detection.py - 菌落检测调试工具

import cv2
import numpy as np
import matplotlib.pyplot as plt
import logging
from pathlib import Path
import argparse

from colony_analysis.core import SAMModel, ColonyDetector
from colony_analysis.config import ConfigManager


def debug_sam_detection(image_path, output_dir='debug_output'):
    """调试SAM检测过程的每个步骤"""

    # 设置日志
    logging.basicConfig(level=logging.DEBUG)

    # 创建输出目录
    Path(output_dir).mkdir(exist_ok=True)

    # 加载图像
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"无法读取图像: {image_path}")

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    print(f"✅ 图像加载成功: {img_rgb.shape}")

    # 初始化配置和模型
    config = ConfigManager()
    sam_model = SAMModel(model_type='vit_b', config=config)
    detector = ColonyDetector(sam_model=sam_model, config=config)

    print("✅ 模型初始化完成")

    # 步骤1: 测试SAM原始输出
    print("\n🔍 步骤1: 测试SAM原始输出...")
    masks_data = sam_model.mask_generator.generate(img_rgb)
    print(f"SAM生成了 {len(masks_data)} 个原始掩码")

    # 分析原始掩码
    areas = [m['area'] for m in masks_data]
    scores = [m['stability_score'] for m in masks_data]

    print(f"面积范围: {min(areas):.0f} - {max(areas):.0f}")
    print(f"分数范围: {min(scores):.3f} - {max(scores):.3f}")

    # 步骤2: 测试面积过滤
    print(f"\n🔍 步骤2: 测试面积过滤...")
    min_area = config.get('detection', 'min_colony_area', 1000) // 8
    print(f"最小面积阈值: {min_area}")

    filtered_masks = []
    for mask_data in masks_data:
        if mask_data['area'] >= min_area:
            filtered_masks.append(mask_data)

    print(f"面积过滤后: {len(filtered_masks)} 个掩码")

    # 步骤3: 测试稳定性过滤
    print(f"\n🔍 步骤3: 测试稳定性过滤...")
    stability_thresh = config.get('sam', 'stability_score_thresh', 0.65)
    print(f"稳定性阈值: {stability_thresh}")

    stable_masks = []
    for mask_data in filtered_masks:
        if mask_data['stability_score'] >= stability_thresh:
            stable_masks.append(mask_data)

    print(f"稳定性过滤后: {len(stable_masks)} 个掩码")

    # 步骤4: 完整检测流程
    print(f"\n🔍 步骤4: 完整检测流程...")
    colonies = detector.detect(img_rgb)
    print(f"最终检测到: {len(colonies)} 个菌落")

    # 可视化结果
    create_debug_visualization(
        img_rgb, masks_data, stable_masks, colonies, output_dir)

    # 保存详细信息
    save_debug_info(masks_data, stable_masks, colonies, output_dir)

    return colonies


def create_debug_visualization(img_rgb, raw_masks, filtered_masks, colonies, output_dir):
    """创建调试可视化"""

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # 原始图像
    axes[0, 0].imshow(img_rgb)
    axes[0, 0].set_title('原始图像')
    axes[0, 0].axis('off')

    # 所有原始掩码
    overlay1 = img_rgb.copy()
    colors = plt.cm.tab20(np.linspace(0, 1, min(20, len(raw_masks))))

    for i, mask_data in enumerate(raw_masks[:20]):  # 只显示前20个
        mask = mask_data['segmentation']
        color = (colors[i % len(colors)][:3] * 255).astype(np.uint8)
        overlay1[mask] = overlay1[mask] * 0.7 + color * 0.3

    axes[0, 1].imshow(overlay1)
    axes[0, 1].set_title(f'所有SAM掩码 (显示前20个，共{len(raw_masks)}个)')
    axes[0, 1].axis('off')

    # 过滤后的掩码
    overlay2 = img_rgb.copy()
    for i, mask_data in enumerate(filtered_masks):
        mask = mask_data['segmentation']
        color = (colors[i % len(colors)][:3] * 255).astype(np.uint8)
        overlay2[mask] = overlay2[mask] * 0.7 + color * 0.3

    axes[1, 0].imshow(overlay2)
    axes[1, 0].set_title(f'过滤后的掩码 ({len(filtered_masks)}个)')
    axes[1, 0].axis('off')

    # 最终检测结果
    overlay3 = img_rgb.copy()
    for i, colony in enumerate(colonies):
        minr, minc, maxr, maxc = colony['bbox']
        cv2.rectangle(overlay3, (minc, minr), (maxc, maxr), (0, 255, 0), 3)
        cv2.putText(overlay3, f"C{i+1}", (minc, minr-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    axes[1, 1].imshow(overlay3)
    axes[1, 1].set_title(f'最终检测结果 ({len(colonies)}个)')
    axes[1, 1].axis('off')

    plt.tight_layout()
    plt.savefig(f"{output_dir}/debug_visualization.png",
                dpi=150, bbox_inches='tight')
    plt.close()

    print(f"✅ 可视化保存到: {output_dir}/debug_visualization.png")


def save_debug_info(raw_masks, filtered_masks, colonies, output_dir):
    """保存详细的调试信息"""

    info_lines = []
    info_lines.append("=== 菌落检测调试报告 ===\n")

    info_lines.append(f"原始SAM掩码数量: {len(raw_masks)}")
    info_lines.append(f"过滤后掩码数量: {len(filtered_masks)}")
    info_lines.append(f"最终菌落数量: {len(colonies)}\n")

    # 原始掩码统计
    if raw_masks:
        areas = [m['area'] for m in raw_masks]
        scores = [m['stability_score'] for m in raw_masks]

        info_lines.append("原始掩码统计:")
        info_lines.append(
            f"  面积: 最小={min(areas):.0f}, 最大={max(areas):.0f}, 平均={np.mean(areas):.0f}")
        info_lines.append(
            f"  分数: 最小={min(scores):.3f}, 最大={max(scores):.3f}, 平均={np.mean(scores):.3f}\n")

    # 过滤掩码统计
    if filtered_masks:
        areas = [m['area'] for m in filtered_masks]
        scores = [m['stability_score'] for m in filtered_masks]

        info_lines.append("过滤后掩码统计:")
        info_lines.append(
            f"  面积: 最小={min(areas):.0f}, 最大={max(areas):.0f}, 平均={np.mean(areas):.0f}")
        info_lines.append(
            f"  分数: 最小={min(scores):.3f}, 最大={max(scores):.3f}, 平均={np.mean(scores):.3f}\n")

    # 菌落统计
    if colonies:
        areas = [c['area'] for c in colonies]

        info_lines.append("最终菌落统计:")
        info_lines.append(
            f"  面积: 最小={min(areas):.0f}, 最大={max(areas):.0f}, 平均={np.mean(areas):.0f}\n")

        info_lines.append("菌落详细信息:")
        for i, colony in enumerate(colonies):
            info_lines.append(f"  菌落{i+1}: ID={colony['id']}, 面积={colony['area']:.0f}, "
                              f"中心={colony['centroid']}, 边界框={colony['bbox']}")

    # 保存到文件
    with open(f"{output_dir}/debug_info.txt", 'w', encoding='utf-8') as f:
        f.write('\n'.join(info_lines))

    print(f"✅ 调试信息保存到: {output_dir}/debug_info.txt")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='菌落检测调试工具')
    parser.add_argument('--image', '-i', required=True, help='输入图像路径')
    parser.add_argument('--output', '-o', default='debug_output', help='输出目录')

    args = parser.parse_args()

    try:
        colonies = debug_sam_detection(args.image, args.output)
        print(f"\n🎉 调试完成！检测到 {len(colonies)} 个菌落")
        print(f"📁 调试文件保存在: {args.output}")

    except Exception as e:
        print(f"❌ 调试过程中出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
