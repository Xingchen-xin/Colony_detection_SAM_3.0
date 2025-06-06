import argparse
from colony_analysis.pipeline import batch_medium_pipeline


def main():
    parser = argparse.ArgumentParser(
        description="批量处理菌落图像，根据文件名自动调用不同的分析函数"
    )
    parser.add_argument("-i", "--input", required=True, help="原始图片目录")
    parser.add_argument("-o", "--output", required=True, help="结果输出目录")
    args = parser.parse_args()

    batch_medium_pipeline(args.input, args.output)


if __name__ == "__main__":
    main()
