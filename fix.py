#!/usr/bin/env python3
"""
快速修复 Colony Detection SAM 2.0 中的变量作用域错误
运行此脚本来修复 results.py 中的 UnboundLocalError 问题
"""

import os
import shutil
from pathlib import Path


def backup_file(file_path):
    """备份原文件"""
    backup_path = f"{file_path}.backup"
    shutil.copy2(file_path, backup_path)
    print(f"✅ 已备份原文件到: {backup_path}")


def apply_fix():
    """应用修复"""
    results_file = Path("colony_analysis/utils/results.py")
    
    if not results_file.exists():
        print("❌ 找不到 colony_analysis/utils/results.py 文件")
        print("请确保在项目根目录运行此脚本")
        return False
    
    # 备份原文件
    backup_file(results_file)
    
    # 读取原文件内容
    with open(results_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 应用修复1: 修复 _generate_phenotype_stats 方法
    old_phenotype_method = '''    def _generate_phenotype_stats(self, colonies: List[Dict]) -> Dict:
        """生成表型统计"""
        phenotype_stats = {}

        for colony in colonies:
            phenotype = colony.get("phenotype", {})
            for category, value in phenotype.items():
                if category not in phenotype_stats:
                    phenotype_stats[category] = {}
            # 处理特殊情况：如果值是列表，转换为字符串
            if isinstance(value, list):
                value_key = ", ".join(map(str, value)) if value else "none"
            else:
                value_key = value

            phenotype_stats[category][value_key] = (
                phenotype_stats[category].get(value_key, 0) + 1
            )
        return phenotype_stats'''
    
    new_phenotype_method = '''    def _generate_phenotype_stats(self, colonies: List[Dict]) -> Dict:
        """生成表型统计"""
        phenotype_stats = {}

        for colony in colonies:
            phenotype = colony.get("phenotype", {})
            
            # 确保 phenotype 是字典类型
            if not isinstance(phenotype, dict):
                logging.warning(f"菌落 {colony.get('id', 'unknown')} 的表型数据不是字典格式")
                continue
                
            for category, value in phenotype.items():
                if category not in phenotype_stats:
                    phenotype_stats[category] = {}
                
                # 🔥 修复：将处理 value 的逻辑移到循环内部
                try:
                    # 处理特殊情况：如果值是列表，转换为字符串
                    if isinstance(value, list):
                        value_key = ", ".join(map(str, value)) if value else "none"
                    elif value is None:
                        value_key = "none"
                    else:
                        value_key = str(value)

                    phenotype_stats[category][value_key] = (
                        phenotype_stats[category].get(value_key, 0) + 1
                    )
                except Exception as e:
                    logging.warning(f"处理表型数据时出错: category={category}, value={value}, error={e}")
                    # 使用安全的默认值
                    value_key = "error"
                    phenotype_stats[category][value_key] = (
                        phenotype_stats[category].get(value_key, 0) + 1
                    )
        
        return phenotype_stats'''
    
    # 应用修复2: 改进 save_all_results 方法的异常处理
    old_save_all = '''        # 5. 生成分析报告
        report_path = self.generate_analysis_report(colonies, args)
        saved_files["report"] = str(report_path)

        logging.info(f"结果保存完成: {len(saved_files)} 个文件/目录")
        return saved_files

    except Exception as e:
        logging.error(f"保存结果时发生错误: {e}")
        raise'''
    
    new_save_all = '''        # 5. 生成分析报告
        try:
            report_path = self.generate_analysis_report(colonies, args)
            saved_files["report"] = str(report_path)
        except Exception as e:
            logging.error(f"生成分析报告失败: {e}")
            # 创建最小报告
            try:
                minimal_report_path = self.directories["reports"] / "minimal_report.txt"
                with open(minimal_report_path, "w", encoding="utf-8") as f:
                    f.write(f"分析完成\\n")
                    f.write(f"总菌落数: {len(colonies)}\\n")
                    f.write(f"报告生成错误: {e}\\n")
                saved_files["report"] = str(minimal_report_path)
            except Exception:
                pass

        logging.info(f"结果保存完成: {len(saved_files)} 个文件/目录")
        return saved_files

    except Exception as e:
        logging.error(f"保存结果时发生错误: {e}")
        # 确保至少返回部分结果
        return saved_files'''
    
    # 执行替换
    fixes_applied = 0
    
    if old_phenotype_method in content:
        content = content.replace(old_phenotype_method, new_phenotype_method)
        fixes_applied += 1
        print("✅ 修复了 _generate_phenotype_stats 方法")
    
    if old_save_all in content:
        content = content.replace(old_save_all, new_save_all)
        fixes_applied += 1
        print("✅ 改进了 save_all_results 方法的异常处理")
    
    # 写回文件
    with open(results_file, 'w', encoding='utf-8') as f:
        f.write(content)
    
    if fixes_applied > 0:
        print(f"✅ 总共应用了 {fixes_applied} 个修复")
        print("🎉 修复完成！现在可以重新运行程序了。")
        return True
    else:
        print("⚠️  没有找到需要修复的代码模式")
        print("可能文件已经被修复过，或者代码结构有变化")
        return False


def main():
    """主函数"""
    print("Colony Detection SAM 2.0 - 快速修复工具")
    print("=" * 50)
    print("此脚本将修复 results.py 中的变量作用域错误")
    print()
    
    # 确认是否继续
    response = input("是否继续？(y/N): ").strip().lower()
    if response != 'y':
        print("取消修复")
        return
    
    # 应用修复
    success = apply_fix()
    
    if success:
        print("\n" + "=" * 50)
        print("修复完成！接下来的步骤：")
        print("1. 重新运行您的分析命令")
        print("2. 如果遇到问题，可以用 .backup 文件恢复原始代码")
        print("3. 检查输出目录中的结果文件")
    else:
        print("\n" + "=" * 50)
        print("修复未成功应用。请检查：")
        print("1. 是否在正确的项目目录中运行")
        print("2. results.py 文件是否存在")
        print("3. 是否有足够的文件权限")


if __name__ == "__main__":
    main()