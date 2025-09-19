import argparse
import os
from typing import Tuple
from text_processor import process_txt_content
from similarity_calc import calculate_final_repeat_rate


def read_txt_file(file_path: str) -> str:

    # 验证文件格式为TXT
    if not file_path.lower().endswith(".txt"):
        raise ValueError(f"错误：仅支持TXT文件，当前文件：{file_path}")
    # 验证文件存在
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"错误：TXT文件不存在：{file_path}")
    # 验证文件可读
    if not os.access(file_path, os.R_OK):
        raise PermissionError(f"错误：无TXT文件读取权限：{file_path}")

    # 读取文件内容
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except UnicodeDecodeError:
        raise UnicodeDecodeError(f"错误：TXT文件需使用UTF-8编码：{file_path}")


def write_result_file(result_path: str, repeat_rate: float) -> None:

    if not result_path.lower().endswith(".txt"):
        raise ValueError(f"错误：输出文件需为TXT格式：{result_path}")


    formatted_rate = round(repeat_rate, 2)

    result_dir = os.path.dirname(result_path)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir, exist_ok=True)

    # 写入结果
    with open(result_path, 'w', encoding='utf-8') as f:
        f.write(f"{formatted_rate:.2f}")


def parse_command_line_args() -> Tuple[str, str, str]:
    """
    解析命令行参数（严格遵循需求：原文/抄袭版/结果TXT路径）
    :return: (orig_txt_path, copy_txt_path, result_txt_path)
    """
    parser = argparse.ArgumentParser(description="论文查重工具（仅支持TXT文件输入输出）")

    parser.add_argument("orig_txt", help="论文原文TXT文件绝对路径（示例：C:/test/orig.txt）")
    parser.add_argument("copy_txt", help="抄袭版论文TXT文件绝对路径（示例：C:/test/orig_add.txt）")
    parser.add_argument("result_txt", help="输出结果TXT文件绝对路径（示例：C:/test/result.txt）")

    args = parser.parse_args()
    return args.orig_txt, args.copy_txt, args.result_txt


def main():
    try:

        orig_path, copy_path, result_path = parse_command_line_args()


        orig_content = read_txt_file(orig_path)
        copy_content = read_txt_file(copy_path)
        orig_words = process_txt_content(orig_content)
        copy_words = process_txt_content(copy_content)


        repeat_rate = calculate_final_repeat_rate(orig_words, copy_words)


        write_result_file(result_path, repeat_rate)
        print(f"查重完成！重复率：{repeat_rate:.2f}，结果已保存至：{result_path}")


    except (ValueError, FileNotFoundError, PermissionError, UnicodeDecodeError) as e:
        print(f"执行错误：{e}")
        exit(1)
    except Exception as e:
        print(f"未知错误：{str(e)}")
        exit(1)


if __name__ == "__main__":
    main()