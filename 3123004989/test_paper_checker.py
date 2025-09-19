import unittest
import tempfile
import os
from text_processor import process_txt_content
from similarity_calc import jaccard_similarity, cosine_similarity, calculate_final_repeat_rate
from main import read_txt_file, write_result_file

class TestPaperChecker(unittest.TestCase):
    def test_text_preprocessing(self):
        txt_content = "今天是星期天，天气晴！今天晚上我要去看电影。"
        expected = ["星期天", "天气", "晴", "看", "电影"]
        self.assertEqual(process_txt_content(txt_content), expected)

    def test_jaccard_identical(self):
        words1 = ["星期天", "晴", "看电影"]
        words2 = ["星期天", "晴", "看电影"]
        self.assertEqual(jaccard_similarity(words1, words2), 1.0)

    def test_jaccard_different(self):
        words1 = ["星期天", "晴", "看电影"]
        words2 = ["星期一", "雨", "看书"]
        self.assertEqual(jaccard_similarity(words1, words2), 0.0)

    def test_cosine_partial(self):
        words1 = ["星期天", "晴", "看电影"]
        words2 = ["周天", "晴朗", "看电影"]
        self.assertGreater(cosine_similarity(words1, words2), 0.2)

    def test_final_repeat_rate(self):
        words1 = ["星期天", "晴", "看电影"]
        words2 = ["星期天", "晴", "看电影"]
        self.assertEqual(calculate_final_repeat_rate(words1, words2), 1.0)

    def test_empty_both_txt(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f1, \
             tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f2, \
             tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as res:
            cmd = f'python "{os.path.abspath("main.py")}" "{f1.name}" "{f2.name}" "{res.name}"'
            os.system(cmd)
            with open(res.name, 'r') as f:
                self.assertEqual(f.read().strip(), "1.00")

    def test_empty_one_txt(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f1, \
             tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f2, \
             tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as res:
            f2.write("今天是星期天，天气晴。")
            f2.flush()
            f2.close()
            cmd = f'python "{os.path.abspath("main.py")}" "{f1.name}" "{f2.name}" "{res.name}"'
            os.system(cmd)
            with open(res.name, 'r') as f:
                self.assertEqual(f.read().strip(), "0.00")

    def test_non_txt_input(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.docx', delete=False) as f:
            with self.assertRaises(ValueError):
                read_txt_file(f.name)

    def test_txt_wrong_encoding(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', encoding='gbk', delete=False) as f:
            f.write("今天天气晴。")
            f.flush()
            f.close()
            with self.assertRaises(UnicodeDecodeError) as cm:
                read_txt_file(f.name)
            self.assertIn("编码错误", str(cm.exception))

    def test_long_txt_similarity(self):
        txt1 = "Python是解释型、面向对象的高级程序设计语言。1989年由Guido van Rossum发明，1991年发布首个版本，支持跨平台开发，语法简洁易读，适合快速开发。"
        txt2 = "Python是面向对象的解释型高级程序设计语言，由Guido van Rossum于1989年创造，1991年推出第一个公开发行版，可跨平台运行，语法简洁易懂，适用于快速开发项目。"
        words1 = process_txt_content(txt1, is_long_text=True)
        words2 = process_txt_content(txt2, is_long_text=True)
        rate = calculate_final_repeat_rate(words1, words2)
        self.assertGreaterEqual(rate, 0.85, f"长文本相似度{rate}低于预期0.85")

    def test_result_format(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as res:
            write_result_file(res.name, 0.856)
            with open(res.name, 'r') as f:
                self.assertEqual(f.read(), "0.86")

if __name__ == "__main__":
    unittest.main()