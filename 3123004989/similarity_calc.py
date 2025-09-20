import math
from typing import List
from collections import Counter  # 高效词频统计工具


def jaccard_similarity(words1: List[str], words2: List[str]) -> float:
    """优化后的Jaccard相似度计算（词集重叠度）"""
    set1, set2 = set(words1), set(words2)

    # 快速处理空集合场景
    if not set1:
        return 1.0 if not set2 else 0.0

    # 优化：直接计算交集/并集（避免临时变量）
    intersection = len(set1 & set2)
    union = len(set1 | set2)

    return round(intersection / union, 4) if union != 0 else 0.0


def cosine_similarity(words1: List[str], words2: List[str]) -> float:
    """优化后的余弦相似度计算（词频向量夹角）"""
    # 快速处理空列表场景
    if not words1 or not words2:
        return 1.0 if (not words1 and not words2) else 0.0

    # 优化1：用Counter替代手动词频统计（O(n)复杂度）
    cnt1, cnt2 = Counter(words1), Counter(words2)

    # 优化2：集合运算合并词汇表（比列表去重更高效）
    all_words = cnt1.keys() | cnt2.keys()

    # 优化3：单次遍历完成点积和模长计算（减少50%遍历次数）
    dot_product = 0.0
    norm1_sq, norm2_sq = 0.0, 0.0

    for word in all_words:
        w1, w2 = cnt1.get(word, 0), cnt2.get(word, 0)
        dot_product += w1 * w2
        norm1_sq += w1 ** 2
        norm2_sq += w2 ** 2

    # 优化4：用math.sqrt替代numpy（避免小数据类型转换开销）
    if norm1_sq == 0 or norm2_sq == 0:
        return 1.0 if (norm1_sq == 0 and norm2_sq == 0) else 0.0

    return round(
        dot_product / (math.sqrt(norm1_sq) * math.sqrt(norm2_sq)),
        4
    )


def calculate_final_repeat_rate(words1: List[str], words2: List[str]) -> float:
    """优化后的最终重复率计算（支持长文本分块处理）"""
    # 优化5：长文本分块计算（解决大文本内存峰值问题）
    chunk_size = 1000  # 每1000词为一块
    is_extra_long = len(words1) > 10000 or len(words2) > 10000

    if is_extra_long:
        # 分块处理长文本
        chunks1 = [words1[i:i + chunk_size] for i in range(0, len(words1), chunk_size)]
        chunks2 = [words2[i:i + chunk_size] for i in range(0, len(words2), chunk_size)]
        total_chunks = max(len(chunks1), len(chunks2))

        # 累计各块相似度（加权平均）
        total_jaccard, total_cosine = 0.0, 0.0
        for i in range(total_chunks):
            c1 = chunks1[i] if i < len(chunks1) else []
            c2 = chunks2[i] if i < len(chunks2) else []
            total_jaccard += jaccard_similarity(c1, c2)
            total_cosine += cosine_similarity(c1, c2)

        jaccard = total_jaccard / total_chunks
        cosine = total_cosine / total_chunks
    else:
        # 短文本直接计算
        jaccard = jaccard_similarity(words1, words2)
        cosine = cosine_similarity(words1, words2)

    # 保持动态加权逻辑
    is_long_text = len(words1) > 50 or len(words2) > 50
    if is_long_text:
        return round(jaccard * 0.2 + cosine * 0.8, 4)
    else:
        return round((jaccard + cosine) / 2, 4)
