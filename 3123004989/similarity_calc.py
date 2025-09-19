import numpy as np
from typing import List
from text_processor import get_word_frequency


def jaccard_similarity(words1: List[str], words2: List[str]) -> float:
    """Jaccard相似度（保持不变，适配修复后的词列表）"""
    set_a = set(words1)
    set_b = set(words2)

    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0

    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return round(intersection / union, 4)


def cosine_similarity(words1: List[str], words2: List[str]) -> float:
    """余弦相似度（保持不变，适配修复后的词频字典）"""
    freq1 = get_word_frequency(words1)
    freq2 = get_word_frequency(words2)

    all_words = set(freq1.keys()).union(set(freq2.keys()))
    vec1 = [freq1.get(word, 0) for word in all_words]
    vec2 = [freq2.get(word, 0) for word in all_words]

    np_vec1 = np.array(vec1)
    np_vec2 = np.array(vec2)

    dot_product = np.dot(np_vec1, np_vec2)
    norm1 = np.linalg.norm(np_vec1)
    norm2 = np.linalg.norm(np_vec2)

    if norm1 == 0 or norm2 == 0:
        return 1.0 if (norm1 == 0 and norm2 == 0) else 0.0

    return round(dot_product / (norm1 * norm2), 4)


def calculate_final_repeat_rate(words1: List[str], words2: List[str]) -> float:
    """最终重复率（取两种算法平均值，保持不变）"""
    jaccard = jaccard_similarity(words1, words2)
    cosine = cosine_similarity(words1, words2)
    return round((jaccard + cosine) / 2, 4)