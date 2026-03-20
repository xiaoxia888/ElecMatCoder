"""
BIO数据集分析脚本
"""

import os
from collections import Counter


def load_bio_file(file_path: str):
    """
    加载BIO格式文件
    
    Returns:
        sentences: 句子列表，每个句子是 [(char, tag), ...] 的列表
    """
    sentences = []
    current_sentence = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:  # 空行表示句子结束
                if current_sentence:
                    sentences.append(current_sentence)
                    current_sentence = []
            else:
                parts = line.split()
                if len(parts) >= 2:
                    char, tag = parts[0], parts[1]
                    current_sentence.append((char, tag))
                elif len(parts) == 1:
                    # 可能是只有字符没有标签的情况
                    current_sentence.append((parts[0], 'O'))
    
    # 处理最后一个句子
    if current_sentence:
        sentences.append(current_sentence)
    
    return sentences


def analyze_dataset(file_path: str):
    """分析数据集"""
    print(f"=" * 60)
    print(f"数据集分析: {file_path}")
    print(f"=" * 60)
    
    sentences = load_bio_file(file_path)
    
    # 基本统计
    total_chars = sum(len(s) for s in sentences)
    avg_len = total_chars / len(sentences) if sentences else 0
    max_len = max(len(s) for s in sentences) if sentences else 0
    min_len = min(len(s) for s in sentences) if sentences else 0
    
    print(f"\n📊 基本统计:")
    print(f"  - 样本数量: {len(sentences)}")
    print(f"  - 总字符数: {total_chars}")
    print(f"  - 平均长度: {avg_len:.1f}")
    print(f"  - 最大长度: {max_len}")
    print(f"  - 最小长度: {min_len}")
    
    # 标签统计
    tag_counter = Counter()
    entity_counter = Counter()  # 统计实体类型（去掉B-/I-前缀）
    
    for sentence in sentences:
        for char, tag in sentence:
            tag_counter[tag] += 1
            if tag != 'O':
                entity_type = tag.split('-')[1] if '-' in tag else tag
                entity_counter[entity_type] += 1
    
    print(f"\n🏷️ 标签分布:")
    for tag, count in sorted(tag_counter.items(), key=lambda x: -x[1]):
        pct = count / total_chars * 100
        print(f"  - {tag}: {count} ({pct:.1f}%)")
    
    print(f"\n📦 实体类型分布:")
    for entity, count in sorted(entity_counter.items(), key=lambda x: -x[1]):
        print(f"  - {entity}: {count}")
    
    # 获取所有唯一标签
    unique_tags = sorted(set(tag_counter.keys()))
    print(f"\n🔖 标签集合 ({len(unique_tags)}个):")
    print(f"  {unique_tags}")
    
    # 显示几个样本
    print(f"\n📝 样本示例 (前3条):")
    for i, sentence in enumerate(sentences[:3]):
        text = ''.join([c for c, t in sentence])
        print(f"\n  [{i+1}] {text}")
        
        # 提取实体
        entities = extract_entities(sentence)
        if entities:
            print(f"      实体: {entities}")
    
    return {
        'num_samples': len(sentences),
        'total_chars': total_chars,
        'avg_len': avg_len,
        'max_len': max_len,
        'unique_tags': unique_tags,
        'tag_counter': tag_counter,
        'entity_counter': entity_counter,
        'sentences': sentences
    }


def extract_entities(sentence):
    """从BIO标注中提取实体"""
    entities = []
    current_entity = []
    current_type = None
    
    for char, tag in sentence:
        if tag.startswith('B-'):
            # 保存上一个实体
            if current_entity:
                entities.append((current_type, ''.join(current_entity)))
            # 开始新实体
            current_type = tag[2:]
            current_entity = [char]
        elif tag.startswith('I-') and current_type == tag[2:]:
            # 继续当前实体
            current_entity.append(char)
        else:
            # O标签或不匹配的I标签
            if current_entity:
                entities.append((current_type, ''.join(current_entity)))
                current_entity = []
                current_type = None
    
    # 处理最后一个实体
    if current_entity:
        entities.append((current_type, ''.join(current_entity)))
    
    return entities


if __name__ == '__main__':
    import sys
    
    # 默认数据文件路径
    default_path = os.path.join(
        os.path.dirname(__file__), 
        '../../data/annotations_char.bio'
    )
    
    file_path = sys.argv[1] if len(sys.argv) > 1 else default_path
    
    if os.path.exists(file_path):
        analyze_dataset(file_path)
    else:
        print(f"文件不存在: {file_path}")

