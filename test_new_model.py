"""
测试新训练的模型（小写版本）
"""
import sys
from pathlib import Path

# 添加 src 到 path
sys.path.insert(0, str(Path(__file__).parent))

from src.bert_ner.predictor import PipePredictor

# 初始化预测器
predictor = PipePredictor(
    model_path='models/pipe_model',
    device='mps'
)

# 测试案例
test_cases = [
    {
        "name": "OXYGEN误识别问题（大写）",
        "text": "无缝钢管 PIPE SMLS PE HG/T20553-11 Ia SERIES Sch.80 DN25 CS20,GB/T3087 -",
        "expected": {
            "STANDARD_GRADE": ["Ia", "SERIES"],  # OXYGEN 不应该被识别为等级
            "NOT_STANDARD_GRADE": ["OXYGEN"]
        }
    },
    {
        "name": "oxygen小写识别",
        "text": "无缝钢管 PIPE SMLS PE HG/T20553-11 Ia series sch.80 dn25 cs20,gb/t3087 oxygen free",
        "expected": {
            "STANDARD_GRADE": ["Ia", "series"],
            "NOT_STANDARD_GRADE": ["oxygen"]
        }
    },
    {
        "name": "规范等级正常识别（大写）",
        "text": "偏心异径管, GB/T 13401-SF304, BE, GB/T 12459, Series I, WELDED",
        "expected": {
            "STANDARD_GRADE": ["Series I"],
            "STANDARD": ["GB/T 13401", "GB/T 12459"]
        }
    },
    {
        "name": "规范等级正常识别（小写）",
        "text": "偏心异径管, gb/t 13401-sf304, be, gb/t 12459, series i, welded",
        "expected": {
            "STANDARD_GRADE": ["series i"],
            "STANDARD": ["gb/t 13401", "gb/t 12459"]
        }
    },
    {
        "name": "多种格式的等级",
        "text": "管件 GB/T 12459 TYPE I, SH/T 3405 serial II, HG/T 20553 Ia",
        "expected": {
            "STANDARD_GRADE": ["TYPE I", "serial II", "Ia"]
        }
    }
]

print("=" * 80)
print("测试新模型（小写训练版本）")
print("=" * 80)

for i, case in enumerate(test_cases, 1):
    print(f"\n测试 {i}: {case['name']}")
    print(f"文本: {case['text']}")
    
    # 预测
    result = predictor.predict(case['text'])
    
    # 提取识别的实体
    recognized = {}
    for entity in result['entities']:
        entity_type = entity['type']
        entity_text = entity['text']
        if entity_type not in recognized:
            recognized[entity_type] = []
        recognized[entity_type].append(entity_text)
    
    print(f"\n识别结果:")
    for entity_type, entities in recognized.items():
        print(f"  {entity_type}: {entities}")
    
    # 验证期望结果
    print(f"\n验证:")
    all_pass = True
    
    for exp_type, exp_values in case['expected'].items():
        if exp_type.startswith("NOT_"):
            # 负向验证：不应该被识别为某类型
            actual_type = exp_type[4:]  # 去掉 "NOT_" 前缀
            actual_entities = recognized.get(actual_type, [])
            for val in exp_values:
                if any(val.lower() in e.lower() for e in actual_entities):
                    print(f"  ❌ '{val}' 被错误识别为 {actual_type}")
                    all_pass = False
                else:
                    print(f"  ✓ '{val}' 未被识别为 {actual_type} (正确)")
        else:
            # 正向验证：应该被识别为某类型
            actual_entities = recognized.get(exp_type, [])
            for val in exp_values:
                if any(val.lower() in e.lower() for e in actual_entities):
                    print(f"  ✓ '{val}' 被正确识别为 {exp_type}")
                else:
                    print(f"  ❌ '{val}' 未被识别为 {exp_type}")
                    all_pass = False
    
    if all_pass:
        print(f"\n✅ 测试通过")
    else:
        print(f"\n❌ 测试失败")

print("\n" + "=" * 80)
print("测试完成")
print("=" * 80)
