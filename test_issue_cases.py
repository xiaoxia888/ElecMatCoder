"""
测试用户报告的问题案例
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.bert_ner.predictor import PipePredictor
from src.encoder.pipe_encoder import get_pipe_encoder

# 初始化
predictor = PipePredictor(model_path='models/pipe_model', device='mps')
encoder = get_pipe_encoder()

# 测试案例
test_cases = [
    {
        "name": "案例1: 按HG20202执行 被错误识别",
        "text": "WELDOLET, GB/T 19326, NB/T 47010II-S30408, FORGE, BE, OXYGEN S-40S x S-40S 与氧气接触面应平滑, 无锐边、毛刺及焊瘤, 出厂前应严格除锈, 脱脂, 按HG20202执行;DN200×80",
        "issue": "按HG20202执行 应该只识别 HG20202"
    },
    {
        "name": "案例2: NB/T 47010II 识别正确（对照组）",
        "text": "WELDOLET, GB/T 19326, NB/T 47010II-S30408, FORGE, BE S-20 x S-40S;DN500x100",
        "issue": "应该正确识别 NB/T 47010II"
    },
    {
        "name": "案例3: NB/T 47010II 的 II 丢失",
        "text": "SOCKOLET, GB/T 19326, NB/T 47010II-S30408, FORGE, SW, CL 3000;DN500x20",
        "issue": "II 等级丢失了"
    }
]

print("=" * 100)
print("测试问题案例")
print("=" * 100)

for i, case in enumerate(test_cases, 1):
    print(f"\n{'='*100}")
    print(f"测试 {i}: {case['name']}")
    print(f"问题描述: {case['issue']}")
    print(f"原文: {case['text']}")
    print(f"{'='*100}")
    
    # 1. BERT 预测
    print(f"\n【1. BERT识别结果】")
    ner_result = predictor.predict(case['text'])
    
    print("识别的实体:")
    for entity in ner_result['entities']:
        entity_type = entity['type']
        entity_text = entity['text']
        start = entity['start']
        end = entity['end']
        print(f"  {entity_type}: '{entity_text}' (位置 {start}-{end})")
    
    # 2. 编码
    print(f"\n【2. 编码结果】")
    encode_result = encoder.encode_from_tokens(ner_result['tokens'], case['text'])
    
    print(f"最终编码: {encode_result.final_code}")
    
    if 'STANDARD' in encode_result.fields:
        std_field = encode_result.fields['STANDARD']
        print(f"\n规范字段:")
        print(f"  编码: {std_field.code}")
        print(f"  显示: {std_field.display}")
        if hasattr(std_field, 'items') and std_field.items:
            print(f"  详细:")
            for item in std_field.items:
                original = item.get('original', '')
                code = item.get('code', '')
                base_code = item.get('base_code', '')
                grade = item.get('grade', '')
                print(f"    原始: '{original}' -> 编码: '{code}' (base: '{base_code}', grade: '{grade}')")

print("\n" + "=" * 100)
print("分析完成")
print("=" * 100)
