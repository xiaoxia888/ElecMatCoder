import pandas as pd
import re
from collections import Counter

df = pd.read_excel('/Users/guoxi/Desktop/数据集.xlsx')
ss = df[df['材质分类'] == '不锈钢'].copy()

# ====== 安全模式 ======
safe_patterns = [
    r'X\d+Cr[A-Za-z]+\d[\d\-]+(?:\([A-Z0-9\.]+\))?',
    r'(?<!\d)0?\d{1,3}Cr\d{1,2}Ni\d{1,2}(?:Mo\d{0,2})?(?:Ti|Nb|N(?!B|P)|W|Cu)?',
    r'S[23]\d{4}(?!\d)',
    r'SF\d{3,4}[LH]?(?!\d)',
    r'GRADE\s+(?:WP|F|TP|CC)\d{2,4}[LH]?(?:\s*\(UNS\s*S\d{5}\))?',
    r'(?:GR\.|Gr\.)\s*(?:WP|F|TP|CC)\d{2,4}[LH]?(?!\d)',
    r'\bCF\d{3}(?!\d)',
    r'\bLF\d\b',
    r'(?<=A\d{3})F\d{2,4}[LH]?(?!\d)',
    r'(?<=A\d{3}GRADE)F\d{2,4}[LH]?(?!\d)',
    r'(?<=A\d{3}GR\.)F\d{2,4}[LH]?(?!\d)',
    r'(?<=A\d{3}Gr\.)F\d{2,4}[LH]?(?!\d)',
    # Gr./GR.+裸数字牌号
    r'(?:GR\.|Gr\.)\s*(?:2507|2205|321[Hh]?|347[Hh]?|310[SMsm]?|304[LHlh]?|316[LHlh]?)(?!\d)',
    # AISI前缀
    r'(?<=AISI)\d{3}[LH]?(?!\d)',
]

# ====== 需数字前缀过滤 ======
filtered_patterns = [
    r'\bF\d{2,4}[LH]?\b',
    r'\bWP\d{3,4}[LH]?(?:-[A-Z])?\b',
    r'\bTP\d{3,4}[LH]?\b',
    r'\b(?:2507|2205|321[Hh]?|347[Hh]?|310[SMsm]?|304[LHlh]?|316[LHlh]?)\b',
]

# ====== 分组提取模式 ======
group_patterns = [
    # 标准号拼接中国元素牌号（用枚举碳含量避免边界错位）
    (r'(?:NB|GB|HG|SH|DL|JB)/T\s?\d{4,5}((?:022|06|08|0)Cr\d{1,2}Ni\d{1,2}(?:Mo\d{0,2})?(?:Ti|Nb|N(?!B|P)|W|Cu)?)', 1),
    # GRADE后裸数字牌号
    (r'GRADE\s*((?:2507|2205|321[Hh]?|347[Hh]?|310[SMsm]?|304[LHlh]?|316[LHlh]?))(?!\d)', 1),
    # A+数字后GRADE+裸数字
    (r'(?<=A\d{3}GRADE)((?:2507|2205|321[Hh]?|347[Hh]?|310[SMsm]?|304[LHlh]?|316[LHlh]?))(?!\d)', 1),
    # A+数字后Gr.+裸数字
    (r'(?<=A\d{3}Gr\.)((?:2507|2205|321[Hh]?|347[Hh]?|310[SMsm]?|304[LHlh]?|316[LHlh]?))(?!\d)', 1),
]

safe_compiled = [re.compile(p) for p in safe_patterns]
filtered_compiled = [re.compile(p) for p in filtered_patterns]
group_compiled = [(re.compile(p), g) for p, g in group_patterns]

def extract_grades(text):
    if not isinstance(text, str):
        return []
    found = []

    for pat in safe_compiled:
        for m in pat.finditer(text):
            grade = m.group().strip()
            start = m.start()
            prefix = text[max(0, start-3):start]
            if re.search(r'(DN|Φ|φ|NPS)', prefix, re.IGNORECASE):
                continue
            found.append(grade)

    for pat in filtered_compiled:
        for m in pat.finditer(text):
            grade = m.group().strip()
            start = m.start()
            prefix = text[max(0, start-3):start]
            if re.search(r'(DN|Φ|φ|NPS)', prefix, re.IGNORECASE):
                continue
            if start > 0 and text[start-1].isdigit() and text[start-1] != '0':
                continue
            found.append(grade)

    for pat, grp in group_compiled:
        for m in pat.finditer(text):
            grade = m.group(grp).strip()
            found.append(grade)

    normalized = []
    for g in found:
        if len(g) > 1 and g[-1] in 'lhsm':
            g = g[:-1] + g[-1].upper()
        normalized.append(g)
    return list(dict.fromkeys(normalized))

grade_counter = Counter()
no_grade = []

for idx, row in ss.iterrows():
    desc = str(row['材料描述'])
    grades = extract_grades(desc)
    if grades:
        for g in grades:
            grade_counter[g] += 1
    else:
        no_grade.append((desc[:200], str(row['材质代码'])))

print(f'不锈钢总行数: {len(ss)}')
print(f'成功提取: {len(ss) - len(no_grade)} 条 ({(len(ss)-len(no_grade))/len(ss)*100:.1f}%)')
print(f'未提取到: {len(no_grade)} 条')

print(f'\n{"="*70}')
print(f'不锈钢材质牌号分布（共 {len(grade_counter)} 种）')
print(f'{"="*70}')
for g, cnt in grade_counter.most_common():
    print(f'  {g:50s}  {cnt:5d}')

import json
with open('apps/trainer/qwen3_fte/output/parser_train_new_schema.json') as f:
    tdata = json.load(f)
train_grades = Counter()
for rec in tdata:
    out = rec['output'] if isinstance(rec['output'], dict) else json.loads(rec['output'])
    for item in out.get('MATERIAL', {}).get('ITEMS', []):
        if isinstance(item, dict):
            g = item.get('GRADE', '')
            if g: train_grades[g] += 1

print(f'\n{"="*70}')
print(f'Excel有 但 训练集缺失的牌号')
print(f'{"="*70}')
missing = []
for g, cnt in sorted(grade_counter.items(), key=lambda x: -x[1]):
    if g not in train_grades:
        missing.append((g, cnt))
total_miss = 0
for g, cnt in missing:
    print(f'  {g:50s}  {cnt:5d}')
    total_miss += cnt
print(f'\n共 {len(missing)} 种缺失, 覆盖 {total_miss} 条Excel记录')

print(f'\n未提取到的材质代码分布:')
no_codes = Counter(c for _, c in no_grade)
for k, cnt in no_codes.most_common():
    print(f'  {k:20s}  {cnt}')

print(f'\n未提取到的抽样:')
seen = {}
for desc, code in no_grade:
    if code not in seen: seen[code] = 0
    if seen[code] < 2:
        print(f'  [{code}] {desc[:160]}')
        seen[code] += 1