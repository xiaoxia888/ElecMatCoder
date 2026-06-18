#!/usr/bin/env python3
from __future__ import annotations

import json
from collections import Counter, OrderedDict
from copy import deepcopy
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[4]
BASE = PROJECT_ROOT / 'apps/trainer/qwen3_fte/output/pipe_project_sampling_full'
INPUT_DATASET = BASE / '直管训练草稿_语义补强版.json'
AUG_PATH = BASE / '直管种类对抗增强草稿_带标识.json'
MERGED_META_PATH = BASE / '直管训练草稿_语义补强版_种类增强合并_带标识.json'
MERGED_TRAIN_PATH = BASE / '直管训练草稿_语义补强版_种类增强合并.json'
SUMMARY_PATH = BASE / '直管种类对抗增强统计.json'


def make_output(*, body: str, manu: list[str] | None = None, conn: list[str] | None = None,
                dn: str = 'DN100', thickness_kind: str = 'series', thickness_value: str = 'STD',
                material: str = '20', standards: list[tuple[str, str]] | None = None,
                coating_inner: list[str] | None = None, coating_outer: list[str] | None = None) -> OrderedDict[str, Any]:
    out: OrderedDict[str, Any] = OrderedDict()
    out['TYPE'] = {'BODY': body, 'MANU': manu or [], 'CONN': conn or []}
    out['SIZE'] = {'DN': [dn], 'OD': [], 'INCH': [], 'LENGTH': []}
    thk = {'MM': [], 'SCHEDULE': [], 'SERIES': [], 'BWG': [], 'INCH': []}
    if thickness_kind == 'series':
        thk['SERIES'] = [thickness_value]
    elif thickness_kind == 'schedule':
        thk['SCHEDULE'] = [thickness_value]
    elif thickness_kind == 'mm':
        thk['MM'] = [thickness_value]
    out['THICKNESS'] = thk
    out['MATERIAL'] = [{
        'ROLE': 'MAIN',
        'VALUE': material,
        'COATING': {'INNER': coating_inner or [], 'OUTER': coating_outer or []},
        'SPECIAL_REQ': [],
    }]
    out['STANDARD'] = [
        {'BODY': body_, 'GRADE': grade, 'METHOD': '', 'APPENDIX': ''}
        for body_, grade in (standards or [('GB/T8163', ''), ('HG/T20553', 'Ia')])
    ]
    return out


def rec(input_text: str, output: dict[str, Any], reason: str, group: str, family: str) -> dict[str, Any]:
    return {
        'input': input_text,
        'output': output,
        '_source': 'type_adversarial_augmentation',
        '_aug_tag': 'AUG_TYPE',
        '_reason': reason,
        '_group': group,
        '_family': family,
    }


def body_records() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    variants = ['DN25', 'DN50', 'DN80', 'DN100', 'DN150', 'DN200']
    families = [
        (
            'BODY_PIPE_FLANGED_LINED_CS',
            ('PIPE 20 GB/T8163 SMLS BE HG/T20553(Ia) {dn} STD', lambda dn: make_output(body='直管', manu=['SMLS'], dn=dn, material='20')),
            ('FLANGED PIPE FP, CL150, RF, 20, HG/T20538, SH/T3406;{dn}', lambda dn: make_output(body='法兰管', dn=dn, material='20', standards=[('HG/T20538', ''), ('SH/T3406', '')])),
            ('钢衬四氟管道 20/PTFE {dn} HG/T20538', lambda dn: make_output(body='衬里复合管', dn=dn, material='20/PTFE', standards=[('HG/T20538', '')])),
        ),
        (
            'BODY_PIPE_FLANGED_LINED_SS',
            ('PIPE S30408 GB/T14976 SMLS BE HG/T20553(Ia) {dn} SCH40S', lambda dn: make_output(body='直管', manu=['SMLS'], dn=dn, thickness_kind='schedule', thickness_value='SCH40S', material='S30408', standards=[('GB/T14976', ''), ('HG/T20553', 'Ia')])),
            ('FLANGED PIPE FP, CL150, RF, PTFE/304, HG/T20538, SH/T3406;{dn}', lambda dn: make_output(body='法兰管', dn=dn, material='304/PTFE', standards=[('HG/T20538', ''), ('SH/T3406', '')])),
            ('不锈钢衬四氟管道 304/PTFE {dn} HG/T20538', lambda dn: make_output(body='衬里复合管', dn=dn, material='304/PTFE', standards=[('HG/T20538', '')])),
        ),
        (
            'BODY_PRIORITY_FLANGED_OVER_LINED',
            ('PIPE API 5L Gr.B SAWL BE ASME B36.10 {dn} STD', lambda dn: make_output(body='直管', manu=['SAWL'], dn=dn, material='API 5L Gr.B', standards=[('ASME B36.10', '')])),
            ('FLANGED PIPE, CL150, RF, PTFE/20, HG/T20538, SH/T3406;{dn}', lambda dn: make_output(body='法兰管', dn=dn, material='20/PTFE', standards=[('HG/T20538', ''), ('SH/T3406', '')])),
            ('搪玻璃管 {dn} HG/T2130 PN10 无法兰', lambda dn: make_output(body='衬里复合管', dn=dn, material='搪玻璃/GLASS LINED', standards=[('HG/T2130', '')])),
        ),
        (
            'BODY_COATING_PIPE',
            ('PIPE Q235B GB/T3091 ERW BE THK=4.0mm {dn}', lambda dn: make_output(body='直管', manu=['ERW'], dn=dn, thickness_kind='mm', thickness_value='4.0', material='Q235B', standards=[('GB/T3091', '')])),
            ('两端活套法兰管 {dn} Q235B/PE HG/T20538 SH/T3406', lambda dn: make_output(body='法兰管', dn=dn, material='Q235B/PE', standards=[('HG/T20538', ''), ('SH/T3406', '')])),
            ('涂塑复合钢管{dn} PN10bar Q235B外加强级PE内EP CJ/T120', lambda dn: make_output(body='衬里复合管', dn=dn, material='Q235B', standards=[('CJ/T120', '')], coating_inner=['EP'], coating_outer=['加强级PE'])),
        ),
    ]
    for family, direct, flanged, lined in families:
        for dn in variants:
            for label, item in (('direct', direct), ('flanged', flanged), ('lined', lined)):
                desc_tpl, builder = item
                rows.append(rec(desc_tpl.format(dn=dn), builder(dn), 'BODY对抗', 'BODY', f'{family}_{label}'))
    return rows


def manu_records() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    cases = [
        ('SMLS', 'PIPE 20 GB/T8163 SMLS BE HG/T20553(Ia) {dn} STD', lambda dn: make_output(body='直管', manu=['SMLS'], dn=dn, material='20')),
        ('WELDED', '焊接钢管 Q235B GB/T3091 {dn} THK=4.0mm', lambda dn: make_output(body='直管', manu=['WELDED'], dn=dn, thickness_kind='mm', thickness_value='4.0', material='Q235B', standards=[('GB/T3091', '')])),
        ('SAW', '埋弧焊钢管 L245 GB/T9711 {dn} S-STD', lambda dn: make_output(body='直管', manu=['SAW'], dn=dn, material='L245', standards=[('GB/T9711', '')])),
        ('SAWL', '直缝埋弧焊接钢管 L245 GB/T9711 SAWL {dn} SCH20', lambda dn: make_output(body='直管', manu=['SAWL'], dn=dn, thickness_kind='schedule', thickness_value='SCH20', material='L245', standards=[('GB/T9711', '')])),
        ('SAWH', '螺旋缝埋弧焊接钢管 Q235B SY/T5037 SAWH {dn} Thk6.0', lambda dn: make_output(body='直管', manu=['SAWH'], dn=dn, thickness_kind='mm', thickness_value='6.0', material='Q235B', standards=[('SY/T5037', '')])),
        ('DSAW', 'PIPE API 5L Gr.B DSAW BE ASME B36.10 STD;{dn}', lambda dn: make_output(body='直管', manu=['DSAW'], dn=dn, material='API 5L Gr.B', standards=[('ASME B36.10', '')])),
        ('DSAWL', 'PIPE(WS) API 5L Gr.B LONGITUDE DOUBLE SUBMERGED-ARC WELDED BE ASME B36.10 STD;{dn}', lambda dn: make_output(body='直管', manu=['DSAWL'], dn=dn, material='API 5L Gr.B', standards=[('ASME B36.10', '')])),
        ('DSAWH', 'PIPE API 5L Gr.B SPIRAL DOUBLE SUBMERGED-ARC WELDED BE ASME B36.10 STD;{dn}', lambda dn: make_output(body='直管', manu=['DSAWH'], dn=dn, material='API 5L Gr.B', standards=[('ASME B36.10', '')])),
        ('ERW', '管子(WS),Q235B GB/T3091 ERW HG/T20553(Ia) T=3.8mm {dn}', lambda dn: make_output(body='直管', manu=['ERW'], dn=dn, thickness_kind='mm', thickness_value='3.8', material='Q235B', standards=[('GB/T3091', ''), ('HG/T20553', 'Ia')])),
        ('HFW', '高频焊管 Q235B HFW GB/T3091 {dn} THK=4.0mm', lambda dn: make_output(body='直管', manu=['HFW'], dn=dn, thickness_kind='mm', thickness_value='4.0', material='Q235B', standards=[('GB/T3091', '')])),
        ('EFW', '焊接钢管 PIPE EFW,BE,HG/T20553(Ia)THK=8.0mm {dn} S30408,GB/T12771,II类 -', lambda dn: make_output(body='直管', manu=['EFW'], dn=dn, thickness_kind='mm', thickness_value='8.0', material='S30408', standards=[('HG/T20553', 'Ia'), ('GB/T12771', 'II')])),
    ]
    repeat = {'SMLS': 4, 'WELDED': 4, 'SAW': 8, 'SAWL': 8, 'SAWH': 8, 'DSAW': 10, 'DSAWL': 10, 'DSAWH': 12, 'ERW': 8, 'HFW': 12, 'EFW': 6}
    dn_variants = ['DN25', 'DN40', 'DN50', 'DN80', 'DN100', 'DN150', 'DN200', 'DN300', 'DN400', 'DN500', 'DN600', 'DN800']
    for code, desc_tpl, builder in cases:
        for idx in range(repeat[code]):
            dn = dn_variants[idx % len(dn_variants)]
            rows.append(rec(desc_tpl.format(dn=dn), builder(dn), 'MANU稀有归一', 'MANU', f'MANU_{code}'))
    return rows


def conn_records() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    cases = [
        ('MNPT', '管子 ASTM A106 Gr.B SMLS TBE(MNPT) ASME B36.10 SCH80 {dn}', lambda dn: make_output(body='直管', manu=['SMLS'], conn=['MNPT'], dn=dn, thickness_kind='schedule', thickness_value='SCH80', material='ASTM A106 Gr.B', standards=[('ASME B36.10', '')])),
        ('FNPT', '管子 20 GB/T8163 FNPT End HG/T20553(Ia) {dn} SCH80', lambda dn: make_output(body='直管', conn=['FNPT'], dn=dn, thickness_kind='schedule', thickness_value='SCH80', material='20', standards=[('GB/T8163', ''), ('HG/T20553', 'Ia')])),
        ('NPT', 'PIPE(SMLS) ASTM A53 Gr.B GALVANIZED SMLS NPT ASME B36.10 SCH80;{dn}', lambda dn: make_output(body='直管', manu=['SMLS'], conn=['NPT'], dn=dn, thickness_kind='schedule', thickness_value='SCH80', material='ASTM A53 Gr.B', standards=[('ASME B36.10', '')])),
        ('THD', '管子 20 GB/T8163 THD HG/T20553(Ia) {dn} SCH40', lambda dn: make_output(body='直管', conn=['THD'], dn=dn, thickness_kind='schedule', thickness_value='SCH40', material='20', standards=[('GB/T8163', ''), ('HG/T20553', 'Ia')])),
        ('SW', '管子 20 GB/T8163 SW HG/T20553(Ia) {dn} STD', lambda dn: make_output(body='直管', conn=['SW'], dn=dn, material='20', standards=[('GB/T8163', ''), ('HG/T20553', 'Ia')])),
        ('FTE', '管子 20 GB/T8163 FTE HG/T20553(Ia) {dn} SCH80', lambda dn: make_output(body='直管', conn=['FTE'], dn=dn, thickness_kind='schedule', thickness_value='SCH80', material='20', standards=[('GB/T8163', ''), ('HG/T20553', 'Ia')])),
        ('MTE', '管子 20 GB/T8163 MTE HG/T20553(Ia) {dn} SCH80', lambda dn: make_output(body='直管', conn=['MTE'], dn=dn, thickness_kind='schedule', thickness_value='SCH80', material='20', standards=[('GB/T8163', ''), ('HG/T20553', 'Ia')])),
        ('SF', 'FRP/CPVC管，FRP/PVC,SF, HG/T3731，THK=4.0mm {dn}', lambda dn: make_output(body='衬里复合管', conn=['SF'], dn=dn, thickness_kind='mm', thickness_value='4.0', material='FRP/PVC', standards=[('HG/T3731', '')])),
    ]
    repeat = {'MNPT': 14, 'FNPT': 14, 'NPT': 10, 'THD': 10, 'SW': 10, 'FTE': 14, 'MTE': 14, 'SF': 10}
    dn_variants = ['DN15', 'DN20', 'DN25', 'DN32', 'DN40', 'DN50', 'DN65', 'DN80', 'DN100', 'DN125', 'DN150', 'DN200', 'DN250', 'DN300']
    for code, desc_tpl, builder in cases:
        for idx in range(repeat[code]):
            dn = dn_variants[idx % len(dn_variants)]
            rows.append(rec(desc_tpl.format(dn=dn), builder(dn), 'CONN稀有归一', 'CONN', f'CONN_{code}'))
    return rows


def negative_records() -> list[dict[str, Any]]:
    cases = [
        ('PIPE API 5L Gr.B SAWL BE ASME B36.10 300LB DN200 S-STD', make_output(body='直管', manu=['SAWL'], dn='DN200', material='API 5L Gr.B', standards=[('ASME B36.10', '')]), '压力不改BODY_MANU'),
        ('钢衬PTFE管道 20/PTFE PN16 DN80 HG/T20538', make_output(body='衬里复合管', dn='DN80', material='20/PTFE', standards=[('HG/T20538', '')]), '压力不改BODY'),
        ('FLANGED PIPE, CL300, RF, PTFE/20, HG/T20538, SH/T3406;DN80', make_output(body='法兰管', dn='DN80', material='20/PTFE', standards=[('HG/T20538', ''), ('SH/T3406', '')]), '法兰线索优先于衬里线索'),
        ('PIPE 16MnD NB/T47009 SMLS BE THK=3.5mm DN50 300LB', make_output(body='直管', manu=['SMLS'], dn='DN50', thickness_kind='mm', thickness_value='3.5', material='16MnD', standards=[('NB/T47009', '')]), '压力材质规范不改TYPE'),
        ('PIPE Q235B HG/T20553(Ia) ERW DN50 SCH40', make_output(body='直管', manu=['ERW'], dn='DN50', thickness_kind='schedule', thickness_value='SCH40', material='Q235B', standards=[('HG/T20553', 'Ia')]), '规范不改MANU'),
    ]
    return [rec(desc, out, reason, 'NEGATIVE', 'TYPE_NEGATIVE') for desc, out, reason in cases]


def strip_meta(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{'input': r['input'], 'output': r['output']} for r in rows]


def dedupe(rows: list[dict[str, Any]], existing_inputs: set[str]) -> list[dict[str, Any]]:
    out = []
    seen = set(existing_inputs)
    for r in rows:
        inp = r['input'].strip()
        if inp in seen:
            continue
        seen.add(inp)
        out.append(r)
    return out


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    body = Counter()
    manu = Counter()
    conn = Counter()
    group = Counter()
    reason = Counter()
    for r in rows:
        t = r['output']['TYPE']
        body[t['BODY']] += 1
        for m in t.get('MANU', []):
            manu[m] += 1
        for c in t.get('CONN', []):
            conn[c] += 1
        group[r.get('_group', '')] += 1
        reason[r.get('_reason', '')] += 1
    return {'count': len(rows), 'body': dict(body), 'manu': dict(manu), 'conn': dict(conn), 'group': dict(group), 'reason': dict(reason)}


def main() -> None:
    base = json.loads(INPUT_DATASET.read_text(encoding='utf-8'))
    existing_inputs = {item['input'].strip() for item in base}
    aug = body_records() + manu_records() + conn_records() + negative_records()
    aug = dedupe(aug, existing_inputs)
    merged_meta = base + aug
    merged_train = strip_meta(base) + strip_meta(aug)
    AUG_PATH.write_text(json.dumps(aug, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    MERGED_META_PATH.write_text(json.dumps(merged_meta, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    MERGED_TRAIN_PATH.write_text(json.dumps(merged_train, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    summary = {'base_count': len(base), 'aug_summary': summarize(aug), 'merged_count': len(merged_train)}
    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
