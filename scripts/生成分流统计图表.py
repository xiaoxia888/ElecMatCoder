# -*- coding: utf-8 -*-
"""读取 Excel 并生成 shadcn 风格的分流统计 HTML 仪表盘。"""

from __future__ import annotations

import argparse
from html import escape
from pathlib import Path
from typing import Any

import pandas as pd


DIFFICULTY_COL_CANDIDATES = (
    "分流最终难度（0=困难，1=中等，2=简单）",
    "分流最终难度（0=困难，1=简单，2=二次简单）",
    "分流最终难度",
)
PROJECT_COL_CANDIDATES = ("项目名称",)
CORRECT_COL_CANDIDATES = ("是否正确", "是否匹配正确", "是否编码正确", "是否审核正确")

LEVEL_LABELS = {0: "困难", 1: "中等", 2: "简单"}
LEVEL_COLORS = {0: "#ef4444", 1: "#f59e0b", 2: "#3b82f6"}


def find_column(df: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    for name in candidates:
        if name in df.columns:
            return name
    return None


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.lower() == "nan" else text


def normalize_level(value: Any) -> int | None:
    text = clean_text(value)
    if not text:
        return None
    try:
        level = int(float(text))
    except ValueError:
        return None
    return level if level in LEVEL_LABELS else None


def normalize_correct(value: Any) -> int | None:
    text = clean_text(value)
    if text in {"1", "1.0", "true", "True", "TRUE", "是"}:
        return 1
    if text in {"0", "0.0", "false", "False", "FALSE", "否"}:
        return 0
    return None


def pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def load_dataframe(excel_path: Path, sheet_name: str | None, correct_col: str | None) -> tuple[pd.DataFrame, dict[str, Any]]:
    df = pd.read_excel(excel_path, sheet_name=sheet_name or 0, dtype=str).fillna("")

    difficulty_col = find_column(df, DIFFICULTY_COL_CANDIDATES)
    project_col = find_column(df, PROJECT_COL_CANDIDATES)
    actual_correct_col = correct_col if correct_col else find_column(df, CORRECT_COL_CANDIDATES)

    missing: list[str] = []
    if difficulty_col is None:
        missing.append("分流最终难度（0=困难，1=中等，2=简单）")
    if project_col is None:
        missing.append("项目名称")
    if missing:
        raise ValueError(f"缺少必要列: {missing}；实际列为: {list(df.columns)}")

    data = pd.DataFrame(
        {
            "项目名称": df[project_col].map(clean_text),
            "分流难度": df[difficulty_col].map(normalize_level),
        }
    )
    if actual_correct_col and actual_correct_col in df.columns:
        data["是否正确"] = df[actual_correct_col].map(normalize_correct)
    else:
        data["是否正确"] = pd.Series([None] * len(df), dtype="object")

    data = data.dropna(subset=["分流难度"])
    data["分流难度"] = data["分流难度"].astype(int)
    data["项目名称"] = data["项目名称"].replace("", "未命名项目")
    if data.empty:
        raise ValueError("清洗后没有可用于统计的数据，请检查“分流难度”列。")

    correct_data = data.dropna(subset=["是否正确"]).copy()
    if not correct_data.empty:
        correct_data["是否正确"] = correct_data["是否正确"].astype(int)

    meta = {
        "difficulty_col": difficulty_col,
        "project_col": project_col,
        "correct_col": actual_correct_col or "",
        "has_correct": bool(actual_correct_col) and not correct_data.empty,
        "all_rows": len(data),
        "correct_rows": len(correct_data),
    }
    return data, meta


def build_project_accuracy(correct_data: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        correct_data.groupby("项目名称", as_index=False)
        .agg(总数=("是否正确", "size"), 正确数=("是否正确", "sum"))
    )
    grouped["准确率"] = grouped["正确数"] / grouped["总数"]
    return grouped.sort_values(["准确率", "总数", "项目名称"], ascending=[False, False, True]).reset_index(drop=True)


def build_project_difficulty_counts(data: pd.DataFrame) -> pd.DataFrame:
    pivot = (
        data.pivot_table(index="项目名称", columns="分流难度", values="项目名称", aggfunc="size", fill_value=0)
        .reindex(columns=[0, 1, 2], fill_value=0)
    )
    pivot["总数"] = pivot.sum(axis=1)
    return pivot.sort_values("总数", ascending=False)


def build_project_level_accuracy(correct_data: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        correct_data.groupby(["项目名称", "分流难度"], as_index=False)
        .agg(总数=("是否正确", "size"), 正确数=("是否正确", "sum"))
    )
    grouped["准确率"] = grouped["正确数"] / grouped["总数"]
    return grouped


def build_level_accuracy(correct_data: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        correct_data.groupby("分流难度", as_index=False)
        .agg(总数=("是否正确", "size"), 正确数=("是否正确", "sum"))
        .sort_values("分流难度")
        .reset_index(drop=True)
    )
    grouped["准确率"] = grouped["正确数"] / grouped["总数"]
    return grouped


def clamp_text(text: str, max_len: int = 16) -> str:
    raw = str(text)
    return raw if len(raw) <= max_len else raw[: max_len - 1] + "…"


def svg_tooltip_rect(
    x: float,
    y: float,
    width: float,
    height: float,
    tip: str,
    radius: float = 6,
    extra_class: str = "",
) -> str:
    safe_tip = escape(tip, quote=True)
    extra = f" {extra_class.strip()}" if extra_class.strip() else ""
    return (
        f'<rect class="tip-target{extra}" data-tip="{safe_tip}" '
        f'x="{x}" y="{y}" width="{width}" height="{height}" rx="{radius}" '
        f'fill="transparent" pointer-events="all" />'
    )


def svg_total_difficulty(data: pd.DataFrame) -> str:
    counts = data["分流难度"].value_counts().reindex([0, 1, 2], fill_value=0)
    max_value = max(int(counts.max()), 1)
    width = 760
    height = 270
    left = 56
    right = 24
    top = 28
    bottom = 54
    chart_w = width - left - right
    chart_h = height - top - bottom
    bar_w = 120
    gap = (chart_w - bar_w * 3) / 4
    parts: list[str] = []

    for i, level in enumerate([0, 1, 2]):
        value = int(counts[level])
        x = left + gap + i * (bar_w + gap)
        bar_h = 0 if max_value == 0 else value / max_value * (chart_h - 8)
        y = top + chart_h - bar_h
        parts.append(f'<line x1="{x}" y1="{top+chart_h}" x2="{x+bar_w}" y2="{top+chart_h}" stroke="#27272a" />')
        parts.append(
            f'<rect class="tip-target tip-fill" data-tip="{escape(f"{LEVEL_LABELS[level]}: {value} 条", quote=True)}" '
            f'x="{x}" y="{y}" width="{bar_w}" height="{bar_h}" rx="8" fill="{LEVEL_COLORS[level]}" />'
        )
        parts.append(svg_tooltip_rect(x, y, bar_w, max(bar_h, 10), f"{LEVEL_LABELS[level]}: {value} 条", extra_class="tip-hitbox"))
        parts.append(f'<text x="{x + bar_w/2}" y="{y - 10}" fill="#fafafa" font-size="15" text-anchor="middle">{value}</text>')
        parts.append(f'<text x="{x + bar_w/2}" y="{height - 18}" fill="#a1a1aa" font-size="13" text-anchor="middle">{LEVEL_LABELS[level]}</text>')

    for tick in range(4):
        value = max_value * tick / 3
        y = top + chart_h - chart_h * tick / 3
        parts.append(f'<line x1="{left}" y1="{y}" x2="{width-right}" y2="{y}" stroke="#27272a" />')
        parts.append(f'<text x="{left-10}" y="{y+5}" fill="#71717a" font-size="12" text-anchor="end">{int(round(value))}</text>')

    return f'<svg viewBox="0 0 {width} {height}" class="chart-svg">{"".join(parts)}</svg>'


def svg_project_accuracy(correct_data: pd.DataFrame) -> str:
    stats = build_project_accuracy(correct_data)
    width = 820
    row_h = 38
    height = 48 + len(stats) * row_h + 20
    bar_left = 210
    bar_right = 150
    bar_w = width - bar_left - bar_right
    parts: list[str] = []

    for i, (_, row) in enumerate(stats.iterrows()):
        y = 30 + i * row_h
        rate = float(row["准确率"])
        fill_w = bar_w * rate
        parts.append(f'<text x="16" y="{y+17}" fill="#e4e4e7" font-size="13">{escape(clamp_text(str(row["项目名称"]), 14))}</text>')
        parts.append(f'<rect x="{bar_left}" y="{y}" width="{bar_w}" height="18" rx="6" fill="#18181b" />')
        parts.append(
            f'<rect class="tip-target tip-fill" '
            f'data-tip="{escape(f"{row["项目名称"]}｜准确率 {pct(rate)}｜正确 {int(row["正确数"])}｜总数 {int(row["总数"])}", quote=True)}" '
            f'x="{bar_left}" y="{y}" width="{fill_w}" height="18" rx="6" fill="url(#barBlue)" />'
        )
        parts.append(
            svg_tooltip_rect(
                bar_left,
                y,
                bar_w,
                18,
                f"{row['项目名称']}｜准确率 {pct(rate)}｜正确 {int(row['正确数'])}｜总数 {int(row['总数'])}",
                6,
                "tip-hitbox",
            )
        )
        parts.append(f'<text x="{bar_left+bar_w+14}" y="{y+14}" fill="#a1a1aa" font-size="12">{pct(rate)} ({int(row["正确数"])}/{int(row["总数"])})</text>')

    defs = """
    <defs>
      <linearGradient id="barBlue" x1="0" x2="1" y1="0" y2="0">
        <stop offset="0%" stop-color="#60a5fa"/>
        <stop offset="100%" stop-color="#3b82f6"/>
      </linearGradient>
    </defs>
    """
    return f'<svg viewBox="0 0 {width} {height}" class="chart-svg">{defs}{"".join(parts)}</svg>'


def svg_project_difficulty_stack(data: pd.DataFrame) -> str:
    counts = build_project_difficulty_counts(data)
    width = 820
    row_h = 40
    height = 48 + len(counts) * row_h + 20
    bar_left = 210
    bar_right = 130
    bar_w = width - bar_left - bar_right
    parts: list[str] = []

    for i, (project, row) in enumerate(counts.iterrows()):
        y = 30 + i * row_h
        total = int(row["总数"])
        cursor = bar_left
        parts.append(f'<text x="16" y="{y+17}" fill="#e4e4e7" font-size="13">{escape(clamp_text(str(project), 14))}</text>')
        parts.append(f'<rect x="{bar_left}" y="{y}" width="{bar_w}" height="18" rx="6" fill="#18181b" />')
        for level in [0, 1, 2]:
            count = int(row[level])
            seg_w = 0 if total == 0 else bar_w * count / total
            if seg_w > 0:
                tip = f"{project}｜{LEVEL_LABELS[level]} {count} 条｜占比 {pct(count / total if total else 0)}"
                parts.append(
                    f'<rect class="tip-target tip-fill" data-tip="{escape(tip, quote=True)}" '
                    f'x="{cursor}" y="{y}" width="{seg_w}" height="18" rx="6" fill="{LEVEL_COLORS[level]}" />'
                )
                parts.append(svg_tooltip_rect(cursor, y, seg_w, 18, tip, 6, "tip-hitbox"))
            cursor += seg_w
        parts.append(
            f'<text x="{bar_left+bar_w+14}" y="{y+14}" fill="#a1a1aa" font-size="12">总数 {total} ｜ 困难 {int(row[0])} ｜ 中等 {int(row[1])} ｜ 简单 {int(row[2])}</text>'
        )

    return f'<svg viewBox="0 0 {width} {height}" class="chart-svg">{"".join(parts)}</svg>'


def svg_total_level_accuracy(correct_data: pd.DataFrame) -> str:
    stats = build_level_accuracy(correct_data)
    lookup = {
        int(row["分流难度"]): {
            "总数": int(row["总数"]),
            "正确数": int(row["正确数"]),
            "准确率": float(row["准确率"]),
        }
        for _, row in stats.iterrows()
    }

    width = 760
    height = 270
    left = 56
    right = 24
    top = 28
    bottom = 54
    chart_w = width - left - right
    chart_h = height - top - bottom
    bar_w = 120
    gap = (chart_w - bar_w * 3) / 4
    parts: list[str] = []

    for tick in range(5):
        y = top + chart_h - chart_h * tick / 4
        parts.append(f'<line x1="{left}" y1="{y}" x2="{width-right}" y2="{y}" stroke="#27272a" />')
        parts.append(f'<text x="{left-10}" y="{y+5}" fill="#71717a" font-size="12" text-anchor="end">{int(tick*25)}%</text>')

    for i, level in enumerate([0, 1, 2]):
        item = lookup.get(level)
        rate = 0.0 if item is None else float(item["准确率"])
        total = 0 if item is None else int(item["总数"])
        correct = 0 if item is None else int(item["正确数"])
        x = left + gap + i * (bar_w + gap)
        bar_h = chart_h * rate
        y = top + chart_h - bar_h
        tip = f"{LEVEL_LABELS[level]}｜准确率 {pct(rate)}｜正确 {correct}｜总数 {total}"
        parts.append(f'<line x1="{x}" y1="{top+chart_h}" x2="{x+bar_w}" y2="{top+chart_h}" stroke="#27272a" />')
        parts.append(
            f'<rect class="tip-target tip-fill" data-tip="{escape(tip, quote=True)}" '
            f'x="{x}" y="{y}" width="{bar_w}" height="{bar_h}" rx="8" fill="{LEVEL_COLORS[level]}" />'
        )
        parts.append(svg_tooltip_rect(x, y, bar_w, max(bar_h, 12), tip, extra_class="tip-hitbox"))
        parts.append(f'<text x="{x + bar_w/2}" y="{max(y - 10, 16)}" fill="#fafafa" font-size="15" text-anchor="middle">{pct(rate)}</text>')
        parts.append(f'<text x="{x + bar_w/2}" y="{height - 18}" fill="#a1a1aa" font-size="13" text-anchor="middle">{LEVEL_LABELS[level]}</text>')

    return f'<svg viewBox="0 0 {width} {height}" class="chart-svg">{"".join(parts)}</svg>'


def svg_project_level_accuracy(correct_data: pd.DataFrame) -> str:
    grouped = build_project_level_accuracy(correct_data)
    project_order = build_project_accuracy(correct_data)["项目名称"].tolist()
    lookup: dict[tuple[str, int], float] = {}
    count_lookup: dict[tuple[str, int], tuple[int, int]] = {}
    for _, row in grouped.iterrows():
        key = (str(row["项目名称"]), int(row["分流难度"]))
        lookup[key] = float(row["准确率"])
        count_lookup[key] = (int(row["正确数"]), int(row["总数"]))

    group_count = max(len(project_order), 1)
    width = max(860, 96 + group_count * 112)
    height = 360
    left = 56
    right = 24
    top = 26
    bottom = 52
    chart_w = width - left - right
    chart_h = height - top - bottom
    group_w = chart_w / group_count
    bar_w = min(18, group_w / 4.8)
    parts: list[str] = []

    for tick in range(5):
        y = top + chart_h - chart_h * tick / 4
        parts.append(f'<line x1="{left}" y1="{y}" x2="{width-right}" y2="{y}" stroke="#27272a" />')
        parts.append(f'<text x="{left-10}" y="{y+5}" fill="#71717a" font-size="12" text-anchor="end">{int(tick*25)}%</text>')

    for idx, project in enumerate(project_order):
        gx = left + idx * group_w
        for offset, level in enumerate([0, 1, 2]):
            rate = lookup.get((project, level))
            x = gx + group_w * 0.16 + offset * (bar_w + 8)
            if rate is None:
                parts.append(f'<rect x="{x}" y="{top+chart_h-4}" width="{bar_w}" height="4" rx="2" fill="#27272a" />')
                continue
            h = chart_h * rate
            y = top + chart_h - h
            color = LEVEL_COLORS[level]
            correct, total = count_lookup[(project, level)]
            tip = f"{project}｜{LEVEL_LABELS[level]}｜准确率 {pct(rate)}｜正确 {correct}｜总数 {total}"
            parts.append(
                f'<rect class="tip-target tip-fill" data-tip="{escape(tip, quote=True)}" '
                f'x="{x}" y="{y}" width="{bar_w}" height="{h}" rx="5" fill="{color}" />'
            )
            parts.append(svg_tooltip_rect(x, y, bar_w, max(h, 12), tip, 5, "tip-hitbox"))
        label_x = gx + group_w / 2
        label_y = height - 18
        parts.append(
            f'<text x="{label_x}" y="{label_y}" fill="#a1a1aa" font-size="12" text-anchor="end" transform="rotate(-18 {label_x} {label_y})">{escape(clamp_text(project, 14))}</text>'
        )

    legend = []
    lx = left
    for level in [0, 1, 2]:
        legend.append(f'<rect x="{lx}" y="6" width="10" height="10" rx="3" fill="{LEVEL_COLORS[level]}" />')
        legend.append(f'<text x="{lx+16}" y="15" fill="#a1a1aa" font-size="12">{LEVEL_LABELS[level]}</text>')
        lx += 86

    return f'<svg viewBox="0 0 {width} {height}" width="{width}" height="{height}" class="chart-svg chart-svg-wide">{"".join(legend)}{"".join(parts)}</svg>'


def render_placeholder(title: str, message: str) -> str:
    return f"""
    <section class="card">
      <div class="card-head">
        <h2>{escape(title)}</h2>
        <p>{escape(message)}</p>
      </div>
      <div class="placeholder">
        <div class="placeholder-icon">∅</div>
        <div class="placeholder-text">{escape(message)}</div>
      </div>
    </section>
    """


def render_card(title: str, subtitle: str, svg: str, footer: str, chart_class: str = "") -> str:
    return f"""
    <section class="card{' ' + chart_class if 'span-2' in chart_class.split() else ''}">
      <div class="card-head">
        <h2>{escape(title)}</h2>
        <p>{escape(subtitle)}</p>
      </div>
      <div class="chart-wrap {chart_class}">{svg}</div>
      <div class="card-foot">{escape(footer)}</div>
    </section>
    """


def build_html(data: pd.DataFrame, meta: dict[str, Any], excel_path: Path) -> str:
    total = len(data)
    counts = data["分流难度"].value_counts().reindex([0, 1, 2], fill_value=0)
    has_correct = bool(meta["has_correct"])
    correct_data = data.dropna(subset=["是否正确"]).copy()
    subtitle = (
        f"输入文件：{escape(str(excel_path))} ｜ 难度列：{escape(meta['difficulty_col'])} ｜ "
        f"项目列：{escape(meta['project_col'])} ｜ 正确列：{escape(meta['correct_col'] or '未提供')} ｜ "
        f"样本数：{total:,}"
    )

    cards: list[str] = []
    if has_correct:
        accuracy = float(correct_data["是否正确"].mean())
        correct = int(correct_data["是否正确"].sum())
        wrong = len(correct_data) - correct
        deg = round(accuracy * 360, 2)
        cards.append(
            f"""
            <section class="card hero-card">
              <div class="hero-copy">
                <div class="eyebrow">Overall Accuracy</div>
                <h2>总体准确率</h2>
                <p>基于可判定正确性的 {len(correct_data):,} 条记录计算。</p>
                <div class="hero-metrics">
                  <div class="metric-chip"><span>{correct:,}</span><small>正确</small></div>
                  <div class="metric-chip"><span>{wrong:,}</span><small>错误</small></div>
                  <div class="metric-chip"><span>{pct(accuracy)}</span><small>准确率</small></div>
                </div>
              </div>
              <div class="hero-chart">
                <div class="ring tip-target" data-tip="总体准确率 {pct(accuracy)}｜正确 {correct}｜错误 {wrong}｜样本 {len(correct_data)}" style="background:conic-gradient(#3b82f6 0deg {deg}deg, #27272a {deg}deg 360deg);">
                  <div class="ring-inner">
                    <strong>{pct(accuracy)}</strong>
                    <span>Accuracy</span>
                  </div>
                </div>
              </div>
            </section>
            """
        )
    else:
        cards.append(render_placeholder("总体准确率", "当前 Excel 没有“是否正确”列，准确率相关图表暂时无法计算。"))

    cards.append(
        render_card(
            "总分流难度数量",
            "按困难 / 中等 / 简单统计全量样本分布",
            svg_total_difficulty(data),
            f"困难 {int(counts[0])} ｜ 中等 {int(counts[1])} ｜ 简单 {int(counts[2])}",
        )
    )

    if has_correct:
        cards.append(
            render_card(
                "总分流难度准确率",
                "从困难 / 中等 / 简单三个维度对比整体准确率",
                svg_total_level_accuracy(correct_data),
                "悬停柱体可查看该难度下的正确数、总数与准确率",
            )
        )
    else:
        cards.append(render_placeholder("总分流难度准确率", "缺少“是否正确”列，无法生成分难度准确率图。"))

    if has_correct:
        cards.append(
            render_card(
                "分项目准确率",
                "展示各项目的整体准确率对比",
                svg_project_accuracy(correct_data),
                "支持滚动查看全部项目，悬停可查看详细数据",
                "scroll-y",
            )
        )
    else:
        cards.append(render_placeholder("分项目准确率", "缺少“是否正确”列，无法生成项目准确率图。"))

    cards.append(
        render_card(
            "分项目分流难度数量",
            "每个项目内部的困难 / 中等 / 简单数量堆叠对比",
            svg_project_difficulty_stack(data),
            "支持滚动查看全部项目，悬停可查看详细数据",
            "scroll-y",
        )
    )

    if has_correct:
        cards.append(
            render_card(
                "分项目分难度准确率",
                "每个项目在不同难度下的准确率分组柱状图",
                svg_project_level_accuracy(correct_data),
                "支持横向滚动查看全部项目，悬停可查看详细数据",
                "scroll-x-wide span-2",
            )
        )
    else:
        cards.append(render_placeholder("分项目分难度准确率", "缺少“是否正确”列，无法生成分难度准确率图。"))

    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>分流统计仪表盘</title>
  <style>
    :root {{
      --bg: #09090b;
      --panel: #111113;
      --panel-2: #18181b;
      --line: #27272a;
      --text: #fafafa;
      --muted: #a1a1aa;
      --blue: #3b82f6;
      --red: #ef4444;
      --amber: #f59e0b;
      --radius: 12px;
      --shadow: 0 20px 60px rgba(0, 0, 0, 0.35);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background:
        radial-gradient(circle at top left, rgba(59,130,246,0.18), transparent 24%),
        radial-gradient(circle at top right, rgba(245,158,11,0.10), transparent 18%),
        var(--bg);
      color: var(--text);
      font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "PingFang SC", "Microsoft YaHei", sans-serif;
    }}
    .page {{
      max-width: 1560px;
      margin: 0 auto;
      padding: 32px 24px 64px;
    }}
    .header {{
      margin-bottom: 24px;
    }}
    .header h1 {{
      margin: 0;
      font-size: 34px;
      line-height: 1.1;
      letter-spacing: -0.03em;
      font-weight: 700;
    }}
    .header p {{
      margin: 10px 0 0;
      color: var(--muted);
      font-size: 13px;
      line-height: 1.7;
      word-break: break-all;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 22px;
    }}
    .card {{
      background: linear-gradient(180deg, rgba(24,24,27,0.98), rgba(17,17,19,0.98));
      border: 1px solid rgba(255,255,255,0.08);
      border-radius: var(--radius);
      box-shadow: var(--shadow);
      padding: 24px;
      overflow: hidden;
    }}
    .span-2 {{
      grid-column: 1 / -1;
    }}
    .hero-card {{
      grid-column: 1 / -1;
      display: grid;
      grid-template-columns: 1.2fr 340px;
      align-items: center;
      gap: 28px;
      min-height: 280px;
    }}
    .eyebrow {{
      display: inline-flex;
      align-items: center;
      gap: 8px;
      color: #93c5fd;
      font-size: 12px;
      font-weight: 600;
      letter-spacing: 0.04em;
      text-transform: uppercase;
    }}
    .hero-copy h2, .card-head h2 {{
      margin: 8px 0 0;
      font-size: 28px;
      line-height: 1.15;
      letter-spacing: -0.03em;
    }}
    .hero-copy p, .card-head p {{
      margin: 10px 0 0;
      color: var(--muted);
      font-size: 14px;
      line-height: 1.6;
    }}
    .hero-metrics {{
      display: flex;
      gap: 14px;
      flex-wrap: wrap;
      margin-top: 26px;
    }}
    .metric-chip {{
      min-width: 132px;
      padding: 14px 16px;
      border-radius: 10px;
      background: rgba(255,255,255,0.04);
      border: 1px solid rgba(255,255,255,0.08);
    }}
    .metric-chip span {{
      display: block;
      font-size: 28px;
      font-weight: 700;
      letter-spacing: -0.03em;
    }}
    .metric-chip small {{
      display: block;
      margin-top: 6px;
      color: var(--muted);
      font-size: 12px;
    }}
    .hero-chart {{
      display: flex;
      justify-content: center;
      align-items: center;
    }}
    .ring {{
      width: 240px;
      height: 240px;
      border-radius: 999px;
      display: flex;
      align-items: center;
      justify-content: center;
    }}
    .ring-inner {{
      width: 152px;
      height: 152px;
      border-radius: 999px;
      background: #0f0f11;
      border: 1px solid rgba(255,255,255,0.08);
      display: flex;
      align-items: center;
      justify-content: center;
      flex-direction: column;
    }}
    .ring-inner strong {{
      font-size: 30px;
      line-height: 1;
      letter-spacing: -0.03em;
    }}
    .ring-inner span {{
      margin-top: 8px;
      color: var(--muted);
      font-size: 12px;
    }}
    .card-head {{
      margin-bottom: 18px;
    }}
    .chart-wrap {{
      border-top: 1px solid rgba(255,255,255,0.06);
      border-bottom: 1px solid rgba(255,255,255,0.06);
      padding: 12px 0;
      overflow-x: auto;
      overflow-y: hidden;
    }}
    .chart-wrap.scroll-y {{
      max-height: 520px;
      overflow-y: auto;
      overflow-x: hidden;
      padding-right: 8px;
    }}
    .chart-wrap.scroll-x-wide {{
      overflow-x: auto;
      overflow-y: hidden;
    }}
    .chart-svg {{
      width: 100%;
      min-width: 680px;
      display: block;
    }}
    .chart-wrap.scroll-y .chart-svg {{
      min-width: 0;
    }}
    .chart-wrap.scroll-x-wide .chart-svg-wide {{
      width: auto;
      min-width: unset;
      max-width: none;
    }}
    .card-foot {{
      margin-top: 16px;
      color: var(--muted);
      font-size: 13px;
      line-height: 1.5;
    }}
    .placeholder {{
      min-height: 220px;
      display: flex;
      align-items: center;
      justify-content: center;
      flex-direction: column;
      gap: 12px;
      border: 1px dashed rgba(255,255,255,0.12);
      border-radius: 10px;
      background: rgba(255,255,255,0.02);
    }}
    .placeholder-icon {{
      width: 56px;
      height: 56px;
      border-radius: 999px;
      display: flex;
      align-items: center;
      justify-content: center;
      background: rgba(255,255,255,0.06);
      color: var(--muted);
      font-size: 24px;
    }}
    .placeholder-text {{
      color: var(--muted);
      text-align: center;
      max-width: 480px;
      line-height: 1.7;
      font-size: 14px;
    }}
    .tooltip {{
      position: fixed;
      z-index: 9999;
      pointer-events: none;
      background: rgba(24,24,27,0.96);
      color: #fafafa;
      border: 1px solid rgba(255,255,255,0.10);
      border-radius: 8px;
      padding: 8px 10px;
      font-size: 12px;
      line-height: 1.5;
      box-shadow: 0 14px 30px rgba(0,0,0,0.28);
      max-width: 280px;
      opacity: 0;
      transform: translateY(4px);
      transition: opacity .12s ease, transform .12s ease;
      white-space: normal;
    }}
    .tooltip.show {{
      opacity: 1;
      transform: translateY(0);
    }}
    .tip-target {{
      cursor: pointer;
    }}
    .tip-fill {{
      transition: filter .12s ease, opacity .12s ease, transform .12s ease;
      transform-box: fill-box;
      transform-origin: center;
    }}
    .tip-fill.active {{
      filter: brightness(1.18) drop-shadow(0 0 10px rgba(255,255,255,0.18));
      opacity: 1;
    }}
    .tip-hitbox.active {{
      fill: rgba(255,255,255,0.04);
      stroke: rgba(255,255,255,0.18);
      stroke-width: 1;
    }}
    .chart-wrap::-webkit-scrollbar {{
      width: 10px;
      height: 10px;
    }}
    .chart-wrap::-webkit-scrollbar-track {{
      background: rgba(255,255,255,0.04);
      border-radius: 999px;
    }}
    .chart-wrap::-webkit-scrollbar-thumb {{
      background: rgba(255,255,255,0.18);
      border-radius: 999px;
    }}
    .chart-wrap::-webkit-scrollbar-thumb:hover {{
      background: rgba(255,255,255,0.28);
    }}
    @media (max-width: 1100px) {{
      .grid {{
        grid-template-columns: 1fr;
      }}
      .hero-card {{
        grid-template-columns: 1fr;
      }}
    }}
  </style>
</head>
<body>
  <div class="page">
    <header class="header">
      <h1>分流统计仪表盘</h1>
      <p>{subtitle}</p>
    </header>
    <main class="grid">
      {''.join(cards)}
    </main>
  </div>
  <div id="chart-tooltip" class="tooltip"></div>
  <script>
    (function() {{
      const tooltip = document.getElementById('chart-tooltip');
      const targets = document.querySelectorAll('.tip-target');
      if (!tooltip || !targets.length) return;

      function moveTooltip(event) {{
        const offset = 14;
        const maxX = window.innerWidth - tooltip.offsetWidth - 12;
        const maxY = window.innerHeight - tooltip.offsetHeight - 12;
        const x = Math.min(event.clientX + offset, maxX);
        const y = Math.min(event.clientY + offset, maxY);
        tooltip.style.left = x + 'px';
        tooltip.style.top = y + 'px';
      }}

      targets.forEach((node) => {{
        function activate() {{
          node.classList.add('active');
        }}
        function deactivate() {{
          node.classList.remove('active');
        }}
        node.addEventListener('mouseenter', (event) => {{
          const text = node.getAttribute('data-tip');
          if (!text) return;
          tooltip.textContent = text;
          tooltip.classList.add('show');
          activate();
          moveTooltip(event);
        }});
        node.addEventListener('mousemove', moveTooltip);
        node.addEventListener('mouseleave', () => {{
          tooltip.classList.remove('show');
          deactivate();
        }});
      }});
    }})();
  </script>
</body>
</html>
"""


def write_summary_excel(data: pd.DataFrame, output_dir: Path, has_correct: bool) -> None:
    overall = pd.DataFrame(
        [
            {
                "总样本数": len(data),
                "困难数量": int((data["分流难度"] == 0).sum()),
                "中等数量": int((data["分流难度"] == 1).sum()),
                "简单数量": int((data["分流难度"] == 2).sum()),
            }
        ]
    )
    if has_correct:
        correct_data = data.dropna(subset=["是否正确"]).copy()
        overall["正确样本数"] = len(correct_data)
        overall["正确数"] = int(correct_data["是否正确"].sum())
        overall["准确率"] = float(correct_data["是否正确"].mean()) if len(correct_data) else None

    with pd.ExcelWriter(output_dir / "图表汇总数据.xlsx", engine="openpyxl") as writer:
        overall.to_excel(writer, sheet_name="总体概览", index=False)
        build_project_difficulty_counts(data).reset_index().to_excel(writer, sheet_name="分项目难度数量", index=False)
        if has_correct:
            correct_data = data.dropna(subset=["是否正确"]).copy()
            build_level_accuracy(correct_data).to_excel(writer, sheet_name="总分难度准确率", index=False)
            build_project_accuracy(correct_data).to_excel(writer, sheet_name="分项目准确率", index=False)
            build_project_level_accuracy(correct_data).to_excel(writer, sheet_name="分项目分难度准确率", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="读取 Excel 并生成 shadcn 风格的分流统计 HTML 仪表盘")
    parser.add_argument("excel_path", help="输入 Excel 路径")
    parser.add_argument("-o", "--output", help="输出 HTML 路径，默认在输入文件旁边生成 *_仪表盘.html")
    parser.add_argument("--sheet", help="Excel sheet 名，默认读取第一个")
    parser.add_argument("--correct-col", help="是否正确列名，默认自动尝试常见列名")
    args = parser.parse_args()

    excel_path = Path(args.excel_path).expanduser().resolve()
    if not excel_path.exists():
        raise FileNotFoundError(f"找不到文件: {excel_path}")

    output_path = (
        Path(args.output).expanduser().resolve()
        if args.output
        else excel_path.with_name(f"{excel_path.stem}_仪表盘.html")
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data, meta = load_dataframe(excel_path, args.sheet, args.correct_col)
    output_path.write_text(build_html(data, meta, excel_path), encoding="utf-8")
    write_summary_excel(data, output_path.parent, bool(meta["has_correct"]))

    print(f"输入文件: {excel_path}")
    print(f"使用难度列: {meta['difficulty_col']}")
    print(f"使用项目列: {meta['project_col']}")
    print(f"使用正确列: {meta['correct_col'] or '未提供'}")
    print(f"样本数: {len(data)}")
    print(f"输出 HTML: {output_path}")
    print(f"汇总数据: {output_path.parent / '图表汇总数据.xlsx'}")
    if not meta["has_correct"]:
        print("提示: 当前未检测到有效的“是否正确”列，准确率相关卡片已自动降级为提示卡片。")


if __name__ == "__main__":
    main()
