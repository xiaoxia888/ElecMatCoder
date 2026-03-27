# -*- coding: utf-8 -*-
"""
训练过程曲线导出工具。

从 HuggingFace Trainer 的 log_history 中提取 step/loss/eval_loss/learning_rate，
并在训练结束后导出 PNG 曲线和原始 JSON。
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def _ensure_mpl_config_dir(output_dir: Path) -> None:
    mpl_dir = output_dir / ".mplconfig"
    mpl_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_dir))


def _prepare_series(log_history: list[dict[str, Any]]) -> dict[str, list[tuple[float, float]]]:
    train_loss: list[tuple[float, float]] = []
    eval_loss: list[tuple[float, float]] = []
    learning_rate: list[tuple[float, float]] = []

    for row in log_history:
        if not isinstance(row, dict):
            continue
        step = row.get("step")
        if step is None:
            continue
        try:
            step_val = float(step)
        except (TypeError, ValueError):
            continue

        if "loss" in row:
            try:
                train_loss.append((step_val, float(row["loss"])))
            except (TypeError, ValueError):
                pass
        if "eval_loss" in row:
            try:
                eval_loss.append((step_val, float(row["eval_loss"])))
            except (TypeError, ValueError):
                pass
        if "learning_rate" in row:
            try:
                learning_rate.append((step_val, float(row["learning_rate"])))
            except (TypeError, ValueError):
                pass

    return {
        "train_loss": train_loss,
        "eval_loss": eval_loss,
        "learning_rate": learning_rate,
    }


def _render_svg_line_chart(
    series_map: dict[str, list[tuple[float, float]]],
    output_file: Path,
    title: str,
    y_label: str,
) -> None:
    non_empty = {k: v for k, v in series_map.items() if v}
    if not non_empty:
        return

    width = 1000
    height = 600
    left = 70
    right = 20
    top = 50
    bottom = 55
    plot_w = width - left - right
    plot_h = height - top - bottom

    xs = [x for points in non_empty.values() for x, _ in points]
    ys = [y for points in non_empty.values() for _, y in points]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    if max_x == min_x:
        max_x = min_x + 1.0
    if max_y == min_y:
        max_y = min_y + 1.0

    def sx(x: float) -> float:
        return left + (x - min_x) / (max_x - min_x) * plot_w

    def sy(y: float) -> float:
        return top + plot_h - (y - min_y) / (max_y - min_y) * plot_h

    colors = {
        "train_loss": "#d97706",
        "eval_loss": "#2563eb",
        "learning_rate": "#059669",
    }

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{width/2}" y="28" text-anchor="middle" font-size="20" font-family="Arial">{title}</text>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_h}" stroke="#444" stroke-width="1"/>',
        f'<line x1="{left}" y1="{top+plot_h}" x2="{left+plot_w}" y2="{top+plot_h}" stroke="#444" stroke-width="1"/>',
        f'<text x="{width/2}" y="{height-15}" text-anchor="middle" font-size="14" font-family="Arial">Step</text>',
        f'<text x="20" y="{height/2}" text-anchor="middle" font-size="14" font-family="Arial" transform="rotate(-90 20,{height/2})">{y_label}</text>',
    ]

    for idx in range(5):
        y = top + idx * plot_h / 4
        val = max_y - idx * (max_y - min_y) / 4
        lines.append(
            f'<line x1="{left}" y1="{y:.2f}" x2="{left+plot_w}" y2="{y:.2f}" stroke="#e5e7eb" stroke-width="1"/>'
        )
        lines.append(
            f'<text x="{left-8}" y="{y+4:.2f}" text-anchor="end" font-size="12" font-family="Arial">{val:.6g}</text>'
        )

    for idx in range(5):
        x = left + idx * plot_w / 4
        val = min_x + idx * (max_x - min_x) / 4
        lines.append(
            f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{top+plot_h}" stroke="#f3f4f6" stroke-width="1"/>'
        )
        lines.append(
            f'<text x="{x:.2f}" y="{top+plot_h+20}" text-anchor="middle" font-size="12" font-family="Arial">{val:.0f}</text>'
        )

    legend_x = left + 10
    legend_y = top - 18
    for i, (name, points) in enumerate(non_empty.items()):
        path = " ".join(
            ("M" if j == 0 else "L") + f" {sx(x):.2f} {sy(y):.2f}"
            for j, (x, y) in enumerate(points)
        )
        color = colors.get(name, "#111827")
        lines.append(f'<path d="{path}" fill="none" stroke="{color}" stroke-width="2"/>')
        ly = legend_y + i * 18
        lines.append(f'<line x1="{legend_x}" y1="{ly}" x2="{legend_x+18}" y2="{ly}" stroke="{color}" stroke-width="3"/>')
        lines.append(f'<text x="{legend_x+24}" y="{ly+4}" font-size="12" font-family="Arial">{name}</text>')

    lines.append("</svg>")
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def export_training_curves(log_history: list[dict[str, Any]], output_dir: str | Path) -> dict[str, str]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    raw_history_path = output_path / "training_log_history.json"
    with open(raw_history_path, "w", encoding="utf-8") as f:
        json.dump(log_history, f, indent=2, ensure_ascii=False)

    series = _prepare_series(log_history)
    series_path = output_path / "training_curve_data.json"
    with open(series_path, "w", encoding="utf-8") as f:
        json.dump(series, f, indent=2, ensure_ascii=False)

    artifacts = {
        "log_history_json": str(raw_history_path),
        "curve_data_json": str(series_path),
    }

    train_loss = series["train_loss"]
    eval_loss = series["eval_loss"]
    learning_rate = series["learning_rate"]

    try:
        _ensure_mpl_config_dir(output_path)
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        if train_loss or eval_loss:
            fig, ax = plt.subplots(figsize=(10, 6))
            if train_loss:
                ax.plot(
                    [x for x, _ in train_loss],
                    [y for _, y in train_loss],
                    label="train_loss",
                    linewidth=1.8,
                    color="#d97706",
                )
            if eval_loss:
                ax.plot(
                    [x for x, _ in eval_loss],
                    [y for _, y in eval_loss],
                    label="eval_loss",
                    linewidth=1.8,
                    marker="o",
                    color="#2563eb",
                )
            ax.set_title("Loss Curve")
            ax.set_xlabel("Step")
            ax.set_ylabel("Loss")
            ax.grid(True, alpha=0.25)
            if train_loss or eval_loss:
                ax.legend()
            fig.tight_layout()
            loss_curve_path = output_path / "loss_curve.png"
            fig.savefig(loss_curve_path, dpi=180)
            plt.close(fig)
            artifacts["loss_curve_png"] = str(loss_curve_path)

        if learning_rate:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(
                [x for x, _ in learning_rate],
                [y for _, y in learning_rate],
                linewidth=1.8,
                color="#059669",
            )
            ax.set_title("Learning Rate Curve")
            ax.set_xlabel("Step")
            ax.set_ylabel("Learning Rate")
            ax.grid(True, alpha=0.25)
            fig.tight_layout()
            lr_curve_path = output_path / "learning_rate_curve.png"
            fig.savefig(lr_curve_path, dpi=180)
            plt.close(fig)
            artifacts["learning_rate_curve_png"] = str(lr_curve_path)
    except Exception as exc:
        artifacts["plot_warning"] = f"matplotlib_unavailable: {exc}"

    if train_loss or eval_loss:
        loss_curve_svg_path = output_path / "loss_curve.svg"
        _render_svg_line_chart(
            {"train_loss": train_loss, "eval_loss": eval_loss},
            loss_curve_svg_path,
            "Loss Curve",
            "Loss",
        )
        artifacts["loss_curve_svg"] = str(loss_curve_svg_path)

    if learning_rate:
        lr_curve_svg_path = output_path / "learning_rate_curve.svg"
        _render_svg_line_chart(
            {"learning_rate": learning_rate},
            lr_curve_svg_path,
            "Learning Rate Curve",
            "Learning Rate",
        )
        artifacts["learning_rate_curve_svg"] = str(lr_curve_svg_path)

    return artifacts
