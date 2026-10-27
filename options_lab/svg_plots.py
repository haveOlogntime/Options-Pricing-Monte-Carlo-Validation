"""Small dependency-free SVG charts for reproducible experiment outputs."""
from __future__ import annotations

from html import escape
from pathlib import Path
import math
import numpy as np

WIDTH, HEIGHT = 880, 500
COLORS = {
    "plain": "#2563eb",
    "antithetic": "#dc2626",
    "control_variate": "#059669",
    "zero": "#111827",
}


def _text(x, y, label, size=12, anchor="middle", fill="#374151", weight="400", rotate=None):
    transform = "" if rotate is None else f' transform="rotate({rotate} {x} {y})"'
    return (
        f'<text x="{x:.2f}" y="{y:.2f}" text-anchor="{anchor}" '
        f'font-family="Arial,sans-serif" font-size="{size}" font-weight="{weight}" '
        f'fill="{fill}"{transform}>{escape(str(label))}</text>'
    )


def _start(title, subtitle=""):
    pieces = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        _text(WIDTH / 2, 30, title, 21, fill="#111827", weight="600"),
    ]
    if subtitle:
        pieces.append(_text(WIDTH / 2, 52, subtitle, 12))
    return pieces


def _save(path, pieces):
    pieces.append("</svg>")
    Path(path).write_text(chr(10).join(pieces), encoding="utf-8")


def line_chart(path, series, title, xlabel, ylabel, x_log=False, y_log=False, benchmark=None):
    """Render lines from dictionaries with label, x, y and optional color."""
    left, right, top, bottom = 92, 816, 82, 382
    if not series:
        raise ValueError("series must not be empty")
    valid_x = np.concatenate([np.asarray(item["x"], dtype=float) for item in series])
    valid_y = np.concatenate([np.asarray(item["y"], dtype=float) for item in series])
    if valid_x.size == 0 or valid_y.size == 0:
        raise ValueError("series must contain at least one point")
    if x_log and np.any(valid_x <= 0):
        raise ValueError("x values must be positive on a log axis")
    if y_log and np.any(valid_y <= 0):
        raise ValueError("y values must be positive on a log axis")
    xmin, xmax = float(valid_x.min()), float(valid_x.max())
    ymin, ymax = float(valid_y.min()), float(valid_y.max())
    if x_log:
        xmin, xmax = math.log10(xmin), math.log10(xmax)
    if y_log:
        ymin, ymax = math.log10(ymin), math.log10(ymax)
    span_x = max(xmax - xmin, 1e-12)
    span_y = max(ymax - ymin, 1e-12)
    ymin -= 0.08 * span_y
    ymax += 0.12 * span_y

    def x_pixel(value):
        transformed = math.log10(value) if x_log else value
        return left + (transformed - xmin) / span_x * (right - left)

    def y_pixel(value):
        transformed = math.log10(value) if y_log else value
        return bottom - (transformed - ymin) / (ymax - ymin) * (bottom - top)

    pieces = _start(title)
    for i in range(5):
        fraction = i / 4
        yy = bottom - fraction * (bottom - top)
        value_y = 10 ** (ymin + fraction * (ymax - ymin)) if y_log else ymin + fraction * (ymax - ymin)
        label_y = f"{value_y:.2e}" if abs(value_y) < 0.01 or abs(value_y) >= 1000 else f"{value_y:.3g}"
        pieces.append(f'<line x1="{left}" y1="{yy:.2f}" x2="{right}" y2="{yy:.2f}" stroke="#e5e7eb"/>')
        pieces.append(_text(left - 12, yy + 4, label_y, 11, anchor="end"))
        xx = left + fraction * (right - left)
        value_x = 10 ** (xmin + fraction * span_x) if x_log else xmin + fraction * span_x
        pieces.append(f'<line x1="{xx:.2f}" y1="{top}" x2="{xx:.2f}" y2="{bottom}" stroke="#f3f4f6"/>')
        pieces.append(_text(xx, bottom + 22, f"{value_x:.3g}", 11))
    pieces.extend([
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" stroke="#9ca3af"/>',
        f'<line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" stroke="#9ca3af"/>',
        _text((left + right) / 2, 436, xlabel, 13),
        _text(25, (top + bottom) / 2, ylabel, 13, rotate=-90),
    ])
    for item in series:
        x_values = np.asarray(item["x"], dtype=float)
        y_values = np.asarray(item["y"], dtype=float)
        coordinates = [(x_pixel(float(x)), y_pixel(float(y))) for x, y in zip(x_values, y_values)]
        points = " ".join(f"{x:.2f},{y:.2f}" for x, y in coordinates)
        color = item.get("color", COLORS.get(item["label"], "#7c43ed"))
        pieces.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2.4" stroke-linejoin="round"/>')
        for x, y in coordinates:
            pieces.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="3.2" fill="{color}"/>')
    if benchmark is not None and (not y_log or benchmark > 0):
        yy = y_pixel(float(benchmark))
        pieces.append(f'<line x1="{left}" y1="{yy:.2f}" x2="{right}" y2="{yy:.2f}" stroke="#111827" stroke-dasharray="6 5" stroke-width="1.5"/>')
    spacing = (right - left) / (len(series) + 1)
    for index, item in enumerate(series, 1):
        xx = left + index * spacing
        color = item.get("color", COLORS.get(item["label"], "#7c43ed"))
        pieces.append(f'<line x1="{xx - 28:.2f}" y1="466" x2="{xx - 4:.2f}" y2="466" stroke="{color}" stroke-width="2.4"/>')
        pieces.append(_text(xx + 3, 470, item["label"].replace("_", " "), 11, anchor="start"))
    _save(path, pieces)


def _quantile(sorted_values, probability):
    index = (len(sorted_values) - 1) * probability
    low = int(math.floor(index))
    high = int(math.ceil(index))
    return sorted_values[low] + (sorted_values[high] - sorted_values[low]) * (index - low)


def boxplot_chart(path, groups, title, ylabel, benchmark=None):
    """Render median, interquartile box, 5th/95th whiskers and sample mean."""
    left, right, top, bottom = 105, 815, 82, 385
    all_values = np.concatenate([np.asarray(group["values"], dtype=float) for group in groups])
    low, high = float(all_values.min()), float(all_values.max())
    if benchmark is not None:
        low, high = min(low, benchmark), max(high, benchmark)
    span = max(high - low, 1e-9)
    low -= 0.10 * span
    high += 0.12 * span
    y_pixel = lambda value: bottom - (value - low) / (high - low) * (bottom - top)
    pieces = _start(title, "Boxes show the interquartile range; whiskers show the 5th–95th percentiles")
    for index in range(5):
        value = low + index * (high - low) / 4
        yy = y_pixel(value)
        pieces.append(f'<line x1="{left}" y1="{yy:.2f}" x2="{right}" y2="{yy:.2f}" stroke="#e5e7eb"/>')
        pieces.append(_text(left - 12, yy + 4, f"{value:.3f}", 11, anchor="end"))
    if benchmark is not None:
        yy = y_pixel(benchmark)
        pieces.append(f'<line x1="{left}" y1="{yy:.2f}" x2="{right}" y2="{yy:.2f}" stroke="#111827" stroke-dasharray="6 5" stroke-width="1.4"/>')
    spacing = (right - left) / (len(groups) + 1)
    for index, group in enumerate(groups, 1):
        values = np.sort(np.asarray(group["values"], dtype=float))
        q05, q25, median, q75, q95 = [_quantile(values, p) for p in (0.05, 0.25, 0.5, 0.75, 0.95)]
        mean = float(values.mean())
        xx = left + index * spacing
        color = group.get("color", COLORS.get(group["label"], "#7c43ed"))
        pieces.extend([
            f'<line x1="{xx:.2f}" y1="{y_pixel(q05):.2f}" x2="{xx:.2f}" y2="{y_pixel(q95):.2f}" stroke="{color}" stroke-width="2"/>',
            f'<line x1="{xx-9:.2f}" y1="{y_pixel(q05):.2f}" x2="{xx+9:.2f}" y2="{y_pixel(q05):.2f}" stroke="{color}" stroke-width="2"/>',
            f'<line x1="{xx-9:.2f}" y1="{y_pixel(q95):.2f}" x2="{xx+9:.2f}" y2="{y_pixel(q95):.2f}" stroke="{color}" stroke-width="2"/>',
            f'<rect x="{xx-29:.2f}" y="{y_pixel(q75):.2f}" width="58" height="{max(1, y_pixel(q25)-y_pixel(q75)):.2f}" fill="{color}" fill-opacity="0.18" stroke="{color}" stroke-width="1.8"/>',
            f'<line x1="{xx-29:.2f}" y1="{y_pixel(median):.2f}" x2="{xx+29:.2f}" y2="{y_pixel(median):.2f}" stroke="{color}" stroke-width="2.4"/>',
            f'<circle cx="{xx:.2f}" cy="{y_pixel(mean):.2f}" r="3.5" fill="{color}"/>',
            _text(xx, bottom + 24, group["label"].replace("_", " "), 11),
        ])
    pieces.extend([_text(25, (top + bottom) / 2, ylabel, 13, rotate=-90), _text((left + right) / 2, 436, "Estimator", 13)])
    _save(path, pieces)


def scatter_chart(path, x_values, y_values, color_values, title, xlabel, ylabel, color_label):
    """Scatter option quotes by strike and expiry, with volatility encoded by hue."""
    left, right, top, bottom = 100, 795, 82, 380
    x_values = np.asarray(x_values, dtype=float)
    y_values = np.asarray(y_values, dtype=float)
    color_values = np.asarray(color_values, dtype=float)
    xmin, xmax = float(x_values.min()), float(x_values.max())
    ymin, ymax = float(y_values.min()), float(y_values.max())
    dx, dy = max(xmax - xmin, 1e-9), max(ymax - ymin, 1e-9)
    x_pixel = lambda value: left + (value - (xmin - 0.05 * dx)) / (1.10 * dx) * (right - left)
    y_pixel = lambda value: bottom - (value - (ymin - 0.05 * dy)) / (1.10 * dy) * (bottom - top)
    pieces = _start(title, "Each marker is a quote; color encodes the inferred volatility")
    for index in range(5):
        xx_value = xmin + index * dx / 4
        yy_value = ymin + index * dy / 4
        xx, yy = x_pixel(xx_value), y_pixel(yy_value)
        pieces.extend([
            f'<line x1="{xx:.2f}" y1="{top}" x2="{xx:.2f}" y2="{bottom}" stroke="#f3f4f6"/>',
            _text(xx, bottom + 21, f"{xx_value:g}", 11),
            f'<line x1="{left}" y1="{yy:.2f}" x2="{right}" y2="{yy:.2f}" stroke="#e5e7eb"/>',
            _text(left - 12, yy + 4, f"{yy_value:.3g}", 11, anchor="end"),
        ])
    color_min, color_max = float(color_values.min()), float(color_values.max())
    color_span = max(color_max - color_min, 1e-12)
    for xx_value, yy_value, value in zip(x_values, y_values, color_values):
        hue = 230 - 230 * (value - color_min) / color_span
        pieces.append(f'<circle cx="{x_pixel(xx_value):.2f}" cy="{y_pixel(yy_value):.2f}" r="7" fill="hsl({hue:.1f},68%,48%)" stroke="#fff" stroke-width="1.3"><title>Strike {xx_value:g}; maturity {yy_value:g}; {escape(color_label)} {value:.4f}</title></circle>')
    pieces.extend([
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" stroke="#9ca3af"/>',
        f'<line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" stroke="#9ca3af"/>',
        _text((left + right) / 2, 432, xlabel, 13),
        _text(25, (top + bottom) / 2, ylabel, 13, rotate=-90),
        _text(right - 8, 416, f"{color_min*100:.1f}% to {color_max*100:.1f}%", 11, anchor="end"),
    ])
    _save(path, pieces)


def range_chart(path, labels, lows, medians, highs, title, ylabel):
    """Plot terminal error medians and 5th–95th percentile intervals."""
    left, right, top, bottom = 105, 815, 82, 365
    low, high = min(float(np.min(lows)), 0.0), max(float(np.max(highs)), 0.0)
    span = max(high - low, 1e-9)
    low -= 0.08 * span
    high += 0.08 * span
    y_pixel = lambda value: bottom - (value - low) / (high - low) * (bottom - top)
    pieces = _start(title, "Bars show the 5th–95th percentile; point marks the median")
    for index in range(5):
        value = low + index * (high - low) / 4
        yy = y_pixel(value)
        pieces.append(f'<line x1="{left}" y1="{yy:.2f}" x2="{right}" y2="{yy:.2f}" stroke="#e5e7eb"/>')
        pieces.append(_text(left - 12, yy + 4, f"{value:.2f}", 11, anchor="end"))
    pieces.append(f'<line x1="{left}" y1="{y_pixel(0):.2f}" x2="{right}" y2="{y_pixel(0):.2f}" stroke="#111827" stroke-dasharray="6 5"/>')
    spacing = (right - left) / (len(labels) + 1)
    for index, (label, low_value, median, high_value) in enumerate(zip(labels, lows, medians, highs), 1):
        xx = left + index * spacing
        color = ("#2563eb", "#dc2626", "#059669")[(index - 1) // 2 % 3]
        pieces.extend([
            f'<line x1="{xx:.2f}" y1="{y_pixel(low_value):.2f}" x2="{xx:.2f}" y2="{y_pixel(high_value):.2f}" stroke="{color}" stroke-width="4"/>',
            f'<line x1="{xx-7:.2f}" y1="{y_pixel(low_value):.2f}" x2="{xx+7:.2f}" y2="{y_pixel(low_value):.2f}" stroke="{color}" stroke-width="2"/>',
            f'<line x1="{xx-7:.2f}" y1="{y_pixel(high_value):.2f}" x2="{xx+7:.2f}" y2="{y_pixel(high_value):.2f}" stroke="{color}" stroke-width="2"/>',
            f'<circle cx="{xx:.2f}" cy="{y_pixel(median):.2f}" r="4.5" fill="{color}"/>',
            _text(xx, bottom + 20, label, 10),
        ])
    pieces.extend([_text(25, (top + bottom) / 2, ylabel, 13, rotate=-90), _text((left + right) / 2, 438, "Rehedging intervals / proportional transaction cost", 13)])
    _save(path, pieces)
