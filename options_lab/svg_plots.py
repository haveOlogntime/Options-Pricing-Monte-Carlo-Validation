"""Small dependency-free SVG charts for the checked-in experiment outputs."""
from __future__ import annotations
from html import escape
from pathlib import Path
import math
import numpy as np

WIDTH, HEIGHT = 880, 500
COLORS = {"plain": "#2563eb", "antithetic": "#dc2626", "control_variate": "#059669", "zero": "#111827"}


def _text(x, y, label, size=12, anchor="middle", fill="#374151", weight="400", rotate=None):
    transform = f' transform="rotate({rotate} {x} {y})"' if rotate is not None else ""
    return f'<text x="{x:.2f}" y="{y:.2f}" text-anchor="{anchor}" font-family="Arial,sans-serif" font-size="{size}" font-weight="{weight}" fill="{fill}"{transform}>{escape(str(label))}</text>'


def _start(title, subtitle=""):
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}">', '<rect width="100%" height="100%" fill="#fff"/>', _text(WIDTH/2, 30, title, 21, fill="#111827", weight="600")]
    if subtitle:
        parts.append(_text(WIDTH/2, 52, subtitle, 12))
    return parts


def _save(path, parts):
    parts.append("</svg>")
    Path(path).write_text("\n".join(parts), encoding="utf-8")


def line_chart(path, series, title, xlabel, ylabel, x_log=False, y_log=False, benchmark=None):
    left, right, top, bottom = 92, 816, 82, 382
    all_x = np.concatenate([np.asarray(item["x"], dtype=float) for item in series])
    all_y = np.concatenate([np.asarray(item["y"], dtype=float) for item in series])
    if x_log:
        all_x = all_x[all_x > 0]
    if y_log:
        all_y = all_y[all_y > 0]
    xmin, xmax = float(all_x.min()), float(all_x.max())
    ymin, ymax = float(all_y.min()), float(all_y.max())
    if x_log:
        xmin, xmax = math.log10(xmin), math.log10(xmax)
    if y_log:
        ymin, ymax = math.log10(ymin), math.log10(ymax)
    span_x = max(xmax - xmin, 1e-12)
    span_y = max(ymax - ymin, 1e-12)
    ymin -= 0.08 * span_y
    ymax += 0.12 * span_y
    def px(value):
        value = math.log10(value) if x_log else value
        return left + (value - xmin) / span_x * (right - left)
    def py(value):
        value = math.log10(value) if y_log else value
        return bottom - (value - ymin) / (ymax - ymin) * (bottom - top)
    parts = _start(title)
    for i in range(5):
        frac = i / 4
        yy = bottom - frac * (bottom-top)
        val = (10 ** (ymin + frac*(ymax-ymin))) if y_log else ymin + frac*(ymax-ymin)
        label = f"{val:.1e}" if abs(val) < .01 or abs(val) >= 1000 else f"{val:.3g}"
        parts.extend([f'<line x1="{left}" y1="{yy:.2f}" x2="{right}" y2="{yy:.2f}" stroke="#e5e7eb"/>', _text(left-12, yy+4, label, 11, anchor="end")])
        valx = (10 ** (xmin + frac*span_x)) if x_log else xmin + frac*span_x
        xlab = f"{valx:.0f}" if valx >= 100 else f"{valx:.2g}"
        xx = left + frac*(right-left)
        parts.extend([f'<line x1="{xx:.2f}" y1="{top}" x2="{xx:.2f}" y2="{bottom}" stroke="#f3f4f6"/>', _text(xx, bottom+22, xlab, 11)])
    parts.extend([f'<line x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" stroke="#9ca3af"/>', f'<line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" stroke="#9ca3af"/>', _text((left+right)/2, 436, xlabel, 13), _text(25, (top+bottom)/2, ylabel, 13, rotate=-90)])
    for item in series:
        coords = [(px(float(x)), py(float(y))) for x, y in zip(item["x"], item["y"]) if (float(x) > 0 if x_log else True) and (float(y) > 0 if y_log else True)]
        points = " ".join(f"{x:.2f},{y:.2f}" for x,y in coords)
        color = item.get("color", COLORS.get(item["label"], "#7c3aed"))
        parts.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2.4" stroke-linejoin="round"/>')
        for x,y in coords:
            parts.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="3.2" fill="{color}"/>')
    if benchmark is not None and (not y_log or benchmark > 0):
        yy=py(float(benchmark)); parts.append(f'<line x1="{left}" y1="{yy:.2f}" x2="{right}" y2="{yy:.2f}" stroke="#111827" stroke-dasharray="6 5" stroke-width="1.5"/>')
    total = len(series)
    legend_y = 466
    start_x = max(135, (WIDTH - total*205)/2)
    for i,item in enumerate(series):
        xx = start_x + i*205
        color = item.get("color", COLORS.get(item["label"], "#7c3aed"))
        parts.extend([f'<line x1="{xx:.2f}" y1="{legend_y}" x2="{xx+24:.2f}" y2="{legend_y}" stroke="{color}" stroke-width="2.4"/>', _text(xx+31, legend_y+4, item["label"].replace("_", " "), 11, anchor="start")])
    _save(path, parts)


def boxplot_chart(path, groups, title, ylabel, benchmark=None):
    left,right,top,bottom=105,815,82,385
    allvals=np.concatenate([np.asarray(g["values"],dtype=float) for g in groups])
    lo=float(allvals.min()); hi=float(allvals.max())
    if benchmark is not None: lo=min(lo,benchmark); hi=max(hi,benchmark)
    span=max(hi-lo,1e-9); lo-=.1*span; hi+=.12*span
    py=lambda v: bottom-(v-lo)/(hi-lo)*(bottom-top)
    parts=_start(title)
    for i in range(5):
        val=lo+i*(hi-lo)/4; yy=py(val)
        parts.extend([f'<line x1="{left}" y1="{yy:.2f}" x2="{right}" y2="{yy:.2f}" stroke="#e5e7eb"/>',_text(left-12,yy+4,f"{val:.3f}",11,anchor="end")])
    if benchmark is not None:
        yy=py(benchmark); parts.append(f'<line x1="{left}" y1="{yy:.2f}" x2="{right}" y2="{yy:.2f}" stroke="#111827" stroke-dasharray="6 5" stroke-width="1.4"/>')
    n=len(groups); spacing=(right-left)/(n+1)
    for i,g in enumerate(groups,1):
        vals=np.asarray(g["values"],dtype=float); q05,q25,med,q75,q95=np.quantile(vals,[.05,.25,.5,.75,.95]); mean=float(vals.mean()); x=left+i*spacing; color=g.get("color",COLORS.get(g["label"],"#7c43ed"))
        parts.extend([f'<line x1="{x:.2f}" y1="{py(q05):.2f}" x2="{x:.2f}" y2="{py(q95):.2f}" stroke="{color}" stroke-width="2"/>',f'<line x1="{x-9:.2f}" y1="{py(q05):.2f}" x2="{x+9:.2f}" y2="{py(q05):.2f}" stroke="{color}" stroke-width="2"/>',f'<line x1="{x-9:.2f}" y1="{py(q95):.2f}" x2="{x+9:.2f}" y2="{py(q95):.2f}" stroke="{color}" stroke-width="2"/>',f'<rect x="{x-29:.2f}" y="{py(q75):.2f}" width="58" height="{max(1,py(q25)-py(q75)):.2f}" fill="{color}" fill-opacity="0.18" stroke="{color}" stroke-width="1.8"/>',f'<line x1="{x-29:.2f}" y1="{py(med):.2f}" x2="{x+29:.2f}" y2="{py(med):.2f}" stroke="{color}" stroke-width="2.4"/>',f'<circle cx="{x:.2f}" cy="{py(mean):.2f}" r="3.5" fill="{color}"/>',_text(x,bottom+24,g["label"].replace("_"," "),11)])
    parts.extend([_text(25,(top+bottom)/2,ylabel,13,rotate=-90),_text((left+right)/2,436,"Estimator",13)])
    _save(path,parts)


def scatter_chart(path, x_values, y_values, color_values, title, xlabel, ylabel, color_label):
    left,right,top,bottom=100,795,82,380
    xv=np.asarray(x_values,float); yv=np.asarray(y_values,float); cv=np.asarray(color_values,float)
    xmin,xmax=float(xv.min()),float(xv.max()); ymin,ymax=float(yv.min()),float(yv.max())
    dx=max(xmax-xmin,1e-9); dy=max(ymax-ymin,1e-9)
    px=lambda v:left+(v-(xmin-.05*dx))/(1.1*dx)*(right-left)
    py=lambda v:bottom-(v-(ymin-.05*dy))/(1.1*dy)*(bottom-top)
    parts=_start(title,"Color encodes implied volatility; each marker is a quote")
    for i in range(5):
        vx=xmin+i*dx/4; xx=px(vx); vy=ymin+i*dy/4; yy=py(vy)
        parts.extend([f'<line x1="{xx:.2f}" y1="{top}" x2="{xx:.2f}" y2="{bottom}" stroke="#f3f4f6"/>',_text(xx,bottom+21,f"{vx:.0f}",11),f'<line x1="{left}" y1="{yy:.2f}" x2="{right}" y2="{yy:.2f}" stroke="#e5e7eb"/>',_text(left-12,yy+4,f"{vy:.2g}",11,anchor="end")])
    cmin,cmax=float(cv.min()),float(cv.max()); span=max(cmax-cmin,1e-12)
    for x,y,v in zip(xv,yv,cv):
        hue=230-230*(v-cmin)/span
        parts.append(f'<circle cx="{px(x):.2f}" cy="{py(y):.2f}" r="7" fill="hsl({hue:.1f},68%,48%)" stroke="#fff" stroke-width="1.3"><title>Strike {x:g}; maturity {y:g}; {color_label} {v:.4f}</title></circle>')
    parts.extend([f'<line x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" stroke="#9ca3af"/>',f'<line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" stroke="#9ca3af"/>',_text((left+right)/2,432,xlabel,13),_text(25,(top+bottom)/2,ylabel,13,rotate=-90),_text(right-40,416,f"{cmin*100:.1f}% to {cmax*100:.1f}%",11)])
    _save(path,parts)


def range_chart(path, labels, lows, medians, highs, title, ylabel):
    left,right,top,bottom=105,815,82,365
    lo=min(float(np.min(lows)),0.0); hi=max(float(np.max(highs)),0.0); span=max(hi-lo,1e-9); lo-=.08*span; hi+=.08*span
    py=lambda v:bottom-(v-lo)/(hi-lo)*(bottom-top)
    parts=_start(title,"Bars show the 5th–95th percentile; point marks the median")
    for i in range(5):
        v=lo+i*(hi-lo)/4; yy=py(v)
        parts.extend([f'<line x1="{left}" y1="{yy:.2f}" x2="{right}" y2="{yy:.2f}" stroke="#e5e7eb"/>',_text(left-12,yy+4,f"{v:.2f}",11,anchor="end")])
    parts.append(f'<line x1="{left}" y1="{py(0):.2f}" x2="{right}" y2="{py(0):.2f}" stroke="#111827" stroke-dasharray="6 5"/>')
    spacing=(right-left)/(len(labels)+1)
    for i,(label,low,med,high) in enumerate(zip(labels,lows,medians,highs),1):
        x=left+i*spacing; color=["#2563eb","#dc2626","#059669"][((i-1)//2)%3]
        parts.extend([f'<line x1="{x:.2f}" y1="{py(low):.2f}" x2="{x:.2f}" y2="{py(high):.2f}" stroke="{color}" stroke-width="4"/>',f'<line x1="{x-7:.2f}" y1="{py(low):.2f}" x2="{x+7:.2f}" y2="{py(low):.2f}" stroke="{color}" stroke-width="2"/>',f'<line x1="{x-7:.2f}" y1="{py(high):.2f}" x2="{x+7:.2f}" y2="{py(high):.2f}" stroke="{color}" stroke-width="2"/>',f'<circle cx="{x:.2f}" cy="{py(med):.2f}" r="4.5" fill="{color}"/>',_text(x,bottom+20,label,10)])
    parts.extend([_text(25,(top+bottom)/2,ylabel,13,rotate=-90),_text((left+right)/2,438,"Rehedging intervals / proportional transaction cost",13)])
    _save(path,parts)
