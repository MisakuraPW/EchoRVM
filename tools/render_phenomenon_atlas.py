"""Render an offline HTML atlas from sealed measurements; never reads labels."""

from __future__ import annotations

import argparse
import base64
from collections import defaultdict
import csv
from html import escape
import io
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


COLORS = ("#007f76", "#bc562f", "#365a99", "#77633e")
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False,
                     "axes.prop_cycle": plt.cycler(color=COLORS), "font.size": 10,
                     "figure.dpi": 120, "svg.hashsalt": "phenomenon-atlas-v1"})


def per_case(rows, value):
    groups = defaultdict(list)
    for row in rows:
        v = value(row)
        if v is not None and np.isfinite(v):
            groups[row["case"]].append(v)
    return np.array([np.mean(v) for v in groups.values()])


def trend(ax, x, samples, label, **kwargs):
    used = [(a, s) for a, s in zip(x, samples) if len(s)]
    if not used:
        return
    x, s = zip(*used)
    y = np.array([np.median(z) for z in s])
    q = np.array([np.quantile(z, [.25, .75]) for z in s])
    result = ax.plot(x, y, "o-", label=label, **kwargs)
    ax.fill_between(x, q[:, 0], q[:, 1], color=result[0].get_color(), alpha=.13)


def finish(fig):
    stream = io.BytesIO()
    fig.savefig(stream, format="svg", bbox_inches="tight", metadata={"Date": None})
    plt.close(fig)
    return '<img alt="Quantitative plot" src="data:image/svg+xml;base64,' + base64.b64encode(stream.getvalue()).decode() + '">'


def figure(columns=2):
    return plt.subplots(1, columns, figsize=(4.65*columns, 3.1), layout="constrained", squeeze=False)


def timing(data):
    rows = [r for r in data["metadata"] if r["gap_frames"] > 0]
    if not rows:
        return None, "无可核验的顺向端点。"
    fig, (axes,) = figure(3)
    for ax, k, unit in zip(axes, ("gap_frames", "gap_ms"), ("native frames", "ms")):
        z = [r[k] for r in rows]
        ax.hist(z, bins=30, color=COLORS[0])
        ax.set(xlabel=f"ED to ES ({unit})", ylabel="Patients")
        if k == "gap_frames":
            ax.axvline(16, color=COLORS[1], linestyle="--", label="16 frames")
            ax.legend()
    valid = [r for r in rows if r["same"] is not None]
    if valid:
        both, same = [np.mean([r[k] for r in valid]) for k in ("both", "same")]
        axes[2].bar(["Same clip", "Across clips", "Not both"], [same, both-same, 1-both])
        axes[2].set(ylabel="Mean fraction of valid W64 starts", ylim=(0, 1))
    return fig, f"顺向端点 {len(rows):,} 例；实际间隔中位数 {np.median([r['gap_frames'] for r in rows]):.0f} 帧 / {np.median([r['gap_ms'] for r in rows]):.0f} ms。ED/ES 由两张描记的较大／较小面积指代；非顺向配对排除，不推断搏次。"


def geometry(data):
    rows = [r for r in data["dense_pairs"] if r["lag"] == 1]
    if not rows:
        return None, "缺少可信的全周期图像与参考轮廓，不能用 EchoNet 两个端点冒充邻帧。"
    fig, (axes,) = figure()
    counts = defaultdict(int)
    for r in rows:
        counts[r["case"]] += 1
    h = axes[0].hexbin([r["image"] for r in rows], [r["geometry"] for r in rows],
                       C=[1/counts[r["case"]] for r in rows], reduce_C_function=np.sum,
                       mincnt=0, gridsize=28, cmap="viridis")
    fig.colorbar(h, ax=axes[0], label="Patient-balanced mass")
    axes[0].set(xlabel="Adjacent-frame sector RMSE", ylabel="Contour disagreement (1-IoU)")
    for phase, label in ((False, "Before area minimum"), (True, "After area minimum")):
        lags = sorted({r["lag"] for r in data["dense_pairs"]})
        trend(axes[1], lags, [per_case([r for r in data["dense_pairs"] if r["lag"] == k
                                       and (r["phase"] >= r["es_phase"]) == phase],
                                      lambda r: 100*r["floor"]) for k in lags], label)
    axes[1].legend(fontsize=8)
    axes[1].set(xlabel="Frame separation", ylabel="Minimum shared-mask error (% pixels)")
    return fig, (f"{len(counts)} 例，{len(rows):,} 个相邻帧对。Hamming 下界仅约束两帧共用同一掩码，"
                 "不是 ViT、tubelet 或 Dice 的下界；TED 中间参考含插值和人工复核。")


def function(data):
    rows = [r for r in data["metadata"] if r["ef"] is not None and r["ed_area"] > 0]
    if not rows:
        return None, "缺少 EF 或端点描记。"
    fig, (axes,) = figure(3 if data["curves"] else 2)
    for a, key, label in zip(axes[:2], ("ed_area", "es_area"), ("ED area", "ES area")):
        h = a.hexbin([r[key] for r in rows], [r["ef"] for r in rows], gridsize=30, mincnt=1, cmap="viridis")
        a.set(xlabel=f"{label} (pixels)", ylabel="Annotated EF (%)")
        fig.colorbar(h, ax=a, label="Patients per bin")
    if data["curves"]:
        for r in data["curves"]:
            a = np.array(r["area"])
            axes[2].plot(1000*np.arange(len(a))/r["fps"], a/a.max(), alpha=.55)
        axes[2].set(xlabel="Time from cycle start (ms)", ylabel="TED area / cycle maximum")
    return fig, (f"EchoNet {len(rows):,} 例。看相近端点面积下 EF 的条件分布；像素面积不等于容积，不能由此断言某项表征充分。"
                 "右图（如有）是独立画法的 TED 参考面积轨迹，不是 EchoNet 同一批病例。")


def budgets(data):
    rows = data["budget"]
    if not rows:
        return None, "缺少全周期参考，无法计算重建轨迹与预算。"
    fig, (axes,) = figure()
    for ax, task in zip(axes, ("sampling", "partition")):
        tags = sorted({(r["slots"], r["strategy"]) for r in rows if r["task"] == task})
        if not tags:
            ax.set(title=task.capitalize())
            continue
        ax.boxplot([100*np.array([r["mae"] for r in rows if r["task"] == task and
                                 (r["slots"], r["strategy"]) == tag]) for tag in tags],
                   showfliers=True)
        ax.set_xticks(range(1, len(tags)+1), [f"{n}: {s.replace('offline_', '')}" for n, s in tags],
                      rotation=20, ha="right")
        ax.set(title=task.capitalize(), ylabel="Area-trajectory MAE (% of image)")
    return fig, "同一批周期；等帧数的离线选点与同帧等摘要的切段是两种预算。图像运动选点允许看全窗，不能作为在线性能；切分用等容量标量摘要，不等于 A-R 的性能。"


def past_reference(data):
    rows = data["references"]
    if not rows:
        return None, "缺全周期参考，不能核验历史轮廓是否过时。"
    fig, ((ax,),) = figure(1)
    lags = sorted({r["lag"] for r in rows})
    midpoint = np.median([r["contrast"] for r in rows])
    for strong, label in ((False, "Lower current contrast"), (True, "Higher current contrast")):
        trend(ax, lags, [per_case([r for r in rows if r["lag"] == k and (r["contrast"] >= midpoint) == strong],
                                 lambda r: r["geometry"]) for k in lags], label)
    ax.set(xlabel="Age of borrowed mask (frames)", ylabel="Mismatch to current outline (1-IoU)")
    ax.legend(fontsize=8)
    return fig, "直接借用过去参考掩码是特权几何参照；当前对比度和位移共变，不说明状态因果，也不等于经过配准的历史上界。"


def dense_example(data):
    rows = [r for r in data["curves"] if "example" in r]
    if not rows:
        return ""
    r = rows[0]["example"]
    fig, (axes,) = figure(2)
    for i, ax in enumerate(axes):
        ax.imshow(r["frames"][i], cmap="gray", vmin=0, vmax=255)
        ax.contour(r["outlines"][i], levels=[.5], colors=["#e6b845"], linewidths=.7)
        ax.set_title(f"Frame t{'+1' if i else ''}")
        ax.axis("off")
    return (finish(fig) + f"<p>示例规则：每病例外观距离最低四分位内取轮廓差异最大帧对，"
            f"展示病例间该差异的中位附近。图像 RMSE {r['image']:.3f}，轮廓 1−IoU {r['geometry']:.3f}。"
            "TED：Painchaud 等，IEEE TMI 2022，CC BY-NC-SA 4.0；缩放至 112×112。</p>")


def masks(data):
    rows = [r for r in data["history"] if all(r[k] is not None for k in ("now", "past", "stale"))]
    if not rows:
        return None, "未提取图像或无合法的历史对照。"
    has_delta = any(r.get("past_delta") is not None for r in rows)
    fig, plots = plt.subplots(2 if has_delta else 1, 3, figsize=(13.95, 5.8 if has_delta else 3.1),
                              layout="constrained", squeeze=False)
    for col, region in enumerate(("edge", "cavity", "remainder")):
        ax = plots[0, col]
        a = [r for r in rows if r["region"] == region]
        for key, label in (("now", "Current visible"), ("past", "Recent copy"), ("stale", "Older copy")):
            trend(ax, (1, 2, 4, 8, 16), [per_case([r for r in a if r["lag"] == l], lambda r: r[key])
                                         for l in (1, 2, 4, 8, 16)], label)
        ax.set(title=region, xlabel="Recent reference lag (frames)", ylabel="Masked-pixel RMSE")
        if has_delta:
            ax = plots[1, col]
            for key, label in (("now_delta", "Current visible"), ("past_delta", "Recent copy")):
                trend(ax, (1, 2, 4, 8, 16), [per_case([r for r in a if r["lag"] == l], lambda r: r.get(key))
                                             for l in (1, 2, 4, 8, 16)], label)
            ax.set(xlabel="Recent reference lag (frames)", ylabel="Change RMSE (consecutive frames)")
    plots[0, 0].legend(fontsize=7)
    return fig, ("像素近邻填补参照，不使用 MAE。历史帧充分可见，较旧参照年龄为横轴两倍；"
                 "这不是年龄匹配的相位随机对照。标注决定统计区域，不参与预测。若有下排，纵轴是相同遮挡下两次重建的帧差误差。")


def spatial(data):
    rows = data["spatial"]
    if not rows:
        return None, "未提取可对齐的端点影像。"
    fig, (axes,) = figure()
    for phase in ("ED", "ES"):
        trend(axes[0], (1, 4, 8, 16),
              [per_case([r for r in rows if r["phase"] == phase and r["patch"] == p], lambda r: r["geometry"])
               for p in (1, 4, 8, 16)], phase)
    axes[0].set(xlabel="One-label grid size (pixels)", ylabel="Disagreement to GT (1-IoU)")
    axes[0].legend()
    axes[1].hist([[r["cnr"] for r in rows if r["phase"] == phase and r["patch"] == 1]
                  for phase in ("ED", "ES")], bins=20, histtype="step", label=["ED", "ES"])
    axes[1].set(xlabel="Contour-band contrast proxy", ylabel="Patients")
    axes[1].legend()
    return fig, "粗格只允许一个二值标签，因此不代表能保留子 patch 几何的 ViT token；对比度仅是已知轮廓两侧的图像代理。"


def scales(data):
    rows = data["scale"]
    if not rows:
        return None, "未提取端点和历史影像。"
    fig, (axes,) = figure(3)
    for ax, reg in zip(axes, ("edge", "cavity", "remainder")):
        means = [[per_case([r for r in rows if r["region"] == reg and r["lag"] == l and r["sigma"] == s],
                           lambda r: r["error"]) for l in (1, 2, 4, 8, 16)] for s in (1, 2, 4)]
        for s, values in zip((1, 2, 4), means):
            trend(ax, (1, 2, 4, 8, 16), values, f"Gaussian scale {s}px")
        ax.set(title=reg, xlabel="History lag (frames)", ylabel="High-pass change RMSE")
    axes[0].legend(fontsize=7)
    return fig, "同一坐标的高通残差，不是组织追踪或 RF；不同尺度是嵌套高通而非正交子带。光有能量差不证明应加频域分支。"


def perturbations(data):
    rows = data["perturb"]
    if not rows:
        return None, "缺少端点影像扰动结果。"
    fig, (axes,) = figure()
    for strength in (0, .25, .5):
        g = [r for r in rows if r["strength"] == strength]
        for ax, k in zip(axes, ("interior_error", "edge_error")):
            trend(ax, (0, .7, 1.4, 2.1), [per_case([r for r in g if r["sigma"] == s], lambda r: r[k])
                                          for s in (0, .7, 1.4, 2.1)], f"Strength {strength}")
            ax.set(xlabel="Gaussian smoothing sigma (px)", ylabel=k.replace("_", " ") + " RMSE")
    axes[0].legend(fontsize=8)
    return fig, "同帧合成局部衰减与加性噪声，参考是未施加扰动的原 B-mode 帧，不是无噪声真值；曲线衡量外观误差而非真实分割性能。"


def prefixes(data):
    rows = [r for r in data["metadata"] if r["gap_frames"] > 0 and r["frames"] >= 64]
    if not rows:
        return None, "没有可用的顺向端点长视频。"
    fig, (axes,) = figure()
    times = (16, 32, 48, 64, 80, 96, 112, 128)
    for k, label in (("ed", "ED has arrived"), ("es", "Both annotations have arrived")):
        axes[0].plot(times, [np.mean([r[k] < t for r in rows]) for t in times], "o-", label=label)
    axes[0].legend(fontsize=8)
    axes[0].set(xlabel="Frames arrived from video start", ylabel="Fraction of patients", ylim=(0, 1.03))
    axes[1].hist([1000*(r["es"]+1)/r["fps"] for r in rows], bins=25, color=COLORS[0])
    axes[1].set(xlabel="Time until both annotations (ms)", ylabel="Patients")
    return fig, f"{len(rows):,} 例有至少 64 帧且描记顺向。这里只测索引覆盖，绝非模型置信度、临床充分性或无标注窗口没有收缩。"


PLOTS = (
    ("T1", "帧数与实际时长", timing), ("T2/T3", "相似外观与共享帧输出的几何代价", geometry),
    ("C1", "端点几何和功能", function), ("T4", "观察与摘要预算", budgets),
    ("H1", "历史轮廓何时过时", past_reference), ("H2", "有历史时的掩码条件", masks),
    ("S1", "扇区轮廓与空间预算", spatial), ("F1", "空间尺度与历史间隔", scales),
    ("S2", "抗扰动与边界保留", perturbations), ("H3", "到达的帧与标注证据", prefixes),
)


def readings(d):
    out = []
    r = [r for r in d["metadata"] if r["gap_frames"] > 0]
    if r:
        fraction = np.mean([row["gap_frames"] >= 16 for row in r])
        out.append(f"顺向端点里 {fraction:.1%} 的端点索引差至少为 16 帧，单个连续 16 帧不能同时容纳它们。"
                   "这说明观测范围有差异，不证明变长比跨段保留更好。")
    z = [r for r in d["dense_pairs"] if r["lag"] == 1]
    if z:
        floor = 100*per_case(z, lambda r: r["floor"])
        iou = per_case(z, lambda r: r["geometry"])
        out.append(f"TED 小面板：相邻帧的病例均值再取中位数，共享二值掩码的全图 Hamming 下界为 "
                   f"{np.median(floor):.3f}% 像素，参考轮廓自身的 1−IoU 为 {np.median(iou):.3f}。"
                   "不可把插值复核参考的细微变化当作独立逐帧人工真值。")
    h = [r for r in d["history"] if r["lag"] == 1 and r["past"] is not None and r["now"] is not None]
    gains = [per_case([r for r in h if r["region"] == k], lambda r: r["now"]-r["past"])
             for k in ("edge", "cavity", "remainder")]
    if all(len(z) for z in gains):
        out.append("最近充分可见图像拷贝相对只用当前可见像素的 RMSE 降低量（0–1 灰度，病例中位）："
                   + "、".join(f"{k} {np.median(v):+.3f}" for k, v in
                               zip(("边界带", "腔内", "扇区其他"), gains))
                   + "。该结果只约束简单恢复参照，不证明 SSL 使用了捷径。")
    for task in ("sampling", "partition"):
        x = [r for r in d["budget"] if r["task"] == task and r["slots"] == 4]
        if x:
            out.append(("等样本数采样" if task == "sampling" else "等摘要容量切段") + "的轨迹 MAE 中位数"
                       "（占画布面积百分点）："
                       + "、".join(f"{s} {100*np.median([r['mae'] for r in x if r['strategy']==s]):.3f}"
                                   for s in sorted({r["strategy"] for r in x}))
                       + "。不以探索面板的最小值选算法。")
    return "<ul>" + "".join("<li>" + escape(s) + "</li>" for s in out) + "</ul>"

LITERATURE = (
    ("OSA 图 2", "https://openaccess.thecvf.com/content/CVPR2026/papers/Wang_OSA_Echocardiography_Video_Segmentation_via_Orthogonalized_State_Update_and_Anatomical_CVPR_2026_paper.pdf#page=2", "散斑、弱轮廓与周期内形变；不重跑样例。"),
    ("DISCOVR 图 1", "https://openreview.net/pdf/22fb0560a2b0620b3207e4db56c7641f08fb9081.pdf", "作者模型下的时序相似现象，不等于几何相同。"),
    ("EchoFM 图 5", "https://pmc.ncbi.nlm.nih.gov/articles/PMC12616925/", "表征的周期相似度，不是模型无关规律。"),
    ("LRM-Functa 图 1B", "https://arxiv.org/html/2603.25951v1", "作者潜变量的周期性／低秩线索。"),
    ("EchoCardMAE 图 2", "https://papers.miccai.org/miccai-2025/paper/2462_paper.pdf#page=8", "移动窗口起点，不等于保持原帧而移动内部切分。"),
    ("TED 图 4/5", "https://www.creatis.insa-lyon.fr/~bernard/publis/tmi_2022_nathan.pdf", "轮廓轨迹与时序一致性，以及全周期参考的人工锚点和插值复核。"),
    ("VideoMAE 图 2", "https://papers.neurips.cc/paper_files/paper/2022/file/416f9cb3276121c42eebb86352a4354a-Paper-Conference.pdf#page=4", "同位置复制的既有动机；本次限定到递归条件下的可见路径。"),
)


def render(data, prior, prior_rows=(), references=()):
    sections = []
    for code, title, fn in PLOTS:
        fig, desc = fn(data)
        rendered = finish(fig) if fig is not None else '<p class="missing">未运行</p>'
        if code == "T2/T3":
            rendered += dense_example(data)
        sections.append(f'<section><h2>{code} · {title}</h2>{rendered}<p>{escape(desc)}</p></section>')
    notes = "".join(f'<li><a href="{escape(u)}">{escape(title)}</a>：{escape(note)}</li>'
                    for title, u, note in LITERATURE)
    for title, path in references:
        encoded = base64.b64encode(Path(path).read_bytes()).decode()
        notes += (f'<li><details><summary>{escape(title)} · 用户提供的文献截图</summary>'
                  f'<img alt="{escape(title)}" src="data:image/png;base64,{encoded}"></details></li>')
    previous = ""
    for title, path in prior:
        image = base64.b64encode(Path(path).read_bytes()).decode()
        previous += f'<details><summary>{escape(title)}</summary><img alt="{escape(title)}" src="data:image/png;base64,{image}"></details>'
    keys = ("stage", "arm", "mode", "split", "task", "patients", "mae_pp", "dice_patient_pct", "dice_global_pct")
    if prior_rows:
        def cell(row, k):
            value = row.get(k, "")
            if value and k in keys[-3:]:
                value = f"{float(value):.3f}"
            return "<td>" + escape(value) + "</td>"
        previous = ("<details><summary>已完成对照：按任务、评估模式和划分查看</summary>"
                    "<div class='table'><table><thead><tr>" + "".join(f"<th>{k}</th>" for k in keys)
                    + "</tr></thead><tbody>" + "".join("<tr>" + "".join(cell(r, k) for k in keys) + "</tr>"
                                                       for r in prior_rows) + "</tbody></table></div></details>" + previous)
    return f"""<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>心超 MAE｜前置现象图谱</title>
<style>
body{{margin:auto;max-width:1180px;padding:26px;color:#182620;background:#fafbf9;font:16px/1.65 system-ui,sans-serif}}
section{{padding:16px 0;border-top:1px solid #cdd7d2}} h1,h2,h3{{line-height:1.25}}
img{{display:block;max-width:100%;height:auto}} a{{color:#00695f}} .missing{{color:#8e492d;font-weight:600}}
th,td{{text-align:left;padding:5px 10px;border-bottom:1px solid #d7dfdb}} .table{{overflow:auto}}
summary{{cursor:pointer;padding:5px 0}}
</style>
<h1>心超 MAE：哪些观测可以合并？</h1>
<p>开发集探索。图按物理时间、几何与功能、历史与学习、空间与外观组织。未读封存 TEST 标签，不训练模型。
趋势线通常为患者均值的中位数，色带为患者间四分位数，不是置信区间。</p>
<p>EchoNet 开发索引：{data['counts']['development_rows']:,}；有效双端点：{data['counts']['with_endpoint_pair']:,}；
图像分析 VAL：{data['counts']['image_cases']:,}；TED 开发参考：{data['dense_cases']:,}。</p>
<section><h2>外部现象和已知结果</h2><ul>{notes}</ul></section>
<section><h2>本次读数与解释边界</h2>{readings(data)}</section>
{''.join(sections)}
<section><h2>本项目旧实验，不当作新测量</h2>
<p>固定 L16 是现用对照。Stage B 的 B1 在同一权重和 VAL 窗口上由随机切分改为固定切分，
EF MAE 4.055 → 3.923 pp；不是全新训练对照。R-M 的特定非对称 mask 未带来实用双任务增益，
旧的频域先导有有限正向信号但普通读出与状态读出方向不同。</p>{previous}</section>
<section><h2>读取与复查</h2><p>原始聚合测量：<a href="measurements.json">measurements.json</a>。
TED 和 CAMUS 患者重合；受控恢复或扰动不等于临床结果，跨图不要混用人口和单位。</p></section></html>"""


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("result", type=Path)
    p.add_argument("--prior", nargs=2, action="append", default=[], metavar=("TITLE", "PNG"))
    p.add_argument("--prior-table", type=Path, help="Prior aggregate metrics CSV, not a label input")
    p.add_argument("--reference-image", nargs=2, action="append", default=[], metavar=("TITLE", "PNG"))
    a = p.parse_args()
    path = a.result / "measurements.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema") != 1:
        raise ValueError("Unsupported measurement schema")
    rows = []
    if a.prior_table:
        with a.prior_table.open(encoding="utf-8-sig", newline="") as fp:
            rows = list(csv.DictReader(fp))
    html = render(data, a.prior, rows, a.reference_image)
    (a.result / "report.html").write_text(html, encoding="utf-8")
    print(f"Saved {a.result / 'report.html'}")


if __name__ == "__main__":
    main()
