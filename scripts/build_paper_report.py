"""Render manuscript-style figures and a review page from independent outputs."""

import argparse
import hashlib
import html
import json
import shutil
import sys
from pathlib import Path

import numpy as np

from uav_mfg.paper_plots import CASES, density_load_comparison, make_paper_figures
from uav_mfg.plots import load
from uav_mfg.problem import Problem
from uav_mfg.transport import Grid, Transport


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def copy_reference_paper(source, output):
    """Original manuscript comparisons are an explicit optional, private input."""
    if source is None:
        return {"enabled": False, "manuscript_sha256": None, "figure_sha256": {}}
    names = ("fig2", "fig3", "fig4", "fig5", "height_slices")
    inputs = [source / "main.tex", *(source / "figures" / f"{name}.png" for name in names)]
    for path in inputs:
        if not path.is_file():
            raise FileNotFoundError(path)
    destination = output / "reference"
    destination.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for name in names:
        path = source / "figures" / f"{name}.png"
        shutil.copyfile(path, destination / path.name)
        hashes[name] = digest(path)
    return {"enabled": True, "manuscript_sha256": digest(source / "main.tex"), "figure_sha256": hashes}


def reference_section(comparison, names, label):
    if not comparison["enabled"]:
        return ""
    figures = ''.join(f'<figure><img src="reference/{name}.png" alt="论文原图">'
                      '<figcaption>论文原图，非本次计算输出。</figcaption></figure>' for name in names)
    return f'<details><summary>{label}，仅用于外观对照</summary><div class="pair">{figures}</div></details>'


def audit_case(root, name):
    config, data = load(root, name)
    metadata = json.loads((root / name / "metadata.json").read_text())
    saved = json.loads((root / name / "diagnostics.json").read_text())
    grid = Grid(Problem(config), data["density"].shape)
    actual = Transport(grid, data["velocity"]).audit(data["density"])
    differences = {key: abs(actual[key] - saved[key]) for key in actual if actual[key] is not None}
    finite = all(np.isfinite(data[key]).all() for key in ("phi", "density", "velocity"))
    valid = (finite and data["density"].min() >= 0 and data["phi"][grid.target].max() == 0
             and data["density"][grid.target | grid.obstacle].max() == 0
             and max(differences.values()) < 1e-10
             and actual["relative_steady_balance"] < config["transport"]["balance_tolerance"]
             and actual["pde_relative_residual"] < config["coupling"]["continuity_tolerance"]
             and actual["outer_leakage"] == 0 and actual["obstacle_leakage"] == 0
             and saved["total_projection_mass_removed"] == 0
             and metadata["status"] == "converged")
    return {"case": name, "directory": str(root / name), "status": metadata["status"],
            "passed": bool(valid), "fresh_flux_audit": actual, "saved_audit_max_difference": max(differences.values()),
            "eikonal_mean": saved["eikonal_mean"], "eikonal_p95": saved["eikonal_p95"],
            "source_strength": config["source"]["rate"], "spatial_dimension": config.get("spatial_dimension", 3),
            "grid": list(grid.shape), "fields_sha256": digest(root / name / "fields.npz")}


def render(output, details, audits, planar_root, three_d_root):
    rows = []
    for item in audits:
        flux = item["fresh_flux_audit"]
        rows.append(f'<tr><td>{item["spatial_dimension"]}D {html.escape(item["case"])}</td><td>{"通过" if item["passed"] else "未通过"}</td>'
                    f'<td>{flux["absorption_over_source"]:.6f}</td><td>{flux["relative_steady_balance"]:.2e}</td>'
                    f'<td>{flux["pde_relative_residual"]:.2e}</td><td>{item["eikonal_mean"]:.3g}</td>'
                    f'<td>{flux["rho_max"]:.4f}</td></tr>')
    peak, source_peak = details["density_peak"], details["density_source_region_peak"]
    paths = details["source_seeded_streamlines"]
    absorbed = sum(path["reached_target"] for path in paths)
    blocks = sum(path["stopped_at_obstacle"] for path in paths)
    run_record = "selection.json" if (planar_root / "selection.json").exists() else "command.json"
    load_section = ""
    if details.get("load_comparison"):
        comparison_rows = []
        for item in details["load_comparison"]["cases"]:
            comparison_rows.append(f'<tr><td>{item["dimension"]}D {html.escape(item["case"])}</td>'
                                   f'<td>{item["low_source_rate"]:g} → {item["high_source_rate"]:g}</td>'
                                   f'<td>{item["low_density_peak"]:.4f} → {item["high_density_peak"]:.4f}</td>'
                                   f'<td>{item["density_peak_ratio"]:.2f} 倍</td><td>{item["mean_density_ratio"]:.2f} 倍</td></tr>')
        load_section = ('<h2>真实密度提高：相同色标的前后对照</h2><figure class="wide">'
                        '<img src="figures/density_load_comparison.png" alt="相同线性色标下的低负载与高负载密度">'
                        '<figcaption>左侧此前独立计算，右侧增加源项后重新求解；两侧都用线性 0–0.05。'
                        '超过 0.05 的密度以黄色和色条顶端箭头标示，未裁剪原始密度。</figcaption>'
                        '<div class="downloads"><a href="figures/density_load_comparison.png">PNG</a>'
                        '<a href="figures/density_load_comparison.pdf">PDF</a></div></figure>'
                        '<div class="scroll"><table><thead><tr><th>场景</th><th>源强</th><th>真实密度峰值</th>'
                        '<th>峰值提高</th><th>同一自由域平均密度提高</th></tr></thead><tbody>'
                        + ''.join(comparison_rows) + '</tbody></table></div>'
                        '<p><a href="../paper_style_low_load/report.html">之前的低负载完整结果</a> · '
                        '<a href="density_load_comparison.json">密度前后统计</a></p>')
    text = '''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>ESWA · 论文版式的独立计算结果</title>
<style>
*{box-sizing:border-box}body{margin:0;background:#eef1f2;color:#24333a;font:16px/1.7 system-ui,"Microsoft YaHei",sans-serif}
main{max-width:1390px;margin:auto;padding:28px 28px 60px}h1{font-size:30px;line-height:1.35;margin:6px 0 12px}
h2{font-size:22px;margin:30px 0 12px}p{max-width:1100px}a{color:#087e8b}small,.muted{color:#5d6f78}
.label{font-size:13px;color:#087e8b;letter-spacing:.6px;font-weight:650}.controls{display:flex;gap:8px;align-items:center;flex-wrap:wrap}
button{border:1px solid #afc0c6;border-radius:5px;background:white;color:#334a52;cursor:pointer;padding:9px 12px;font:inherit;font-size:14px}
button.active{background:#176e73;border-color:#176e73;color:white}.pair{display:grid;grid-template-columns:1fr 1fr;gap:18px}
figure{margin:16px 0;background:white;border:1px solid #d7e1e4;border-radius:7px;padding:15px 20px}
figure img{display:block;width:100%;height:auto}figcaption{font-size:15px;margin-top:10px}figcaption strong{font-weight:650}
.downloads{font-size:13px;display:flex;gap:14px;margin-top:7px}.note{background:#e0edeb;border-left:4px solid #21918c;padding:12px 16px}
.scroll{overflow:auto}table{border-collapse:collapse;width:100%;font-size:14px;background:white}th,td{padding:11px 14px;text-align:left;white-space:nowrap;border-bottom:1px solid #dbe4e7}th{background:#e3eaed}
details{margin:18px 0}summary{cursor:pointer;color:#176e73}details figure{max-width:1100px;margin:15px auto}.wide{max-width:1260px;margin:16px auto}
@media(max-width:800px){main{padding:20px 12px}.pair{grid-template-columns:1fr}h1{font-size:25px}figure{padding:10px}}
</style></head><body><main>
<div class="label">EXPERT SYSTEMS WITH APPLICATIONS · ESWA-D-26-05581R1</div>
<h1>按论文版式重新出图</h1>
<p>六宫格价值场、源区流线、密度等值图和多高度切片，沿用论文的 viridis 配色、细白色等值线、独立色条及几何图例。二维主图：100×100 网格，z=5 m，单位厚度层；三维切片来自独立的 40×40×16 计算。</p>
<p class="note">当前 P2P 二维源强 q=__Q__，三维源强 q=__THREE_D_Q__，密度上限 __BOUND__，均为公开的重建选择。默认密度色标采用论文的<strong>线性 0–0.05</strong>，也可切换完整数据范围和平方根显示。__SATURATION__</p>
__LOAD_SECTION__
<h2>论文图 3 / 图 4：诱导流线和密度带</h2>
<div class="controls"><span>密度色标</span><button class="active" data-scale="paper">论文线性：0–0.05</button>
<button data-scale="relative">当前范围：线性 0–__UPPER__</button><button data-scale="sqrt">当前范围：平方根</button></div>
<div class="pair"><figure><img src="figures/fig3_streamlines.png" alt="新模型在 z=5 m 的源区流线">
<figcaption><strong>图 3 · 新计算的诱导运动</strong><br>8 条流线从源区出发，使用实际二维速度积分。__ABSORBED__/8 到达目标，__BLOCKS__ 条在障碍界面停止。它们是瞬时流线。</figcaption>
<div class="downloads"><a href="figures/fig3_streamlines.png">PNG</a><a href="figures/fig3_streamlines.pdf">PDF</a></div></figure>
<figure><img id="density" src="figures/fig4_density.png" alt="新模型的密度等值图">
<figcaption><strong>图 4 · 新计算的稳态密度</strong><br><span id="scale-note">论文线性色标：0–0.05 UAVs/m³。</span>原始密度峰值 __PEAK__，源区峰值 __SOURCE_PEAK__。</figcaption>
<div class="downloads"><a id="density-png" href="figures/fig4_density.png">PNG</a><a id="density-pdf" href="figures/fig4_density.pdf">PDF</a></div></figure></div>
<p>这组计算的热点更靠近目标，论文图 4 的热点则靠近源区；版式接近不代表已复现这一物理差异。论文未完整给出源强及风场系数，本页没有调整像素或把密度热点搬到源区。<a href="figure_manifest.json">图像与数据来源</a> · <a href="independent_output_audit.json">重新计算的面通量审计</a>。</p>
__REF34__
<h2>论文图 2：六场景价值场</h2>
<figure class="wide"><img src="figures/fig2_value.png" alt="六场景价值函数等值填色图">
<figcaption>上排 Homing：无风、横风、横风加障碍；下排 P2P：无风、旋涡风、旋涡风加障碍。每幅图共用 0–8 s 的价值尺度。</figcaption>
<div class="downloads"><a href="figures/fig2_value.png">PNG</a><a href="figures/fig2_value.pdf">PDF</a></div></figure>
__REF2__
<h2>论文返修图：三维多高度密度</h2>
<div class="controls"><span>高度图色标</span><button class="active" data-height-scale="relative">当前切片共用：0–__HEIGHT_UPPER__</button>
<button data-height-scale="paper">论文共用：0–0.05</button></div>
<figure class="wide"><img id="heights" src="figures/height_slices_relative.png" alt="三维多高度密度切片">
<figcaption>3 行 × 4 列，采用论文的 1.9 / 9.4 / 20.6 / 28.1 m 高度。<span id="height-scale-note">全部面板共享线性 0–__HEIGHT_UPPER__ UAVs/m³。</span>红绿圆圈为 XY 投影；高度之间仅作显示插值。当前球形源在最外两层没有形成密度带，保留实际零值。</figcaption>
<div class="downloads"><a id="height-png" href="figures/height_slices_relative.png">PNG</a><a id="height-pdf" href="figures/height_slices_relative.pdf">PDF</a></div></figure>
__REFHEIGHT__
<h2>论文图 5：方法对照版式</h2>
<figure class="wide" style="max-width:940px"><img src="figures/fig5_methods.png" alt="独立混合模型、单体 PINN 和 FSM 的三行两列对照">
<figcaption>沿用三行两列及每幅独立色条。这里展示此前相同三维无风场景、q=__METHOD_Q__ 的对照，实际高度 z=14.06 m；密度按本组数据范围共用线性色标。单体 PINN 为 2000 轮预算且未通过守恒检查，未冒充论文强化基线。</figcaption>
<div class="downloads"><a href="figures/fig5_methods.png">PNG</a><a href="figures/fig5_methods.pdf">PDF</a></div></figure>
__REF5__
<h2>论文图 6：速度映射和训练记录</h2>
<figure class="wide"><img src="figures/fig6_diagnostics.png" alt="当前速度密度曲线与真实训练日志">
<figcaption>理论 SmoothMax–Greenshields 与当前密度样本；右图来自本次训练 CSV，Eikonal 损失从已记录总损失扣除因果项得到。未复制论文的曲线或训练轮数。</figcaption>
<div class="downloads"><a href="figures/fig6_diagnostics.png">PNG</a><a href="figures/fig6_diagnostics.pdf">PDF</a></div></figure>
<h2>本组二维 / 三维场景的实际审计</h2><div class="scroll"><table><thead><tr><th>场景</th><th>检查</th><th>吸收 / 源项</th><th>全局质量差</th><th>连续性相对残差</th><th>Eikonal 均值</th><th>ρ 峰值</th></tr></thead><tbody>__ROWS__</tbody></table></div>
<p class="muted">等值填色仅在原始网格单元中心之间作显示插值，没有修改 fields.npz。计算使用项目 .venv、现有 WSL NVIDIA 驱动、GPU PINN 与 CUDA 图输运、两个场景进程及每进程 8 个 CPU 线程。</p>
<p><a href="../eswa_verified/report.html">此前三维验证、独立方法对照与加速实测</a> · <a href="../__PLANAR__/__RUN_RECORD__">当前二维运行来源与选择</a></p>
</main><script>
const variants={paper:['fig4_density','论文线性色标：0–0.05 UAVs/m³。'],relative:['fig4_density_relative','当前范围的线性色标：0–__UPPER__ UAVs/m³。'],sqrt:['fig4_density_sqrt','平方根颜色映射：刻度保持真实密度，非论文线性色标。']};
document.querySelectorAll('[data-scale]').forEach(button=>button.addEventListener('click',()=>{
const [name,note]=variants[button.dataset.scale];document.getElementById('density').src='figures/'+name+'.png';
document.getElementById('density-png').href='figures/'+name+'.png';document.getElementById('density-pdf').href='figures/'+name+'.pdf';
document.getElementById('scale-note').textContent=note;document.querySelectorAll('[data-scale]').forEach(b=>b.classList.toggle('active',b===button));
}));
document.querySelectorAll('[data-height-scale]').forEach(button=>button.addEventListener('click',()=>{
const paper=button.dataset.heightScale==='paper',name=paper?'height_slices':'height_slices_relative';
document.getElementById('heights').src='figures/'+name+'.png';document.getElementById('height-png').href='figures/'+name+'.png';
document.getElementById('height-pdf').href='figures/'+name+'.pdf';document.getElementById('height-scale-note').textContent=
paper?'全部面板共享论文线性 0–0.05 UAVs/m³。':'全部面板共享当前范围线性 0–__HEIGHT_UPPER__ UAVs/m³。';
document.querySelectorAll('[data-height-scale]').forEach(b=>b.classList.toggle('active',b===button));
}));
</script></body></html>'''
    for key, value in {"__Q__": str(details["source_rate"]), "__UPPER__": f'{details["adaptive_density_range"][1]:g}',
                       "__PEAK__": f"{peak:.4f}", "__SOURCE_PEAK__": f"{source_peak:.4f}",
                       "__ABSORBED__": str(absorbed), "__BLOCKS__": str(blocks),
                       "__ROWS__": "".join(rows), "__PLANAR__": planar_root.name,
                       "__HEIGHT_UPPER__": f'{details["height_gallery"]["adaptive_shared_range"][1]:g}',
                       "__THREE_D_Q__": str(details["height_gallery"]["source_rate"]),
                       "__METHOD_Q__": str(details["method_comparison"]["source_rate"]),
                       "__BOUND__": str(details["density_upper_bound"]), "__LOAD_SECTION__": load_section,
                       "__RUN_RECORD__": run_record,
                       "__REF34__": reference_section(details["reference_comparison"], ("fig3", "fig4"), "查看论文图 3 / 图 4 原图"),
                       "__REF2__": reference_section(details["reference_comparison"], ("fig2",), "查看论文图 2 原图"),
                       "__REFHEIGHT__": reference_section(details["reference_comparison"], ("height_slices",), "查看论文多高度原图"),
                       "__REF5__": reference_section(details["reference_comparison"], ("fig5",), "查看论文图 5 原图"),
                       "__SATURATION__": ('超过 0.05 的密度以黄色和色条顶端箭头标示；完整范围选项显示所有真实值。'
                                          if details["paper_color_saturation"] else '切换显示不改变原始密度。')}.items():
        text = text.replace(key, value)
    (output / "report.html").write_text(text, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--planar-root", type=Path, default=Path("results/paper_figures_verified"))
    parser.add_argument("--three-d-root", type=Path, default=Path("results/eswa_verified"))
    parser.add_argument("--comparison-root", type=Path, default=Path("results/eswa_verified"))
    parser.add_argument("--baseline-planar-root", type=Path)
    parser.add_argument("--baseline-three-d-root", type=Path)
    parser.add_argument("--output", type=Path, default=Path("results/paper_style"))
    parser.add_argument("--reference-paper", type=Path,
                        help="Optional private folder containing main.tex and figures/; omitted in public reproduction")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    source_paths = [args.planar_root / name / "fields.npz" for name, _ in CASES]
    source_paths += [args.three_d_root / name / "fields.npz" for name in ("p2p_none", "p2p_obstacle", "p2p_height_obstacle")]
    source_paths += [args.comparison_root / "p2p_none/fields.npz"]
    source_paths += [args.comparison_root / "baselines" / name / "fields.npz" for name in ("fsm", "monolithic")]
    if bool(args.baseline_planar_root) != bool(args.baseline_three_d_root):
        raise ValueError("Both baseline roots are required for demand comparison")
    if args.baseline_planar_root:
        source_paths += [args.baseline_planar_root / name / "fields.npz" for name, _ in CASES]
        source_paths += [args.baseline_three_d_root / name / "fields.npz"
                         for name in ("p2p_none", "p2p_obstacle", "p2p_height_obstacle")]
    hashes_before = {str(path): digest(path) for path in source_paths}
    audits = [audit_case(args.planar_root, name) for name, _ in CASES]
    audits += [audit_case(args.three_d_root, name) for name in ("p2p_none", "p2p_obstacle", "p2p_height_obstacle")]
    (args.output / "independent_output_audit.json").write_text(json.dumps(audits, indent=2) + "\n")
    if not all(item["passed"] for item in audits):
        raise RuntimeError("Unverified outputs cannot be presented as the accepted figure set")
    details = make_paper_figures(args.planar_root, args.three_d_root, args.output / "figures", comparison_root=args.comparison_root)
    if args.baseline_planar_root:
        details["load_comparison"] = density_load_comparison(args.baseline_planar_root, args.planar_root,
                                                            args.output / "figures", low_three_d_root=args.baseline_three_d_root,
                                                            high_three_d_root=args.three_d_root)
        (args.output / "density_load_comparison.json").write_text(json.dumps(details["load_comparison"], indent=2) + "\n")
    if any(digest(Path(path)) != value for path, value in hashes_before.items()):
        raise RuntimeError("Rendering changed a numerical field file")
    reference_comparison = copy_reference_paper(args.reference_paper, args.output)
    details.update(command=sys.argv, planar_root=str(args.planar_root), three_d_root=str(args.three_d_root),
                   original_fields_unchanged=True, numerical_inputs_sha256=hashes_before,
                   manuscript_sha256=reference_comparison["manuscript_sha256"],
                   reference_comparison=reference_comparison,
                   renderer_sha256=digest(Path("src/uav_mfg/paper_plots.py")),
                   references="Explicit optional manuscript appearance comparison; never numerical inputs")
    (args.output / "figure_manifest.json").write_text(json.dumps(details, indent=2) + "\n")
    render(args.output, details, audits, args.planar_root, args.three_d_root)
    print(json.dumps({"report": str(args.output / "report.html"), "audited_cases": len(audits), **details}, indent=2))


if __name__ == "__main__":
    main()
