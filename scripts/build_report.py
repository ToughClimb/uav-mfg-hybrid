"""Create a local, self-contained report from actual solver outputs."""

import argparse
import csv
import html
import json
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize

from uav_mfg.plots import load, make_figures, markers, save


def comparison(root):
    directories = [root / "p2p_none", root / "baselines/monolithic", root / "baselines/fsm"]
    if not all((p / "fields.npz").exists() for p in directories):
        return None
    datasets = [load(p.parent, p.name) for p in directories]
    labels = ["Hybrid PINN–FVM", "Monolithic PINN (2000 epochs)", "Independent FSM–FVM"]
    fig, axes = plt.subplots(3, 2, figsize=(10, 12), layout="constrained")
    maxima = {field: max(float(np.max(data[field])) for _, data in datasets) for field in ("phi", "density")}
    images = []
    for row, (label, (config, data)) in enumerate(zip(labels, datasets)):
        k = int(np.abs(data["z"] - 15).argmin())
        for col, field in enumerate(("phi", "density")):
            ax = axes[row, col]
            background = np.ma.array(data[field][:, :, k].T, mask=data["obstacle"][:, :, k].T)
            image = ax.pcolormesh(data["x"], data["y"], background, shading="nearest", cmap="viridis",
                                 norm=Normalize(0, maxima[field]))
            markers(ax, config, float(data["z"][k]))
            ax.set_title(label + (" — value" if field == "phi" else " — density"))
            if row == 0:
                images.append(image)
    for col, label in enumerate((r"Value $\phi$ (s)", r"Density $\rho$ (UAVs/m$^3$)")):
        fig.colorbar(images[col], ax=list(axes[:, col]), label=label, shrink=0.85)
    fig.suptitle("Same reconstructed geometry; shared scale in each column", fontsize=14)
    save(fig, root / "figures", "method_comparison")
    hybrid, reference = datasets[0][1], datasets[2][1]
    if hybrid["phi"].shape != reference["phi"].shape:
        raise ValueError("Method comparison requires the same grid")
    mask = hybrid["active"] & reference["active"]
    difference = hybrid["phi"][mask] - reference["phi"][mask]
    diagnostics = {"scope": "Current reconstructed no-wind/no-obstacle setting; same demo grid",
                   "relative_l2_value_error": float(np.linalg.norm(difference) / np.linalg.norm(reference["phi"][mask])),
                   "rmse_value": float(np.sqrt(np.mean(difference ** 2))),
                   "evaluation_points": int(mask.sum()), "monolithic_epochs": 2000}
    (root / "method_comparison.json").write_text(json.dumps(diagnostics, indent=2) + "\n")
    return diagnostics


def collect(root):
    records = []
    for directory in sorted(root.iterdir()):
        if not directory.is_dir() or not (directory / "diagnostics.json").exists():
            continue
        diagnostics = json.loads((directory / "diagnostics.json").read_text())
        metadata = json.loads((directory / "metadata.json").read_text())
        with (directory / "picard.csv").open() as stream:
            iterations = list(csv.DictReader(stream))
        timings = {key: sum(float(row.get(key) or 0) for row in iterations)
                   for key in ("value_seconds", "velocity_seconds", "transport_seconds")}
        chain=[];current=directory;visited=set()
        while str(current) not in visited:
            visited.add(str(current))
            run_metadata=json.loads((current/"metadata.json").read_text())
            chain.append({"directory":str(current),"status":run_metadata["status"],
                          "duration_seconds":run_metadata["duration_seconds"]})
            parent=run_metadata.get("resume")
            if not parent:
                break
            current=Path(parent).parent
            if not (current/"metadata.json").exists():
                raise FileNotFoundError(f"Missing continuation provenance: {current}")
        records.append({"case": directory.name, "status": metadata["status"], "grid": metadata["config"]["grid"],
                        "seed": metadata["seed"], "setup_seconds": metadata.get("setup_seconds", 0),
                        "timings": timings, "last_accepted_density_change": float(iterations[-1]["accepted_density_change"]),
                        "last_outer_index": int(iterations[-1]["outer"]),
                        "trained_value_epochs": sum(int(row.get("trained_value_epochs") or 0) for row in iterations),
                        "run_chain":chain,"cumulative_compute_seconds":sum(run["duration_seconds"] for run in chain),
                        **diagnostics})
    return records


def render(root, records, benchmark, comparison_diagnostics):
    rows, slices, density_statistics = [], {}, []
    rho_upper = 0
    for item in records:
        name = item["case"]
        config, fields = load(root, name)
        rho_upper = max(rho_upper, float(np.max(fields["density"])))
        positive=fields["density"][fields["active"] & (fields["density"]>1e-12)]
        slice_statistics=[]
        for height in (10,15,20):
            k=int(np.abs(fields["z"]-height).argmin())
            slab=fields["density"][:,:,k]
            slice_statistics.append({"z":float(fields["z"][k]),"maximum":float(slab.max()),
                                     "p99":float(np.quantile(slab,.99))})
        density_statistics.append({"case":name,"source_rate_per_volume":config["source"]["rate"],
                                   "maximum_density":float(fields["density"].max()),
                                   "rho_jam":config["physics"]["rho_jam"],
                                   "peak_over_rho_jam":float(fields["density"].max()/config["physics"]["rho_jam"]),
                                   "positive_density_p50_p90_p99":np.quantile(positive,[.5,.9,.99]).tolist(),
                                   "slices":slice_statistics})
        # Display precision only; the underlying NPZ retains solver precision.
        slices[name] = {"shape": list(fields["density"].shape), "z": fields["z"].tolist(), "config": config,
                        "density": np.asarray([float(f"{v:.6g}") for v in fields["density"].ravel()]).tolist(),
                        "obstacle": fields["obstacle"].ravel().astype(int).tolist(), "status": item["status"]}
        status = "达到停止条件" if item["picard_converged"] else "未收敛"
        style = "ok" if item["picard_converged"] else "pending"
        rows.append(f'<tr><td><a href="{name}/diagnostics.json">{html.escape(name)}</a></td>'
                    f'<td><span class="badge {style}">{status}</span></td>'
                    f'<td>{item["absorption_over_source"]:.6f}</td><td>{item["relative_steady_balance"]:.2e}</td>'
                    f'<td>{item["pde_relative_residual"]:.2e}</td><td>{item["eikonal_mean"]:.3g}</td>'
                    f'<td>{item["eikonal_p95"]:.3g}</td><td>{item["last_accepted_density_change"]:.2e}</td>'
                    f'<td>{item["cumulative_compute_seconds"]:.1f}</td></tr>')
    baseline_rows = []
    for name in ("fsm", "monolithic"):
        path = root / "baselines" / name / "diagnostics.json"
        if path.exists():
            data = json.loads(path.read_text())
            baseline_rows.append(f'<tr><td><a href="baselines/{name}/diagnostics.json">{name}</a></td>'
                                 f'<td>{data["absorption_over_source"]:.6f}</td><td>{data["outer_leakage"]:.3g}</td>'
                                 f'<td>{data["relative_steady_balance"]:.3g}</td><td>{data["duration_seconds"]:.1f}</td></tr>')
    benchmark_rows = [f'<tr><td>{"×".join(map(str,b["grid"]))}</td><td>{b["median_cpu_seconds"]:.3f}</td>'
                      f'<td>{b["median_cuda_seconds"]:.3f}</td><td>{b["speedup"]:.1f}×</td>'
                      f'<td>{b["maximum_density_difference"]:.2e}</td></tr>' for b in benchmark]
    panels = []
    captions = {"value_comparison": "价值场：六场景共享色标", "density_height_slices": "密度高度切片：整张图共享色标",
                "density_scale_comparison": "相同密度数据：线性色标与平方根色标对照",
                "density_height_slices_sqrt": "密度高度切片：共享平方根色标，刻度为真实密度",
                "motion_and_density": "诱导运动与密度：瞬时平面流线", "method_comparison": "独立方法对照：每列共享色标",
                "conservation_diagnostics": "实际面通量守恒与价值残差",
                "main_grid_motion_and_density": "100×100×50 主网格：无风 P2P 代表算例"}
    for name, caption in captions.items():
        if (root / "figures" / f"{name}.png").exists():
            panels.append(f'<figure><figcaption>{caption} · <a href="figures/{name}.pdf">PDF</a></figcaption>'
                          f'<a href="figures/{name}.png"><img src="figures/{name}.png" alt="{caption}" loading="lazy"></a></figure>')
    verified = sum(item["picard_converged"] for item in records)
    max_error = max(item["maximum_inner_mass_budget_error"] for item in records)
    max_balance = max(item["relative_steady_balance"] for item in records)
    preferred="p2p_vortex_obstacle" if "p2p_vortex_obstacle" in slices else next(iter(slices))
    options = "".join(f'<option value="{html.escape(name)}"'+(' selected' if name==preferred else '')+
                      f'>{html.escape(name)}</option>' for name in slices)
    palette = (plt.colormaps["viridis"](np.linspace(0, 1, 256))[:, :3] * 255).round().astype(int).tolist()
    comparison_text = (f'混合价值场相对 FSM 的 L2 差为 {comparison_diagnostics["relative_l2_value_error"]:.4g}，'
                       f'在 {comparison_diagnostics["evaluation_points"]} 个当前网格点上计算。') if comparison_diagnostics else ""
    body = '''<!doctype html>
<html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>ESWA UAV PINN–FVM · 独立重建结果</title>
<style>
body{margin:0;background:#f3f5f5;color:#20343b;font:16px/1.65 system-ui,"Microsoft YaHei",sans-serif}
main{max-width:1200px;margin:auto;padding:32px 24px}h1{font-size:32px;line-height:1.35;margin:4px 0 14px}
h2{margin:32px 0 12px;font-size:22px}p{max-width:1000px}a{color:#087e8b}small,.muted{color:#596f78}
.eyebrow{color:#087e8b;letter-spacing:1px;font-weight:600}.cards{display:flex;gap:14px;flex-wrap:wrap;margin:22px 0}
.card{background:white;border:1px solid #d9e4e5;border-radius:9px;padding:16px 22px;min-width:190px}
.card strong{display:block;font-size:27px;color:#17666b}.card span{font-size:14px;color:#596f78}
figure,.box{margin:22px 0;background:white;border:1px solid #d9e4e5;border-radius:9px;padding:20px}
img{width:100%;height:auto;display:block}figcaption{font-weight:600;margin-bottom:12px}.scroll{overflow:auto}
table{width:100%;border-collapse:collapse;font-size:14px;background:white}th,td{text-align:left;padding:12px;white-space:nowrap;border-bottom:1px solid #e3ebeb}
th{background:#e8f0f0}.badge{padding:3px 8px;border-radius:12px;font-size:12px}.ok{background:#dceee6;color:#166c49}.pending{background:#fff0cc;color:#8b5700}
.controls{display:flex;gap:20px;align-items:center;flex-wrap:wrap}select,input{font:inherit}select{padding:7px;border:1px solid #b9cbcc;border-radius:5px}
canvas{width:100%;max-width:650px;background:white;margin:16px auto;display:block}.note{border-left:4px solid #21918c;padding:10px 16px;background:#eaf3f2}
code{background:#e8eeee;padding:2px 5px;border-radius:4px}.timings{font-variant-numeric:tabular-nums}
</style><main>
<div class="eyebrow">EXPERT SYSTEMS WITH APPLICATIONS · ESWA-D-26-05581R1</div>
<h1>三维风场下的 UAV 宏观输运<br>PINN–FVM 独立重建结果</h1>
<p>依据 2026-05-15 返修投稿包最终清稿，重新实现价值方程、诱导速度与保守输运。旧程序仅作为配色和图片布局参考。此页所有计算数据均来自新实现。</p>
__PAPER_LINK__
<div class="cards"><div class="card"><strong>__VERIFIED__ / __COUNT__</strong><span>演示场景达到停止条件</span></div>
<div class="card"><strong>__BALANCE__</strong><span>最大源项–目标吸收相对差</span></div>
<div class="card"><strong>__ERROR__</strong><span>最大逐步质量账本绝对误差</span></div></div>
<p class="note">几何、源强及风场系数包含文中未完整给出的重建选择。图旅行时间预训练和 Bellman 因果约束属于本次实现补充。页面区分 Picard 停止、连续性残差和价值残差，不能将停止条件解释为逐点精确解。详情见项目 README 与每个场景的 config、metadata、checkpoint 和 CSV。</p>
<h2>三维密度：选择场景和高度</h2><div class="box">
<div class="controls"><label>场景 <select id="case">__OPTIONS__</select></label><label>高度 <input id="height" type="range" min="0" value="8"> <b id="zlabel"></b></label>
<label>色标 <select id="scale"><option value="sqrt" selected>平方根（显示弱密度）</option><option value="linear">线性</option></select></label></div>
<canvas id="slice" width="660" height="560"></canvas><small id="slice-status"></small><br><small id="density-readout">移动鼠标到图上查看单元密度</small>
<p class="muted">所有高度和场景共用同一密度上限。平方根映射增强低密度颜色；刻度和鼠标读数均为真实密度。灰色为障碍，红虚线为目标，绿色为源；球形区域按当前高度绘制实际截面。</p>
<p>当前 P2P 验证源强为 0.001，基础重建配置为 0.01。部分走廊密度约为 10⁻⁴～10⁻³，而目标附近局部峰值约为 0.0116；共享线性色标会把弱密度区压到低色阶。<a href="density_statistics.json">查看原始密度统计</a>。</p></div>
<h2>收敛与守恒记录</h2><div class="scroll"><table><thead><tr><th>场景 / 原始诊断</th><th>状态</th><th>吸收 / 源项</th><th>全局质量差</th><th>连续性相对残差</th><th>Eikonal 均值</th><th>Eikonal P95</th><th>接受 Δρ</th><th>累计计算 / s</th></tr></thead><tbody>__ROWS__</tbody></table></div>
<p><a href="selection.json">结果选择与续算关系</a> · <a href="performance.json">分阶段耗时</a>。耗时为本次实际运行记录，续算时间单独记录；并发运行的时间不能直接用于不同方法的性能排序。</p>
__PANELS__
<h2>独立对照</h2><p>相同无风、无障碍场景。FSM 使用 8 线程 OpenMP；单体 PINN 为 2000 轮演示预算，未执行论文 20000 轮强化变体。单体 PINN 的外边界通量直接来自神经网络，采用随机积分，不通过 FVM 强制归零。</p>
<div class="scroll"><table><thead><tr><th>方法</th><th>目标吸收 / 源项</th><th>外边界绝对通量</th><th>相对质量差</th><th>实际耗时 / s</th></tr></thead><tbody>__BASELINES__</tbody></table></div><p>__COMPARISON__</p>
<h2>加速实测</h2><p>相同算子执行 500 个显式输运步，预热后测量 3 次，取中位数。包含传输、CUDA 图捕获与每步守恒审计；不包含网格与矩阵组装。PINN 在现有 RTX 5070 Ti 上训练，不同场景采用两个进程并行。</p>
<div class="scroll"><table><thead><tr><th>网格</th><th>CPU / s</th><th>CUDA 图 / s</th><th>输运加速</th><th>密度最大差</th></tr></thead><tbody>__BENCHMARK__</tbody></table></div>
<p class="muted">依赖只放在项目 .venv，通过清华镜像获取。使用已有 WSL NVIDIA 驱动。__MAIN_LINK__</p>
</main><script>
const datasets=__DATA__, palette=__PALETTE__, upper=__UPPER__;
const selector=document.getElementById('case'), slider=document.getElementById('height'), scale=document.getElementById('scale'), canvas=document.getElementById('slice'), ctx=canvas.getContext('2d');
function draw(reset=false){
 const d=datasets[selector.value], [nx,ny,nz]=d.shape, domain=d.config.domain;
 slider.max=nz-1;
 if(reset){const target=d.config.target, height=target.type==='sphere'?target.center[2]:(target.lower[2]+target.upper[2])/2;
  slider.value=d.z.reduce((best,z,i)=>Math.abs(z-height)<Math.abs(d.z[best]-height)?i:best,0);}
 const k=+slider.value,z=d.z[k],isSqrt=scale.value==='sqrt';
 document.getElementById('zlabel').textContent=z.toFixed(3)+' m';
 document.getElementById('slice-status').textContent='当前场景状态：'+(d.status==='converged'?'达到停止条件':'未收敛');
 ctx.clearRect(0,0,canvas.width,canvas.height);const left=70,top=20,width=480,height=480;
 for(let i=0;i<nx;i++)for(let j=0;j<ny;j++){
  const p=(i*ny+j)*nz+k, ratio=d.density[p]/upper, position=isSqrt?Math.sqrt(ratio):ratio;
  const color=palette[Math.max(0,Math.min(255,Math.round(position*255)))];
  ctx.fillStyle=d.obstacle[p]?'#606060':'rgb('+color.join(',')+')';
  ctx.fillRect(left+i*width/nx,top+(ny-1-j)*height/ny,width/nx+.2,height/ny+.2);
 }
 const px=x=>left+(x-domain[0][0])/(domain[0][1]-domain[0][0])*width;
 const py=y=>top+height-(y-domain[1][0])/(domain[1][1]-domain[1][0])*height;
 function sphere(region,color,dashed){if(region.type!=='sphere')return;const square=region.radius**2-(z-region.center[2])**2;if(square<=0)return;
  ctx.beginPath();ctx.strokeStyle=color;ctx.lineWidth=2.5;ctx.setLineDash(dashed?[7,5]:[]);
  ctx.arc(px(region.center[0]),py(region.center[1]),Math.sqrt(square)*width/(domain[0][1]-domain[0][0]),0,2*Math.PI);ctx.stroke();ctx.setLineDash([]);}
 sphere(d.config.target,'#ff3333',true);sphere(d.config.source,'#00ad50',false);
 ctx.strokeStyle='#20343b';ctx.lineWidth=1;ctx.strokeRect(left,top,width,height);ctx.fillStyle='#20343b';ctx.font='14px system-ui';
 for(let t=0;t<=5;t++){const ratio=t/5,x=left+ratio*width,y=top+height-ratio*height;ctx.textAlign='center';ctx.fillText((domain[0][0]+ratio*(domain[0][1]-domain[0][0])).toFixed(0),x,top+height+23);ctx.textAlign='right';ctx.fillText((domain[1][0]+ratio*(domain[1][1]-domain[1][0])).toFixed(0),left-10,y+5);}
 ctx.textAlign='center';ctx.fillText('x (m)',left+width/2,top+height+48);ctx.save();ctx.translate(22,top+height/2);ctx.rotate(-Math.PI/2);ctx.fillText('y (m)',0,0);ctx.restore();
 for(let q=0;q<256;q++){ctx.fillStyle='rgb('+palette[q].join(',')+')';ctx.fillRect(578,top+(255-q)*height/256,20,height/256+1);}
 ctx.fillStyle='#20343b';ctx.font='12px system-ui';ctx.textAlign='left';
 for(let tick=0;tick<=4;tick++){const fraction=tick/4, value=upper*(isSqrt?fraction*fraction:fraction);
  ctx.fillText(value===0?'0':value.toExponential(1),604,top+(1-fraction)*height+4);}
 ctx.save();ctx.translate(651,top+height/2);ctx.rotate(-Math.PI/2);ctx.textAlign='center';ctx.fillText('density (UAV/m³)',0,0);ctx.restore();
 document.getElementById('density-readout').textContent='移动鼠标到图上查看单元密度';
}
selector.addEventListener('change',()=>draw(true));slider.addEventListener('input',()=>draw());scale.addEventListener('change',()=>draw());
canvas.addEventListener('mousemove',event=>{
 const bounds=canvas.getBoundingClientRect(),x=(event.clientX-bounds.left)*canvas.width/bounds.width,y=(event.clientY-bounds.top)*canvas.height/bounds.height;
 if(x<70||x>=550||y<20||y>=500)return;
 const d=datasets[selector.value],[nx,ny,nz]=d.shape,i=Math.floor((x-70)/480*nx),j=ny-1-Math.floor((y-20)/480*ny),k=+slider.value;
 const density=d.density[(i*ny+j)*nz+k],domain=d.config.domain;
 const physicalX=domain[0][0]+(i+.5)/nx*(domain[0][1]-domain[0][0]),physicalY=domain[1][0]+(j+.5)/ny*(domain[1][1]-domain[1][0]);
 document.getElementById('density-readout').textContent='ρ = '+density.toExponential(3)+' UAV/m³；x = '+physicalX.toFixed(2)+' m，y = '+physicalY.toFixed(2)+' m，z = '+d.z[k].toFixed(2)+' m';
});draw(true);
</script></html>'''
    paper_link = ('<p class="note"><a href="../paper_style/report.html"><strong>查看论文版式的新图：z=5 m 主图、六宫格价值场、源区流线与原图对照</strong></a></p>'
                  if (root.parent / "paper_style/report.html").exists() else "")
    replacements = {"__PAPER_LINK__": paper_link, "__VERIFIED__": str(verified), "__COUNT__": str(len(records)), "__BALANCE__": f"{max_balance:.2e}",
                    "__ERROR__": f"{max_error:.2e}", "__OPTIONS__": options, "__ROWS__": "".join(rows),
                    "__PANELS__": "".join(panels), "__BASELINES__": "".join(baseline_rows),
                    "__COMPARISON__": comparison_text, "__BENCHMARK__": "".join(benchmark_rows),
                    "__DATA__": json.dumps(slices, ensure_ascii=False).replace("</", "<\\/"),
                    "__PALETTE__": json.dumps(palette), "__UPPER__": repr(rho_upper),
                    "__MAIN_LINK__": '<a href="../eswa_main_verified/p2p_none/diagnostics.json">100×100×50 主网格代表算例</a>'
                       if (root.parent / "eswa_main_verified/p2p_none/diagnostics.json").exists() else ""}
    for key, value in replacements.items():
        body = body.replace(key, value)
    save_statistics={"shared_density_upper":rho_upper,"color_mappings":["linear","square_root_gamma_0.5"],
                     "scope":"Display transforms only; original density arrays unchanged", "cases":density_statistics}
    (root/"density_statistics.json").write_text(json.dumps(save_statistics,indent=2,ensure_ascii=False)+'\n')
    (root / "report.html").write_text(body, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("results/eswa_verified"))
    parser.add_argument("--benchmark", type=Path, default=Path("results/acceleration_benchmark.json"))
    args = parser.parse_args()
    make_figures(args.root)
    main_fields=args.root.parent/"eswa_main_verified/figures"
    if main_fields.exists():
        for extension in ("png","pdf"):
            shutil.copyfile(main_fields/f"motion_and_density.{extension}",
                            args.root/"figures"/f"main_grid_motion_and_density.{extension}")
    diagnostics = comparison(args.root)
    records = collect(args.root)
    if not records:
        raise ValueError("Report requires completed solver fields and diagnostics")
    (args.root / "performance.json").write_text(json.dumps(records, indent=2) + "\n")
    benchmark = json.loads(args.benchmark.read_text())
    render(args.root, records, benchmark, diagnostics)
    print(args.root / "report.html")


if __name__ == "__main__":
    main()
