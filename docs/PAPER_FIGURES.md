# 论文版式出图与二维重算

依据同一份 ESWA-D-26-05581R1 清稿及其图片。原代码仍只读归档；
没有导入旧算法、密度数组、训练结果或原图像素来生成新数据。

新图片同时输出 PNG 与 PDF。公共复现流程不依赖投稿包，完整顺序见
[REPRODUCTION.md](REPRODUCTION.md)。本地可查看生成的 `results/paper_style/report.html`。
只有显式提供 `--reference-paper reference/paper` 时才加入私有论文原图外观对照；
这些原图明确标为非本次计算输出，不随公共代码发布。

图 2–4 采用等值填色、24 阶 viridis 色条、细白色等值线、目标红虚线、
源绿线和灰色障碍；每面板单独色条，六宫格有下方分图标题。
图 3 从源区横断面选 8 个起点，用实际二维速度积分，记录到达目标、碰到障碍
或外壁的停止事件，不手绘期望路线。图 4 默认线性 0–0.05；自适应线性及
平方根仅为额外显示方式，色条保留真实物理密度。

最终论文既报告二维障碍/风场诊断，也报告三维高度诊断，主图 2–5 标为
z=5 m。因此本次额外建立明确的二维场景，而不把先前 z≈15 m 的三维球源截面
伪装成原主图。二维部分仍解相同耦合方程，但梯度和诱导运动仅限 XY：

- 域为 `[0,100]²`，输出使用 `[4.5,5.5]` 的单位厚度层存储，z 网格只有 1 层。
- 源和目标为 XY 圆盘；网络输入只使用 XY，目标距离为平面圆盘有符号距离。
- `phi` 与 z 无关，诱导速度的 z 分量严格为零。输运拒绝非零竖直速度。
- 面共享上风通量、吸收目标、障碍和外壁无通量及停止阈值保持原实现语义。
- 风场、源强、几何仍是公开的重建选择，文中没有给出可恢复完整原运行的参数。

先运行的 P2P 源强 q=0.01 触及 rho_max=0.05，三个 P2P 场景均因裁剪移除质量
而未收敛。原试算完整保存在 `results/paper_figures_2d`，没有删除或改标成功。
P2P 改以 q=0.006 从头求解，不跨不同源强恢复旧 checkpoint。六幅正式主图
选择三个已通过的 Homing 和三个新 P2P；来源及未通过试算记录保存于
`results/paper_figures_verified/selection.json`。

三维高度图沿用 `[1.875,9.375,20.625,28.125]` m 的论文诊断网格中心高度，在此前
40×40×16 输出之间作显示插值。红绿标记为 XY 投影，不宣称是球的实际截面。
全部切片默认共用当前切片范围的线性色标，另可切换论文的 0–0.05，后者会很淡；
两种显示保存为独立图片，没有逐面板调节颜色范围。
当前球形源与诱导速度没有在最外两层生成密度带，图中保留真实零值。
方法对照图沿用原三维无风、无障碍的同一组数据和实际 z=14.0625 m。
单体 PINN 为 2000 轮且未通过质量检查；列内共用当前数据范围的线性色标。
图 6 的样本和曲线来自新密度与训练 CSV，不复制论文训练历史。

当前保守平流结果在目标附近聚集，原图 4 则源附近最亮。这一物理差异仍未
复现。调色、插值或提高源强不能诚实地搬动热点，页面明确保留该区别。

```bash
mkdir -p results/paper_figures_2d results/paper_figures_verified
OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 .venv/bin/python -m uav_mfg.cli run \
  --config configs/paper_figures_2d.yaml --profile paper --workers 2 \
  --output results/paper_figures_2d --device cuda
OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 .venv/bin/python -m uav_mfg.cli run \
  --config configs/paper_figures_verified.yaml --profile paper --workers 2 \
  --output results/paper_figures_verified --device cuda
.venv/bin/python scripts/select_paper_figures.py
OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 .venv/bin/python scripts/build_paper_report.py
.venv/bin/python scripts/build_report.py --root results/eswa_verified
```

选择脚本拒绝覆盖已有运行目录。出图脚本重新加载所有正式二维场，按实际共享面
通量重算全局质量及连续性残差，要求零裁剪、零边界泄漏和真实收敛状态。
生成前后检查全部输入 `fields.npz` SHA256，保证显示处理未修改数值文件。
代码中有二维硬目标、竖直梯度为零、圆盘采样、面通量平衡和维度错误拒绝测试。

## 增加真实密度

按后续请求重新增加注入源项，未直接乘已有密度数组。二维 P2P 的 q 从 0.006
提高到 0.018，三维的 q 从 0.001 提高到 0.01。Homing 的无障碍源强提高三倍，
障碍 Homing 的三倍试算触及 0.25 上限，保留为未收敛；最终该场景采用两倍源强。
原始三维和二维结果仍保留，低负载完整图片页复制到 `results/paper_style_low_load`。

旧的 rho_max=0.05 是本次重建的场景上限，不是最终论文中明确给出的数值。
高密度场景选用 rho_max=0.25，保留原 rho_jam、v_free、v_min、beta、风场与几何。
仍要求真实质量平衡、最终连续性残差小于 1e-3、零裁剪质量和零禁止边界通量；
不能通过提高色标范围或放松这些停止条件来获得“成功”的高密度结果。

高密度从零密度开始重新求解，两个 GPU 场景进程、每进程 8 个 CPU 线程。
后续仅在源强、风场、几何、速度模型和网格都相同的条件下恢复 checkpoint
补足收敛，记录原试算与续算关系。此前同体积自由域平均密度和源区密度也参与
前后统计，比较图两侧共用线性 0–0.05。显示上溢时填色采用黄色、色条有顶端
箭头；完整线性色标涵盖当前全场峰值，原始 NPZ 未裁剪至图像色标。

```bash
OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 .venv/bin/python scripts/run_high_density.py
OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 .venv/bin/python -m uav_mfg.cli run \
  --config configs/high_density_2d.yaml --profile paper --cases homing_uniform \
  --output results/high_density_continuation_2d \
  --resume results/high_density_2d/homing_uniform/checkpoint.pt --device cuda
OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 .venv/bin/python -m uav_mfg.cli run \
  --config configs/high_density_homing_obstacle.yaml --profile paper \
  --output results/high_density_homing_obstacle --device cuda
OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 .venv/bin/python -m uav_mfg.cli run \
  --config configs/high_density_homing_obstacle.yaml --profile paper \
  --output results/high_density_homing_obstacle_continued \
  --resume results/high_density_homing_obstacle/homing_uniform_obstacle/checkpoint.pt --device cuda
.venv/bin/python scripts/select_high_density.py
OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 .venv/bin/python scripts/build_paper_report.py \
  --planar-root results/high_density_verified_2d \
  --three-d-root results/high_density_verified_3d \
  --comparison-root results/eswa_verified \
  --baseline-planar-root results/paper_figures_verified \
  --baseline-three-d-root results/eswa_verified
```

选择脚本拒绝覆盖原运行；失败高负载试算、降低负载的障碍 Homing 试算、同参数
续算均独立存储。方法图 5 继续展示 q=0.001 的旧负载方法对照，标题说明该来源；
没有将旧基线与新高负载混合声称是同一个对照实验。
