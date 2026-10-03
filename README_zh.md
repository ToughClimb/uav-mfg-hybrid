# UAV-MFG-Hybrid：ESWA 论文独立重建

[English](README.md) · [完整复现顺序](docs/REPRODUCTION.md) · [旧版迁移说明](docs/MIGRATION.md) · [验证摘要](validation/summary.json)

当前代码入口为 **`src/uav_mfg/`**，依据投向 *Expert Systems with Applications*
的 **ESWA-D-26-05581R1 最终返修清稿**独立实现 PINN 价值函数与保守 FVM 密度输运。

**旧 `uavpinn/`、`paperconfig/`、`runs/` 已移出默认代码树**，保存在
[legacy/pre-eswa-reconstruction](https://github.com/ToughClimb/uav-mfg-hybrid/tree/legacy/pre-eswa-reconstruction)
分支与完整 Git 历史中。旧算法、参数、checkpoint 和输出没有参与新计算。
旧版 checkout 的说明仅适用于旧版，见 [迁移记录](docs/MIGRATION.md)。

这里公开的是可审计的独立重建。论文未完整给出的几何、源强和风场系数均是
明确记录的重建选择；图旅行时间初始化、单侧 Bellman 约束和残差达标后跳过
重复训练是本次新增做法。**当前结果不宣称逐值复现论文原图或原表格。**
完整敏感性扫描、ABM、可选扩散和 20000 轮强化单体 PINN 基线不在当前范围中。

## 环境与运行

使用 Python 3.12+ 和 `uv`，依赖全部安装到项目 `.venv`，默认走清华 PyPI 镜像。
WSL 使用已有 NVIDIA 驱动，无须为此项目安装系统驱动或系统 CUDA 开发套件。

```bash
uv venv
uv sync --locked --extra test
.venv/bin/python scripts/build_native.py
.venv/bin/python -m pytest -q
```

`build_native.py` 用已有 `g++` 编译可选 OpenMP FSM 对照核，产物只写入 `build/`；
混合求解器不依赖这个可选核。没有编译器时可省略该步，相应测试会明确跳过。

八个三维场景使用两个 GPU 进程，每个进程 8 个 CPU 线程：

```bash
OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 .venv/bin/python -m uav_mfg.cli run \
  --config configs/verified_demo.yaml --profile demo --workers 2 \
  --output results/eswa_rebuild --device cuda
```

仅 CPU 的运行可显式选择，价值网络与输运都会使用 CPU：

```bash
OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 .venv/bin/python -m uav_mfg.cli run \
  --config configs/verified_demo.yaml --profile diagnostic --cases p2p_none \
  --output results/cpu_diagnostic --device cpu
```

请求 CUDA 而 GPU 不可用时会报错。三维网格 `diagnostic` 为 20×20×8、`demo`
为 40×40×16、`paper` 为 100×100×50；二维配置的 `paper` 是明确声明的
100×100×1 单位厚度平面。网格名称不表示所有场景参数来自论文原实验。

增加真实密度的九个初始试算：

```bash
.venv/bin/python scripts/run_high_density.py --workers 2
```

原试算、失败记录、同参数续算及障碍 Homing 的单独两倍源强试算均保留。
后续选择与出图需要按 [完整复现顺序](docs/REPRODUCTION.md) 执行。

## 审计与公开结果

每个场景保存完整参数、seed、环境版本、源码散列、checkpoint、训练与 Picard
日志、真实场和收敛状态。质量审计使用实际共享面通量；CUDA 图逐步保留质量账本。
续算不清空累计裁剪质量。正式出图要求全局质量差及最终连续性相对残差小于
1e-3，零外壁/障碍泄漏，零累计裁剪损失，并在渲染前后核对数值文件散列。
公共出图不需要私有投稿包；只有显式提供 `--reference-paper` 才加入原图外观对照。

九个已接受高密度场景的最大相对质量差为 **3.679e-4**、最大连续性残差为
**9.919e-4**，全部零禁止边界通量、零累计裁剪。参数、场文件散列、失败试算
和独立面通量审计摘要公开在 [validation/](validation/README.md)。

![主场景真实密度，线性色标覆盖实际范围](validation/figures/fig4_density_relative.png)

二维 P2P 源强由 0.006 增至 0.018，三维由 0.001 增至 0.01，重建密度上限选为
0.25。前后对照共用线性 0–0.05，色条顶端箭头标明上溢，未裁剪真实密度。

![同一色标下的真实低负载与高负载](validation/figures/density_load_comparison.png)

当前密度热点靠近目标，与论文源区热点仍有差异；三维球形源在最外两层未形成
密度带。增大密度并未消除这些物理差异。2000 轮单体 PINN 对照未通过质量审计，
按失败结果披露，不据此宣称论文级方法优势。详见
[方程与范围](docs/REIMPLEMENTATION.md)及[图像语义](docs/PAPER_FIGURES.md)。

沿用仓库的 [Apache-2.0 许可证](LICENSE)。
