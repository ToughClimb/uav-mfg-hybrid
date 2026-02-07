# UAV-MFG-Hybrid（UAVPINN++）

面向**三维风场下无人机均值场博弈（Mean-Field Game, MFG）**的一套**混合求解框架**：

- 用 PINN 学习势函数 **φ**
- 用保守的迎风有限体积（Finite-Volume）格式求解密度 **ρ**
- 通过 Picard（固定点）迭代 + 松弛实现 φ–ρ 耦合

English version: see [README.md](./README.md).

## 方法概览

本仓库提供两种训练/求解路线：

- **Route B（推荐）**：固定点迭代（Picard + relaxation）
  - 固定 ρ，训练 φ（PINN）
  - 由 φ 构造速度场 u
  - 用**守恒迎风有限体积**求解 ρ 的守恒方程，得到 ρ̃
  - 松弛更新：**ρ ← (1−α)ρ + α·ρ̃**
- **Route A（基线对照）**：端到端双网络 PINN（联合训练 φ 与 ρ）

运行输出写入 `runs/<timestamp>_<exp_name>/`，包含日志、checkpoint、图像等。

## 特性

- **三维计算域**：可配置 x/y/z 边界
- **目标区域（sink）**：球体或 AABB
- **障碍物**：单个 AABB 或多个 AABB 的并集
- **风场模型**：
  - `none`、`uniform`、`vortex`、`height_dependent`
  - 可组合风场：**base + patches**（`CompositeWind` + `RegionConstantWind`，支持 hard/smooth 过渡）
- **可视化**：
  - φ/ρ 的二维切片图
  - 可选导出 **ParaView VTK** 用于三维可视化

## 目录结构

- `scripts/train.py`：主训练入口（Route A / Route B）
- `scripts/batch_train.py`：Route B 的批量复现实验入口
- `scripts/visualize.py`：从 checkpoint 生成可视化结果
- `uavpinn/`：核心库（几何、风场、物理、求解器、训练、可视化）

## 安装

### 1）创建环境

建议 Python **3.10+**。

```bash
python -m venv .venv
# Windows
.venv\\Scripts\\activate
# Linux/Mac
# source .venv/bin/activate
```

### 2）安装依赖

```bash
pip install -r requirements.txt
```

## 快速开始

### 单配置训练

Route B（固定点混合框架，推荐）：

```bash
python scripts/train.py --config path/to/config.yaml --route B --device auto
```

Route A（端到端基线）：

```bash
python scripts/train.py --config path/to/config.yaml --route A --device auto
```

说明：

- **输出目录**：`runs/<timestamp>_<exp_name>/`
- 如需关闭自动出图：加 `--no-viz`
- 若你刻意构造了**不可控**场景（例如某区域风速大于最大航速），可加：

```bash
python scripts/train.py --config path/to/config.yaml --route B --skip-physics-checks
```

### 批量复现（Route B）

对某个目录下所有 YAML 运行：

```bash
python scripts/batch_train.py --config-dir path/to/configs --glob "*.yaml" --device auto
```

或显式指定多个配置（可重复传参）：

```bash
python scripts/batch_train.py --config a.yaml --config b.yaml --device auto
```

关闭批量可视化：

```bash
python scripts/batch_train.py --config-dir path/to/configs --no-viz
```

### 从 checkpoint 可视化

```bash
python scripts/visualize.py --checkpoint runs/<run_dir>/checkpoints/checkpoint_final.pt
```

ParaView（VTK）导出：

```bash
python scripts/visualize.py --checkpoint runs/<run_dir>/checkpoints/checkpoint_final.pt --paraview
```

也可以在训练结束时直接导出：

```bash
python scripts/train.py --config path/to/config.yaml --route B --paraview
```

## 配置说明

所有实验由 YAML 配置驱动（通过 `--config` 指定）。主要字段包括：

- `domain`：x/y/z 边界
- `scenario`：`homing` 或 `p2p`
- `target`：目标区域几何
- `obstacles`：可选障碍列表
- `wind`：风场 base + patches
- `physics`：基本图（fundamental diagram）与源项参数
- `network`、`training`、`fixed_point`、`sampling`、`numerical`、`rho_solver`

## 许可证

若你计划公开发布，请在仓库根目录添加许可证文件（例如 MIT / Apache-2.0）。
