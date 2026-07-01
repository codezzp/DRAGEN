# DRAGEN

DRAGEN 是一个用于级联预测实验的工程目录。当前目录的核心约定是：每批数据固定放在 `work/runs/<run_id>/`，命令入口放在 `scripts/`，真正的实现逻辑放在 `src/dragen/`。

当前阶段已经完成工程结构、数据边界、30m/5m 窗口划分、HybridTree Light 全量树、MultiScale HybridTree 窗口、统计特征、弱监督标签、事件级 pack 和 DRAGEN-Full debug 训练。后续重点转入 baseline、DRAGEN-Full 正式训练、消融和结果表，不再继续扩展预处理结构。

## 目录结构

```text
DRAGEN/
  configs/                 # 实验配置，按 data/window/model/train 分组。
  docs/                    # 项目文档，不放运行代码。
  scripts/                 # 只放命令入口脚本。
  src/dragen/              # 核心源码。
  work/
    runs/<run_id>/         # 每个 run 是一个独立实验数据单元。
    artifacts/             # 跨 run 的实验产物，例如日志、模型、报告。
  tests/                   # 小规模验证测试。
  notebooks/               # 临时分析 notebook，不作为正式代码。
```

`work/` 已在 `.gitignore` 中排除，不提交实验数据、窗口 CSV、pack、checkpoint 或报告产物。需要提交的是代码、配置和文档摘要。

## 数据原则

不要把数据散到项目根目录。每批数据固定保留在：

```text
work/runs/<run_id>/
  processed/
  org_task/
  time_distribution/
  time_distribution/dev_cascade_lists/
```

后续训练和评估表应优先使用稳定的索引字段，例如 `cascade_idx`、`tweet_idx`、`user_idx`。真实原始 ID 只保留在映射文件或调试文件中，不作为模型输入。

当前窗口链路使用 `work/runs/<run_id>/org_task/` 作为输入，生成结果固定写入：

```text
work/runs/<run_id>/windows/obs_<obs>_win<window>_step<step>/
```

窗口构建不会修改 `processed/` 或 `org_task/` 下的原始中间表。

当前实验闭环的本地产物约定如下：

```text
work/runs/<run_id>/edges/
work/runs/<run_id>/windows/
work/runs/<run_id>/features/
work/runs/<run_id>/labels/
work/runs/<run_id>/packs/
work/artifacts/
```

## 脚本原则

`scripts/` 只做入口，不堆核心逻辑。入口脚本应该导入 `src/dragen` 里的 `main()` 再执行：

```python
from dragen.windowing.window_builder import main

if __name__ == "__main__":
    raise SystemExit(main())
```

真正的数据处理、窗口划分、特征构建、训练和评估逻辑都应该放在 `src/dragen/...`。

## 当前保留入口

```text
scripts/01_build_org_tables.py
scripts/02_analyze_time_distribution.py
scripts/03_build_dev_cascade_lists.py
scripts/05_build_windows.py
scripts/06_build_inferred_tree.py
scripts/07_visualize_tree.py
scripts/08_export_reports.py
scripts/11_build_features.py
scripts/12_build_weak_labels.py
scripts/13_build_packs.py
```

窗口入口会从 `work/runs/<run_id>/org_task/` 读取标准任务表，并写出四张窗口表。默认使用星形边：

```powershell
python scripts/05_build_windows.py --run-id run_0002 --window-config configs/window/obs_30m_step5m.yaml --edge-mode star
```

输出目录：

```text
work/runs/run_0002/windows/obs_1800_win300_step300_star/
  window_table.csv
  node_window_table.csv
  edge_window_table.csv
  text_window_table.csv
  cascade_window_index.json
  window_diagnostics.json
```

`run_0002` 的 30m/5m 完整构建结果已经生成，诊断摘要如下：

```text
cascades: 85263
windows_per_cascade: 6
window_rows: 511578
node_window_rows: 5940259
edge_window_rows: 1392078
text_window_rows: 6032866
root_text_window_rows: 511578
retweet_text_early_violations: 0
```

当前正式树结构使用不扫描关注图的 HybridTree Light：

```powershell
python scripts/06_build_inferred_tree.py --run-id run_0002 --method hybrid --max-observation-seconds 1800 --out-dir work/runs/run_0002/edges/hybrid_tree_light
```

正式窗口输入有三套：

```text
work/runs/run_0002/windows/obs_1800_win300_step300/
work/runs/run_0002/windows/obs_1800_win300_step300_hybrid_tree/
work/runs/run_0002/windows/obs_1800_step300_multiscale_hybrid_tree/
```

注意：`time_only` 会把大级联构造成接近线性的深链，只作为结构对照或反例。当前不继续扫描关注图，不继续扩展树结构设计。

树形结构可视化入口：

```powershell
python scripts/07_visualize_tree.py --run-id run_0002 --cascade-id 78857 --tree-edges work/runs/run_0002/edges/_viz_cascade_78857_branching/inferred_tree_edge_table.csv --out work/runs/run_0002/edges/_viz_cascade_78857_branching/cascade_78857_branching_tree.svg --max-nodes 180
```

窗口策略现在有两个层次：

- `Fixed-5m`：当前已生成的 baseline，非重叠 5 分钟窗口。
- `Causal MultiScale`：端点对齐多尺度因果窗口，每 5 分钟一个端点，同时生成 current 5m、context 10m 和 cumulative 历史统计。

正式 MultiScale HybridTree 命令：

```powershell
python scripts/05_build_windows.py --run-id run_0002 --window-config configs/window/obs_30m_step5m_multiscale.yaml --edge-mode inferred_tree --inferred-tree-edge-table work/runs/run_0002/edges/hybrid_tree_light/inferred_tree_edge_table.csv --out-dir work/runs/run_0002/windows/obs_1800_step300_multiscale_hybrid_tree
```

## 实验闭环入口

统计特征：

```powershell
python scripts/11_build_features.py --run-id run_0002 --tree-edges work/runs/run_0002/edges/hybrid_tree_light/inferred_tree_edge_table.csv
```

输出：

```text
work/runs/run_0002/features/obs_1800_win300_step300_star/
work/runs/run_0002/features/obs_1800_win300_step300_hybrid_tree/
work/runs/run_0002/features/obs_1800_step300_multiscale_hybrid_tree/
```

弱监督标签：

```powershell
python scripts/12_build_weak_labels.py --run-id run_0002 --feature-dir work/runs/run_0002/features/obs_1800_step300_multiscale_hybrid_tree
```

pack：

```powershell
python scripts/13_build_packs.py --run-id run_0002 --feature-dir work/runs/run_0002/features/obs_1800_step300_multiscale_hybrid_tree --window-dir work/runs/run_0002/windows/obs_1800_step300_multiscale_hybrid_tree --labels work/runs/run_0002/labels/weak_event_labels.csv --out-dir work/runs/run_0002/packs/obs_1800_step300_multiscale_hybrid_tree
```

当前环境未安装 torch，`train.pt`、`valid.pt`、`test.pt` 是 pickle stream 格式；读取方式见对应 `meta.json`。

下一步入口应优先补训练评估：CAC-Stat、Campaign-GNN、Temporal-GNN、DRAGEN-Full，以及 `w/o Tree`、`w/o MultiScale`、`w/o Role`、`w/o Memory`、`w/o Global Prior`、`w/o Adaptive Sampling`、`w/o Gate`、`w/o Uncertainty` 消融。

## 配置驱动训练

训练、消融、结果表和预测分析脚本现在支持：

```bash
--config configs/train/<name>.yaml
```

参数优先级：

```text
脚本默认值 < YAML 配置 < CLI 覆盖
```

DRAGEN-Full 正式训练：

```bash
python scripts/16_train_dragen_full.py \
  --config configs/train/dragen_full_run0002.yaml
```

Debug 训练：

```bash
python scripts/16_train_dragen_full.py \
  --config configs/train/dragen_full_debug.yaml
```

临时覆盖配置：

```bash
python scripts/16_train_dragen_full.py \
  --config configs/train/dragen_full_run0002.yaml \
  --seed 1 \
  --out-dir work/artifacts/dragen_full_run0002_seed1 \
  --no-tensorboard
```

消融训练：

```bash
python scripts/17_run_ablation.py --config configs/train/ablation_no_role.yaml
python scripts/17_run_ablation.py --config configs/train/ablation_no_memory.yaml
python scripts/17_run_ablation.py --config configs/train/ablation_no_global_prior.yaml
python scripts/17_run_ablation.py --config configs/train/ablation_no_adaptive_sampling.yaml
python scripts/17_run_ablation.py --config configs/train/ablation_no_gate.yaml
python scripts/17_run_ablation.py --config configs/train/ablation_no_uncertainty.yaml
```

`w/o Tree` 和 `w/o MultiScale` 只切换输入 pack，也可直接用训练脚本读取配置：

```bash
python scripts/16_train_dragen_full.py --config configs/train/ablation_no_tree.yaml
python scripts/16_train_dragen_full.py --config configs/train/ablation_no_multiscale.yaml
```

每次训练开始会写入：

```text
reports/resolved_config.yaml
reports/command.txt
reports/git_info.json
```

最终结果表：

```bash
python scripts/18_export_result_tables.py \
  --config configs/train/result_tables_run0002.yaml
```

训练后分析：

```bash
python scripts/19_analyze_predictions.py \
  --artifact-dir work/artifacts/dragen_full_run0002_seed0
```

## 文档

- [docs/window_design.md](docs/window_design.md)：窗口划分设计。
- [docs/data_schema.md](docs/data_schema.md)：核心数据表字段。
- [docs/evidence_features.md](docs/evidence_features.md)：多源证据特征设计。
- [docs/graph_design.md](docs/graph_design.md)：星形边与代理传播树设计。
- [docs/model_design.md](docs/model_design.md)：DRAGEN 模型边界和组件。
- [docs/experiment_protocol.md](docs/experiment_protocol.md)：实验流程约定。
- [docs/run_notes.md](docs/run_notes.md)：实验记录。
- [docs/results_summary.md](docs/results_summary.md)：当前实验摘要。
- [docs/server_experiment_guide.md](docs/server_experiment_guide.md)：服务器迁移与训练说明。
