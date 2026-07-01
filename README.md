# DRAGEN

DRAGEN 是一个用于级联预测实验的工程目录。当前目录的核心约定是：每批数据固定放在 `work/runs/<run_id>/`，命令入口放在 `scripts/`，真正的实现逻辑放在 `src/dragen/`。

当前阶段已经完成工程结构、数据边界和 30m/5m 窗口划分入口。特征构建、模型训练等模块还没有作为正式入口暴露，等窗口表契约稳定后再继续实现。

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

结构重构实验需要先构建代理传播树，再用树边重建窗口：

```powershell
python scripts/06_build_inferred_tree.py --run-id run_0002 --method branching_time --out-dir work/runs/run_0002/edges
python scripts/05_build_windows.py --run-id run_0002 --window-config configs/window/obs_30m_step5m.yaml --edge-mode inferred_tree
```

树形窗口默认输出到：

```text
work/runs/run_0002/windows/obs_1800_win300_step300_tree/
```

注意：`time_only` 会把大级联构造成接近线性的深链，只作为结构对照或反例。正式树形结构默认使用 `branching_time`，它会用时间接近、深度惩罚、父节点活跃度和分支负载共同选择父节点。

树形结构可视化入口：

```powershell
python scripts/07_visualize_tree.py --run-id run_0002 --cascade-id 78857 --tree-edges work/runs/run_0002/edges/_viz_cascade_78857_branching/inferred_tree_edge_table.csv --out work/runs/run_0002/edges/_viz_cascade_78857_branching/cascade_78857_branching_tree.svg --max-nodes 180
```

窗口策略现在有两个层次：

- `Fixed-5m`：当前已生成的 baseline，非重叠 5 分钟窗口。
- `Causal MultiScale`：端点对齐多尺度因果窗口，每 5 分钟一个端点，同时生成 current 5m、context 10m 和 cumulative 历史统计。

MultiScale 调试命令：

```powershell
python scripts/05_build_windows.py --run-id run_0002 --window-config configs/window/obs_30m_step5m_multiscale.yaml --edge-mode star --max-cascades 10 --out-dir work/runs/run_0002/windows/_debug_obs_1800_step300_multiscale_star
```

没有实现完成的特征、训练、评估入口暂时不保留，避免把空壳脚本当成可运行流程。

## 文档

- [docs/window_design.md](docs/window_design.md)：窗口划分设计。
- [docs/data_schema.md](docs/data_schema.md)：核心数据表字段。
- [docs/evidence_features.md](docs/evidence_features.md)：多源证据特征设计。
- [docs/graph_design.md](docs/graph_design.md)：星形边与代理传播树设计。
- [docs/model_design.md](docs/model_design.md)：DRAGEN 模型边界和组件。
- [docs/experiment_protocol.md](docs/experiment_protocol.md)：实验流程约定。
- [docs/run_notes.md](docs/run_notes.md)：实验记录。
