# 武汉大学百科事件 示例

本示例演示如何使用 LightWorld 对武汉大学百度百科事件进行端到端的社会模拟与推演分析。

## 快速开始

```bash
# 1. 确保已安装依赖
cd /path/to/LightWorld
uv sync

# 2. 配置环境变量
cp .env.example .env
# 编辑 .env，填入 LLM_API_KEY 和 ZEP_API_KEY

# 3. 运行全流程
uv run lightworld-full-run --config configs/full_run/whu_baike_event.json
```

## 输入数据

示例输入数据位于 `data/examples/whu_baike_event/`，包含：

- 事件概述文档
- 事件时间线
- 相关图片和视频
- 数据来源清单

## 输出

运行结束后，产物将写入 `backend/uploads/full_runs/run_<timestamp>/` 目录，
包含 `run_manifest.json` 作为所有产物的索引入口。
