# LightWorld Codebase Restructure Plan (v2)

> 基于初版计划增强，增加风险评估、依赖分析、回滚策略与逐文件映射。

## 1. 设计目标

将 LightWorld 从"增长中的原型"升级为**稳定的研究系统仓库**：

- 清晰的顶层产品边界：`src/`、`configs/`、`data/`、`docs/`、`scripts/`、
  `tests/`、`examples/`
- 可安装的 Python 包 `lightworld`，稳定的公共入口
- 按职责命名的领域模块，非实现意外
- 统一的运行日志和中间产物结构
- Google Python Style Guide 命名规范
- 薄脚本和入口；编排逻辑在 Application Service 中

## 2. 目标顶层布局

```text
LightWorld/
  README.md
  LICENSE
  pyproject.toml              ← 从 backend/ 提升到根目录
  uv.lock
  .env.example
  .github/
    workflows/
  configs/
    full_run/
      whu_baike_event.json
    simulation/
      simulation.full.template.json
    local_pipeline/
      whu_baike_event.json
  data/
    examples/
      whu_baike_event/        ← 原 multimodal_inputs/baike_wuda_event
  docs/                       ← 保持现有 GitHub Pages 站点
  examples/
    whu_baike_event/
      README.md
  scripts/
    dev/
      run_api.sh
    data/
      fetch_baidu_baike_event.py
  src/
    lightworld/               ← 唯一可导入的产品包
      __init__.py
      api/
      application/
      config/
      domain/
      graph/
      ingestion/
      infrastructure/
      memory/
      reporting/
      simulation/
      storage/
      telemetry/
      tools/
      cli/
  tests/
    unit/
    integration/
    fixtures/
  tmp/
    .gitkeep
```

## 3. 逐文件迁移映射

### 3.1 核心包 (`backend/app/` → `src/lightworld/`)

| 当前路径 | 目标路径 | 说明 |
|---------|---------|------|
| `app/__init__.py` | `lightworld/__init__.py` | Flask app factory |
| `app/config.py` | `lightworld/config/__init__.py` | Config re-export |
| `app/setting/settings.py` | `lightworld/config/settings.py` | 主配置 |
| `app/setting/settings_local.example.py` | `lightworld/config/settings_local.example.py` | 示例 |
| `app/adapters/http/__init__.py` | `lightworld/api/__init__.py` | Blueprint 注册 |
| `app/adapters/http/graph.py` | `lightworld/api/routes/graph_routes.py` | Graph API |
| `app/adapters/http/simulation.py` | `lightworld/api/routes/simulation_routes.py` | Sim API |
| `app/adapters/http/report.py` | `lightworld/api/routes/report_routes.py` | Report API |
| `app/application/full_run_service.py` | `lightworld/application/full_run_service.py` | 全流程服务 |
| `app/application/simulation_preparation.py` | `lightworld/application/simulation_preparation_service.py` | 模拟准备 |
| `app/domain/project.py` | `lightworld/domain/project.py` | 项目模型 |
| `app/domain/task.py` | `lightworld/domain/task.py` | 任务模型 |
| `app/infrastructure/llm_client.py` | `lightworld/infrastructure/llm_client.py` | LLM 客户端 |
| `app/infrastructure/llm_client_factory.py` | `lightworld/infrastructure/llm_client_factory.py` | LLM 工厂 |
| `app/infrastructure/retry.py` | `lightworld/infrastructure/retry.py` | 重试装饰器 |
| `app/infrastructure/logger.py` | `lightworld/telemetry/logging_config.py` | 日志配置 |
| `app/infrastructure/file_parser.py` | `lightworld/ingestion/file_parser.py` | 文件解析 |
| `app/infrastructure/project_repository.py` | `lightworld/storage/project_repository.py` | 项目存储 |
| `app/infrastructure/report_repository.py` | `lightworld/storage/report_repository.py` | 报告存储 |
| `app/infrastructure/simulation_state_repository.py` | `lightworld/storage/simulation_state_repository.py` | 模拟状态 |
| `app/infrastructure/zep_paging.py` | `lightworld/memory/zep_paging.py` | Zep 分页 |
| `app/modules/graph/local_pipeline.py` | `lightworld/graph/local_graph_pipeline.py` | 本地管线 |
| `app/modules/simulation/cluster_cli.py` | `lightworld/simulation/cluster_cli.py` | 聚类 CLI |
| `app/modules/simulation/cluster_flags.py` | `lightworld/simulation/cluster_flags.py` | 聚类标志 |
| `app/modules/simulation/memory_keywords.py` | `lightworld/simulation/memory_keywords.py` | 记忆关键词 |
| `app/modules/simulation/platform_runner.py` | `lightworld/simulation/platform_runner.py` | 平台运行器 |
| `app/modules/simulation/runtimes.py` | `lightworld/simulation/runtime/topology_aware_runtime.py` + `simple_mem_runtime.py` | 拆分运行时 |
| `app/utils/graph_builder.py` | `lightworld/graph/graph_builder.py` | 图构建 |
| `app/utils/ontology_generator.py` | `lightworld/graph/ontology_generator.py` | 本体生成 |
| `app/utils/zep_entity_reader.py` | `lightworld/graph/zep_entity_reader.py` | Zep 实体读取 |
| `app/utils/zep_graph_memory_updater.py` | `lightworld/graph/zep_graph_memory_updater.py` | Zep 图记忆 |
| `app/utils/zep_tools.py` | `lightworld/graph/zep_tools.py` | Zep 工具 |
| `app/utils/social_relation_graph.py` | `lightworld/graph/social_relation_graph.py` | 社交关系图 |
| `app/utils/multimodal_ingestion.py` | `lightworld/ingestion/multimodal_ingestion.py` | 多模态摄入 |
| `app/utils/text_processor.py` | `lightworld/ingestion/text_processor.py` | 文本处理 |
| `app/utils/entity_prompt_extractor.py` | `lightworld/tools/entity_prompt_extractor.py` | 实体提示提取 |
| `app/utils/oasis_profile_generator.py` | `lightworld/simulation/oasis_profile_generator.py` | 画像生成 |
| `app/utils/simulation_config_generator.py` | `lightworld/simulation/simulation_config_generator.py` | 配置生成 |
| `app/utils/simulation_population.py` | `lightworld/simulation/simulation_population.py` | 人口准备 |
| `app/utils/simulation_manager.py` | `lightworld/simulation/simulation_manager.py` | 模拟管理 |
| `app/utils/simulation_runner.py` | `lightworld/simulation/simulation_runner.py` | 模拟运行器 |
| `app/utils/simulation_ipc.py` | `lightworld/simulation/simulation_ipc.py` | IPC |
| `app/utils/report_agent.py` | `lightworld/reporting/report_agent.py` | 报告代理 |

### 3.2 CLI 入口 (`app/run/` → `lightworld/cli/`)

| 当前路径 | 目标路径 |
|---------|---------|
| `app/run/api.py` | `lightworld/cli/api.py` |
| `app/run/full_run.py` | `lightworld/cli/full_run.py` |
| `app/run/local_pipeline.py` | `lightworld/cli/local_pipeline.py` |
| `app/run/parallel_simulation.py` | `lightworld/cli/parallel_simulation.py` |

### 3.3 脚本 (`backend/run_scripts/` → `scripts/`)

| 当前路径 | 目标路径 |
|---------|---------|
| `run_scripts/fetch_baidu_baike_event.py` | `scripts/data/fetch_baidu_baike_event.py` |
| `run_scripts/run_parallel_simulation.py` | `scripts/dev/run_parallel_simulation.py` |
| `run_scripts/action_logger.py` | `src/lightworld/simulation/action_logger.py` |
| `run_scripts/config_templates/*.json` | `configs/simulation/` + `configs/full_run/` |
| 其余 `run_scripts/run_*.py` | 已被 CLI 入口替代，归档或删除 |

### 3.4 测试 (`backend/tests/` + `backend/test_scripts/` → `tests/`)

| 当前路径 | 目标路径 |
|---------|---------|
| `tests/test_report_outline_structure.py` | `tests/unit/test_report_outline_structure.py` |
| `test_scripts/test_*.py` | `tests/integration/test_*.py`（改写为 pytest 风格）|

### 3.5 数据与配置

| 当前路径 | 目标路径 |
|---------|---------|
| `multimodal_inputs/baike_wuda_event/` | `data/examples/whu_baike_event/` |
| `event_inputs/baike_wuda_event/` | `data/generated/`（gitignored） |

## 4. Import 重写规则

全局替换规则（按优先级排序）：

```
app.setting.settings    → lightworld.config.settings
app.config              → lightworld.config
app.adapters.http       → lightworld.api
app.application         → lightworld.application
app.domain              → lightworld.domain
app.infrastructure.llm* → lightworld.infrastructure.*
app.infrastructure.logger     → lightworld.telemetry.logging_config
app.infrastructure.file_parser → lightworld.ingestion.file_parser
app.infrastructure.*_repository → lightworld.storage.*
app.infrastructure.zep_paging  → lightworld.memory.zep_paging
app.infrastructure.retry       → lightworld.infrastructure.retry
app.modules.graph       → lightworld.graph
app.modules.simulation  → lightworld.simulation
app.utils.graph_builder → lightworld.graph.graph_builder
app.utils.ontology_*    → lightworld.graph.ontology_*
app.utils.zep_*         → lightworld.graph.zep_*
app.utils.social_*      → lightworld.graph.social_*
app.utils.multimodal_*  → lightworld.ingestion.multimodal_*
app.utils.text_*        → lightworld.ingestion.text_*
app.utils.report_agent  → lightworld.reporting.report_agent
app.utils.simulation_*  → lightworld.simulation.*
app.utils.oasis_*       → lightworld.simulation.oasis_*
app.utils.entity_*      → lightworld.tools.entity_*
app.run                 → lightworld.cli
```

相对导入同步更新：`...config` → 按新目录层级调整。

## 5. 执行阶段

### Phase 1: 根目录打包与骨架

1. 将 `backend/pyproject.toml` 提升到仓库根目录
2. 修改 `[tool.hatch.build.targets.wheel]` packages 为 `["src/lightworld"]`
3. 修改 `[project.scripts]` 指向 `lightworld.cli.*`
4. 创建 `src/lightworld/` 完整目录骨架（空 `__init__.py`）
5. 验证：`uv sync` 成功

### Phase 2: 文件迁移

按上述映射表移动所有文件，暂不修改内容。

### Phase 3: Import 重写

1. 全局搜索替换 `from app.` → `from lightworld.`
2. 更新所有相对导入路径
3. 验证：所有 `*.py` 无 `from app.` 残留

### Phase 4: 配置、脚本、测试迁移

1. 移动配置模板到 `configs/`
2. 移动脚本到 `scripts/`
3. 移动测试到 `tests/`
4. 移动示例数据到 `data/examples/`

### Phase 5: 清理与验证

1. 删除 `backend/app/` 空壳
2. 更新 `.gitignore`
3. 更新 `README.md`
4. 验证所有 CLI 入口可用

## 6. 风险与回滚

| 风险 | 缓解措施 |
|------|---------|
| Import 遗漏导致运行时 ImportError | 每阶段后运行 `python -c "import lightworld"` |
| `runtimes.py` 拆分引入 bug | 暂不拆分，整体迁移为 `runtime/runtimes.py` |
| 配置路径硬编码 | `settings.py` 中使用 `__file__` 相对路径 |
| 第三方依赖路径假设 | 检查 OASIS/camel 的 cwd 依赖 |

**回滚**：每个 Phase 为独立 Git commit，可通过 `git revert` 回滚。
