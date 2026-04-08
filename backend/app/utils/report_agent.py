
import os
import json
import time
import re
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from ..config import Config
from ..infrastructure.llm_client import LLMClient
from ..infrastructure.llm_client_factory import LLMClientFactory
from ..infrastructure.logger import get_logger
from ..infrastructure.report_repository import FileReportRepository
from .zep_tools import (
    ZepToolsService,
    SearchResult,
    InsightForgeResult,
    PanoramaResult,
    InterviewResult
)

logger = get_logger('lightworld.report_agent')


class ReportLogger:

    def __init__(self, report_id: str):
        self.report_id = report_id
        self.log_file_path = os.path.join(
            Config.REPORTS_DIR, report_id, 'agent_log.jsonl'
        )
        self.start_time = datetime.now()
        self._ensure_log_file()

    def _ensure_log_file(self):
        log_dir = os.path.dirname(self.log_file_path)
        os.makedirs(log_dir, exist_ok=True)

    def _get_elapsed_time(self) -> float:
        return (datetime.now() - self.start_time).total_seconds()

    def log(
        self,
        action: str,
        stage: str,
        details: Dict[str, Any],
        section_title: str = None,
        section_index: int = None
    ):
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "elapsed_seconds": round(self._get_elapsed_time(), 2),
            "report_id": self.report_id,
            "action": action,
            "stage": stage,
            "section_title": section_title,
            "section_index": section_index,
            "details": details
        }


        with open(self.log_file_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')

    def log_start(self, simulation_id: str, graph_id: str, simulation_requirement: str):
        self.log(
            action="report_start",
            stage="pending",
            details={
                "simulation_id": simulation_id,
                "graph_id": graph_id,
                "simulation_requirement": simulation_requirement,
                "message": "报告生成任务开始"
            }
        )

    def log_planning_start(self):
        self.log(
            action="planning_start",
            stage="planning",
            details={"message": "开始规划报告大纲"}
        )

    def log_planning_context(self, context: Dict[str, Any]):
        self.log(
            action="planning_context",
            stage="planning",
            details={
                "message": "获取模拟上下文信息",
                "context": context
            }
        )

    def log_planning_complete(self, outline_dict: Dict[str, Any]):
        self.log(
            action="planning_complete",
            stage="planning",
            details={
                "message": "大纲规划完成",
                "outline": outline_dict
            }
        )

    def log_section_start(self, section_title: str, section_index: int):
        self.log(
            action="section_start",
            stage="generating",
            section_title=section_title,
            section_index=section_index,
            details={"message": f"开始生成章节: {section_title}"}
        )

    def log_react_thought(self, section_title: str, section_index: int, iteration: int, thought: str):
        self.log(
            action="react_thought",
            stage="generating",
            section_title=section_title,
            section_index=section_index,
            details={
                "iteration": iteration,
                "thought": thought,
                "message": f"ReACT 第{iteration}轮思考"
            }
        )

    def log_tool_call(
        self,
        section_title: str,
        section_index: int,
        tool_name: str,
        parameters: Dict[str, Any],
        iteration: int
    ):
        self.log(
            action="tool_call",
            stage="generating",
            section_title=section_title,
            section_index=section_index,
            details={
                "iteration": iteration,
                "tool_name": tool_name,
                "parameters": parameters,
                "message": f"调用工具: {tool_name}"
            }
        )

    def log_tool_result(
        self,
        section_title: str,
        section_index: int,
        tool_name: str,
        result: str,
        iteration: int
    ):
        self.log(
            action="tool_result",
            stage="generating",
            section_title=section_title,
            section_index=section_index,
            details={
                "iteration": iteration,
                "tool_name": tool_name,
                "result": result,
                "result_length": len(result),
                "message": f"工具 {tool_name} 返回结果"
            }
        )

    def log_llm_response(
        self,
        section_title: str,
        section_index: int,
        response: str,
        iteration: int,
        has_tool_calls: bool,
        has_final_answer: bool
    ):
        self.log(
            action="llm_response",
            stage="generating",
            section_title=section_title,
            section_index=section_index,
            details={
                "iteration": iteration,
                "response": response,
                "response_length": len(response),
                "has_tool_calls": has_tool_calls,
                "has_final_answer": has_final_answer,
                "message": f"LLM 响应 (工具调用: {has_tool_calls}, 最终答案: {has_final_answer})"
            }
        )

    def log_section_content(
        self,
        section_title: str,
        section_index: int,
        content: str,
        tool_calls_count: int
    ):
        self.log(
            action="section_content",
            stage="generating",
            section_title=section_title,
            section_index=section_index,
            details={
                "content": content,
                "content_length": len(content),
                "tool_calls_count": tool_calls_count,
                "message": f"章节 {section_title} 内容生成完成"
            }
        )

    def log_section_full_complete(
        self,
        section_title: str,
        section_index: int,
        full_content: str
    ):
        self.log(
            action="section_complete",
            stage="generating",
            section_title=section_title,
            section_index=section_index,
            details={
                "content": full_content,
                "content_length": len(full_content),
                "message": f"章节 {section_title} 生成完成"
            }
        )

    def log_report_complete(self, total_sections: int, total_time_seconds: float):
        self.log(
            action="report_complete",
            stage="completed",
            details={
                "total_sections": total_sections,
                "total_time_seconds": round(total_time_seconds, 2),
                "message": "报告生成完成"
            }
        )

    def log_error(self, error_message: str, stage: str, section_title: str = None):
        self.log(
            action="error",
            stage=stage,
            section_title=section_title,
            section_index=None,
            details={
                "error": error_message,
                "message": f"发生错误: {error_message}"
            }
        )


class ReportConsoleLogger:

    def __init__(self, report_id: str):
        self.report_id = report_id
        self.log_file_path = os.path.join(
            Config.REPORTS_DIR, report_id, 'console_log.txt'
        )
        self._ensure_log_file()
        self._file_handler = None
        self._setup_file_handler()

    def _ensure_log_file(self):
        log_dir = os.path.dirname(self.log_file_path)
        os.makedirs(log_dir, exist_ok=True)

    def _setup_file_handler(self):
        import logging


        self._file_handler = logging.FileHandler(
            self.log_file_path,
            mode='a',
            encoding='utf-8'
        )
        self._file_handler.setLevel(logging.INFO)


        formatter = logging.Formatter(
            '[%(asctime)s] %(levelname)s: %(message)s',
            datefmt='%H:%M:%S'
        )
        self._file_handler.setFormatter(formatter)


        loggers_to_attach = [
            'lightworld.report_agent',
            'lightworld.zep_tools',
        ]

        for logger_name in loggers_to_attach:
            target_logger = logging.getLogger(logger_name)

            if self._file_handler not in target_logger.handlers:
                target_logger.addHandler(self._file_handler)

    def close(self):
        import logging

        if self._file_handler:
            loggers_to_detach = [
                'lightworld.report_agent',
                'lightworld.zep_tools',
            ]

            for logger_name in loggers_to_detach:
                target_logger = logging.getLogger(logger_name)
                if self._file_handler in target_logger.handlers:
                    target_logger.removeHandler(self._file_handler)

            self._file_handler.close()
            self._file_handler = None

    def __del__(self):
        self.close()


class ReportStatus(str, Enum):
    PENDING = "pending"
    PLANNING = "planning"
    GENERATING = "generating"
    COMPLETED = "completed"
    FAILED = "failed"


class ReportMode(str, Enum):
    PUBLIC = "public_report"
    TECHNICAL = "technical_report"


@dataclass
class ReportSection:
    title: str
    content: str = ""
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "content": self.content,
            "description": self.description,
        }

    def to_markdown(self, level: int = 2) -> str:
        md = f"{'#' * level} {self.title}\n\n"
        if self.content:
            md += f"{self.content}\n\n"
        return md


@dataclass
class ReportOutline:
    title: str
    summary: str
    sections: List[ReportSection]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "summary": self.summary,
            "sections": [s.to_dict() for s in self.sections]
        }

    def to_markdown(self) -> str:
        md = f"# {self.title}\n\n"
        md += f"> {self.summary}\n\n"
        for section in self.sections:
            md += section.to_markdown()
        return md


@dataclass
class Report:
    report_id: str
    simulation_id: str
    graph_id: str
    simulation_requirement: str
    report_mode: str
    status: ReportStatus
    outline: Optional[ReportOutline] = None
    markdown_content: str = ""
    created_at: str = ""
    completed_at: str = ""
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "report_id": self.report_id,
            "simulation_id": self.simulation_id,
            "graph_id": self.graph_id,
            "simulation_requirement": self.simulation_requirement,
            "report_mode": self.report_mode,
            "status": self.status.value,
            "outline": self.outline.to_dict() if self.outline else None,
            "markdown_content": self.markdown_content,
            "created_at": self.created_at,
            "completed_at": self.completed_at,
            "error": self.error
        }


TOOL_DESC_INSIGHT_FORGE = """\
【深度洞察检索 - 强大的检索工具】
这是我们强大的检索函数，专为深度分析设计。它会：
1. 自动将你的问题分解为多个子问题
2. 从多个维度检索模拟图谱中的信息
3. 整合语义搜索、实体分析、关系链追踪的结果
4. 返回最全面、最深度的检索内容

【使用场景】
- 需要深入分析某个话题
- 需要了解事件的多个方面
- 需要获取支撑报告章节的丰富素材

【返回内容】
- 相关事实原文（可直接引用）
- 核心实体洞察
- 关系链分析"""

TOOL_DESC_PANORAMA_SEARCH = """\
【广度搜索 - 获取全貌视图】
这个工具用于获取模拟结果的完整全貌，特别适合了解事件演变过程。它会：
1. 获取所有相关节点和关系
2. 区分当前有效的事实和历史/过期的事实
3. 帮助你了解舆情是如何演变的

【使用场景】
- 需要了解事件的完整发展脉络
- 需要对比不同阶段的舆情变化
- 需要获取全面的实体和关系信息

【返回内容】
- 当前有效事实（模拟最新结果）
- 历史/过期事实（演变记录）
- 所有涉及的实体"""

TOOL_DESC_QUICK_SEARCH = """\
【简单搜索 - 快速检索】
轻量级的快速检索工具，适合简单、直接的信息查询。

【使用场景】
- 需要快速查找某个具体信息
- 需要验证某个事实
- 简单的信息检索

【返回内容】
- 与查询最相关的事实列表"""

TOOL_DESC_INTERVIEW_AGENTS = """\
【深度采访 - 真实Agent采访（双平台）】
调用OASIS模拟环境的采访API，对正在运行的模拟Agent进行真实采访！
这不是LLM模拟，而是调用真实的采访接口获取模拟Agent的原始回答。
默认在Twitter和Reddit两个平台同时采访，获取更全面的观点。

功能流程：
1. 自动读取人设文件，了解所有模拟Agent
2. 智能选择与采访主题最相关的Agent（如学生、媒体、官方等）
3. 自动生成采访问题
4. 调用 /api/simulation/interview/batch 接口在双平台进行真实采访
5. 整合所有采访结果，提供多视角分析

【使用场景】
- 需要从不同角色视角了解事件看法（学生怎么看？媒体怎么看？官方怎么说？）
- 需要收集多方意见和立场
- 需要获取模拟Agent的真实回答（来自OASIS模拟环境）
- 想让报告更生动，包含"采访实录"

【返回内容】
- 被采访Agent的身份信息
- 各Agent在Twitter和Reddit两个平台的采访回答
- 关键引言（可直接引用）
- 采访摘要和观点对比

【重要】需要OASIS模拟环境正在运行才能使用此功能！"""

TOOL_DESC_REVIEW_SIMULATION_STATE = """\
【运行摘要 - 补充模拟后的行为与结构信号】
这个工具会读取 simulation 目录里的运行后摘要文件，用简洁方式返回：
- 动作分布与活跃Agent
- entity keywords 的代表样本
- 协同 unit、影响力差异等结构信号
- memory 写入与检索的概况

适合在这些场景中使用：
- 需要快速把握模拟整体走向
- 需要验证某个章节是否存在明显的群体协同、影响力不对称或记忆回流
- 需要从运行摘要中抓到进一步检索的线索

注意：
- 这是运行摘要，不替代图谱检索和采访
- 正文写作仍应优先使用检索结果和采访内容作为主要证据"""


PLAN_SYSTEM_PROMPT = """\
你是一个「社会模拟演化预测报告」的撰写专家，拥有对模拟世界的「上帝视角」——你可以洞察模拟中每一位Agent的行为、言论和互动。

【核心理念】
我们构建了一个模拟世界，并向其中注入了特定的「模拟需求」作为变量。模拟世界的演化结果，就是对未来可能发生情况的预测。你正在观察的不是"实验数据"，而是"未来的预演"。

【你的任务】
撰写一份「社会模拟演化预测报告」，回答：
1. 过去发生了什么，哪些历史事件与触发点把局势带入当前模拟场景？
2. 在模拟中，各类Agent（人群）是如何反应、扩散、协同、对抗并推动局势继续演化的？
3. 基于这场模拟，未来接下来最可能朝哪些方向继续发展？

【报告定位】
- ✅ 这是一份基于模拟的社会演化预测报告，揭示"过去如何走到这里，以及接下来会怎样"
- ✅ 聚焦于演化过程：事件起点、群体反应、关键转折、涌现现象、潜在风险、未来走向
- ✅ 模拟世界中的Agent言行就是对未来人群行为的预测
- ❌ 不是对现实世界现状的分析
- ❌ 不是泛泛而谈的舆情综述

【章节结构硬性要求】
- 报告必须覆盖以下三个层面，缺一不可：
  1. 过去事件回顾/演化起点
  2. 模拟中的关键演化信号（平台差异、群体行为、关键Agent言行、结构变化）
  3. 未来演化方向预测（下一阶段走向、分叉条件、风险或缓和因素）
- 最后一章优先用于输出未来预测，不能只停留在现状总结
- 如果需要增加第4章或第5章，也必须服务于以上三层主线

【章节数量限制】
- 最少3个章节，最多5个章节
- 不需要子章节，每个章节直接撰写完整内容
- 内容要精炼，聚焦于核心演化发现与未来判断
- 章节结构由你根据预测结果自主设计

请输出JSON格式的报告大纲，格式如下：
{
    "title": "报告标题",
    "summary": "报告摘要（一句话概括核心预测发现）",
    "sections": [
        {
            "title": "章节标题",
            "description": "章节内容描述"
        }
    ]
}

注意：sections数组最少3个，最多5个元素！"""

PLAN_USER_PROMPT_TEMPLATE = """\
【预测场景设定】
我们向模拟世界注入的变量（模拟需求）：{simulation_requirement}

【模拟世界规模】
- 参与模拟的实体数量: {total_nodes}
- 实体间产生的关系数量: {total_edges}
- 实体类型分布: {entity_types}
- 活跃Agent数量: {total_entities}

【模拟预测到的部分未来事实样本】
{related_facts_json}

【补充运行信号】
{simulation_artifact_digest}

请以「上帝视角」审视这个未来预演：
1. 过去发生了什么，哪些前置事件构成了后续演化的起点？
2. 在我们设定的条件下，模拟中的各类人群（Agent）是如何反应和行动的？
3. 这些模拟行为揭示了怎样的下一阶段演化方向？

根据预测结果，设计最合适的报告章节结构。

【强制要求】
- 报告必须显式覆盖“过去事件回顾”“模拟关键演化信息”“未来演化方向预测”三部分
- 最后一章最好直接回答：如果当前动能延续，接下来会怎么演化
- 可以增加风险或干预点章节，但不要替代未来预测章节

【再次提醒】报告章节数量：最少3个，最多5个，内容要精炼聚焦于核心预测发现。"""


SECTION_SYSTEM_PROMPT_TEMPLATE = """\
你是一个「社会模拟演化预测报告」的撰写专家，正在撰写报告的一个章节。

报告标题: {report_title}
报告摘要: {report_summary}
预测场景（模拟需求）: {simulation_requirement}

当前要撰写的章节: {section_title}
章节规划说明: {section_description}
章节写作焦点:
{section_focus_guidance}

运行后摘要（辅助线索，不可直接替代检索证据）:
{simulation_artifact_digest}

═══════════════════════════════════════════════════════════════
【核心理念】
═══════════════════════════════════════════════════════════════

模拟世界是对未来的预演。我们向模拟世界注入了特定条件（模拟需求），
模拟中Agent的行为和互动，就是对未来人群行为的预测。

你的任务是：
- 用一个清晰的社会演化视角来写，而不是散点式罗列事实
- 揭示在设定条件下，未来发生了什么
- 说明过去事件如何把局势带入当前模拟阶段
- 预测各类人群（Agent）是如何反应和行动的
- 发现值得关注的未来趋势、风险和机会

❌ 不要写成对现实世界现状的分析
✅ 要聚焦于"过去如何进入模拟、模拟如何演化、未来会怎样"——模拟结果就是预测的未来

═══════════════════════════════════════════════════════════════
【最重要的规则 - 必须遵守】
═══════════════════════════════════════════════════════════════

1. 【必须调用工具观察模拟世界】
   - 你正在以「上帝视角」观察未来的预演
   - 所有内容必须来自模拟世界中发生的事件和Agent言行
   - 禁止使用你自己的知识来编写报告内容
   - 每个章节至少调用3次工具（最多5次）来观察模拟的世界，它代表了未来
   - 上方的运行摘要只是帮助你发现线索，正文论证仍应主要依赖工具返回结果

2. 【必须引用Agent的原始言行】
   - Agent的发言和行为是对未来人群行为的预测
   - 在报告中使用引用格式展示这些预测，例如：
     > "某类人群会表示：原文内容..."
   - 这些引用是模拟预测的核心证据

3. 【语言一致性 - 引用内容必须翻译为报告语言】
   - 工具返回的内容可能包含英文或中英文混杂的表述
   - 如果模拟需求和材料原文是中文的，报告必须全部使用中文撰写
   - 当你引用工具返回的英文或中英混杂内容时，必须将其翻译为流畅的中文后再写入报告
   - 翻译时保持原意不变，确保表述自然通顺
   - 这一规则同时适用于正文和引用块（> 格式）中的内容

4. 【忠实呈现预测结果】
   - 报告内容必须反映模拟世界中的代表未来的模拟结果
   - 不要添加模拟中不存在的信息
   - 如果某方面信息不足，如实说明
   - 如果你在做未来判断，要明确这是基于已观察到的模拟结果做出的推断

═══════════════════════════════════════════════════════════════
【⚠️ 格式规范 - 极其重要！】
═══════════════════════════════════════════════════════════════

【一个章节 = 最小内容单位】
- 每个章节是报告的最小分块单位
- ❌ 禁止在章节内使用任何 Markdown 标题（#、##、###、#### 等）
- ❌ 禁止在内容开头添加章节主标题
- ✅ 章节标题由系统自动添加，你只需撰写纯正文内容
- ✅ 使用**粗体**、段落分隔、引用、列表来组织内容，但不要用标题

【正确示例】
```
本章节分析了事件的舆论传播态势。通过对模拟数据的深入分析，我们发现...

**首发引爆阶段**

微博作为舆情的第一现场，承担了信息首发的核心功能：

> "微博贡献了68%的首发声量..."

**情绪放大阶段**

抖音平台进一步放大了事件影响力：

- 视觉冲击力强
- 情绪共鸣度高
```

【错误示例】
```
## 执行摘要          ← 错误！不要添加任何标题
### 一、首发阶段     ← 错误！不要用###分小节
#### 1.1 详细分析   ← 错误！不要用####细分

本章节分析了...
```

═══════════════════════════════════════════════════════════════
【可用检索工具】（每章节调用3-5次）
═══════════════════════════════════════════════════════════════

{tools_description}

【工具使用建议 - 请混合使用不同工具，不要只用一种】
- insight_forge: 深度洞察分析，自动分解问题并多维度检索事实和关系
- panorama_search: 广角全景搜索，了解事件全貌、时间线和演变过程
- quick_search: 快速验证某个具体信息点
- interview_agents: 采访模拟Agent，获取不同角色的第一人称观点和真实反应
- review_simulation_state: 快速查看运行后动作、协同、影响力与记忆信号，再决定下一步深挖方向

═══════════════════════════════════════════════════════════════
【工作流程】
═══════════════════════════════════════════════════════════════

每次回复你只能做以下两件事之一（不可同时做）：

选项A - 调用工具：
输出你的思考，然后用以下格式调用一个工具：
<tool_call>
{{"name": "工具名称", "parameters": {{"参数名": "参数值"}}}}
</tool_call>
系统会执行工具并把结果返回给你。你不需要也不能自己编写工具返回结果。

选项B - 输出最终内容：
当你已通过工具获取了足够信息，以 "Final Answer:" 开头输出章节内容。

⚠️ 严格禁止：
- 禁止在一次回复中同时包含工具调用和 Final Answer
- 禁止自己编造工具返回结果（Observation），所有工具结果由系统注入
- 每次回复最多调用一个工具

═══════════════════════════════════════════════════════════════
【章节内容要求】
═══════════════════════════════════════════════════════════════

1. 内容必须基于工具检索到的模拟数据
2. 大量引用原文来展示模拟效果
3. 使用Markdown格式（但禁止使用标题）：
   - 使用 **粗体文字** 标记重点（代替子标题）
   - 使用列表（-或1.2.3.）组织要点
   - 使用空行分隔不同段落
   - ❌ 禁止使用 #、##、###、#### 等任何标题语法
4. 【引用格式规范 - 必须单独成段】
   引用必须独立成段，前后各有一个空行，不能混在段落中：

   ✅ 正确格式：
   ```
   校方的回应被认为缺乏实质内容。

   > "校方的应对模式在瞬息万变的社交媒体环境中显得僵化和迟缓。"

   这一评价反映了公众的普遍不满。
   ```

   ❌ 错误格式：
   ```
   校方的回应被认为缺乏实质内容。> "校方的应对模式..." 这一评价反映了...
   ```
5. 保持与其他章节的逻辑连贯性
6. 【避免重复】仔细阅读下方已完成的章节内容，不要重复描述相同的信息
7. 【再次强调】不要添加任何标题！用**粗体**代替小节标题
8. 【如果当前章节是预测章节】必须给出明确的后续演化判断，而不是只总结现状"""

SECTION_USER_PROMPT_TEMPLATE = """\
已完成的章节内容（请仔细阅读，避免重复）：
{previous_content}

═══════════════════════════════════════════════════════════════
【当前任务】撰写章节: {section_title}
═══════════════════════════════════════════════════════════════

【重要提醒】
1. 仔细阅读上方已完成的章节，避免重复相同的内容！
2. 开始前必须先调用工具获取模拟数据
3. 请混合使用不同工具，不要只用一种
4. 报告内容必须来自检索结果，不要使用自己的知识
5. 如果章节涉及群体协同、传播差异或认知变化，优先考虑先看一次运行摘要，再决定深挖什么

【⚠️ 格式警告 - 必须遵守】
- ❌ 不要写任何标题（#、##、###、####都不行）
- ❌ 不要写"{section_title}"作为开头
- ✅ 章节标题由系统自动添加
- ✅ 直接写正文，用**粗体**代替小节标题

请开始：
1. 首先思考（Thought）这个章节需要什么信息
2. 然后调用工具（Action）获取模拟数据
3. 收集足够信息后输出 Final Answer（纯正文，无任何标题）"""


REACT_OBSERVATION_TEMPLATE = """\
Observation（检索结果）:

═══ 工具 {tool_name} 返回 ═══
{result}

═══════════════════════════════════════════════════════════════
已调用工具 {tool_calls_count}/{max_tool_calls} 次（已用: {used_tools_str}）{unused_hint}
- 如果信息充分：以 "Final Answer:" 开头输出章节内容（必须引用上述原文）
- 如果需要更多信息：调用一个工具继续检索
═══════════════════════════════════════════════════════════════"""

REACT_INSUFFICIENT_TOOLS_MSG = (
    "【注意】你只调用了{tool_calls_count}次工具，至少需要{min_tool_calls}次。"
    "请再调用工具获取更多模拟数据，然后再输出 Final Answer。{unused_hint}"
)

REACT_INSUFFICIENT_TOOLS_MSG_ALT = (
    "当前只调用了 {tool_calls_count} 次工具，至少需要 {min_tool_calls} 次。"
    "请调用工具获取模拟数据。{unused_hint}"
)

REACT_TOOL_LIMIT_MSG = (
    "工具调用次数已达上限（{tool_calls_count}/{max_tool_calls}），不能再调用工具。"
    '请立即基于已获取的信息，以 "Final Answer:" 开头输出章节内容。'
)

REACT_UNUSED_TOOLS_HINT = "\n💡 你还没有使用过: {unused_list}，建议尝试不同工具获取多角度信息"

REACT_FORCE_FINAL_MSG = "已达到工具调用限制，请直接输出 Final Answer: 并生成章节内容。"


CHAT_SYSTEM_PROMPT_TEMPLATE = """\
你是一个简洁高效的模拟预测助手。

【背景】
预测条件: {simulation_requirement}

【已生成的分析报告】
{report_content}

【规则】
1. 优先基于上述报告内容回答问题
2. 直接回答问题，避免冗长的思考论述
3. 仅在报告内容不足以回答时，才调用工具检索更多数据
4. 回答要简洁、清晰、有条理

【可用工具】（仅在需要时使用，最多调用1-2次）
{tools_description}

【工具调用格式】
<tool_call>
{{"name": "工具名称", "parameters": {{"参数名": "参数值"}}}}
</tool_call>

【回答风格】
- 简洁直接，不要长篇大论
- 使用 > 格式引用关键内容
- 优先给出结论，再解释原因"""

CHAT_OBSERVATION_SUFFIX = "\n\n请简洁回答问题。"


class ReportAgent:


    MAX_TOOL_CALLS_PER_SECTION = 5


    MAX_REFLECTION_ROUNDS = 3


    MAX_TOOL_CALLS_PER_CHAT = 2

    HISTORY_SECTION_KEYWORDS = ("回顾", "背景", "起点", "前史", "过去", "来龙去脉", "源起")
    EVOLUTION_SECTION_KEYWORDS = ("演化", "传播", "扩散", "行动", "信号", "群体", "平台", "模拟", "协同", "博弈")
    FORECAST_SECTION_KEYWORDS = ("预测", "未来", "走向", "展望", "趋势", "情景", "分叉", "后续", "下一阶段")

    @staticmethod
    def _normalize_report_mode(report_mode: str) -> str:
        value = str(report_mode or "").strip().lower()
        if value == ReportMode.TECHNICAL.value:
            return ReportMode.TECHNICAL.value
        return ReportMode.PUBLIC.value

    @property
    def is_public_report(self) -> bool:
        return self.report_mode == ReportMode.PUBLIC.value

    @property
    def is_technical_report(self) -> bool:
        return self.report_mode == ReportMode.TECHNICAL.value

    def __init__(
        self,
        graph_id: str,
        simulation_id: str,
        simulation_requirement: str,
        report_mode: str = ReportMode.PUBLIC.value,
        llm_client: Optional[LLMClient] = None,
        zep_tools: Optional[ZepToolsService] = None
    ):
        self.graph_id = graph_id
        self.simulation_id = simulation_id
        self.simulation_requirement = simulation_requirement
        self.report_mode = self._normalize_report_mode(report_mode)

        self.llm = llm_client or LLMClientFactory.get_shared_client()
        self.zep_tools = zep_tools or ZepToolsService()
        self._simulation_context_cache: Optional[Dict[str, Any]] = None


        self.tools = self._define_tools()


        self.report_logger: Optional[ReportLogger] = None

        self.console_logger: Optional[ReportConsoleLogger] = None

        logger.info(
            f"ReportAgent 初始化完成: graph_id={graph_id}, simulation_id={simulation_id}, "
            f"report_mode={self.report_mode}"
        )

    def _define_tools(self) -> Dict[str, Dict[str, Any]]:
        tools = {
            "insight_forge": {
                "name": "insight_forge",
                "description": TOOL_DESC_INSIGHT_FORGE,
                "parameters": {
                    "query": "你想深入分析的问题或话题",
                    "report_context": "当前报告章节的上下文（可选，有助于生成更精准的子问题）"
                }
            },
            "panorama_search": {
                "name": "panorama_search",
                "description": TOOL_DESC_PANORAMA_SEARCH,
                "parameters": {
                    "query": "搜索查询，用于相关性排序",
                    "include_expired": "是否包含过期/历史内容（默认True）"
                }
            },
            "quick_search": {
                "name": "quick_search",
                "description": TOOL_DESC_QUICK_SEARCH,
                "parameters": {
                    "query": "搜索查询字符串",
                    "limit": "返回结果数量（可选，默认10）"
                }
            },
            "interview_agents": {
                "name": "interview_agents",
                "description": TOOL_DESC_INTERVIEW_AGENTS,
                "parameters": {
                    "interview_topic": "采访主题或需求描述（如：'了解学生对宿舍甲醛事件的看法'）",
                    "max_agents": "最多采访的Agent数量（可选，默认5，最大10）"
                }
            }
        }
        if self.is_technical_report:
            tools["review_simulation_state"] = {
                "name": "review_simulation_state",
                "description": TOOL_DESC_REVIEW_SIMULATION_STATE,
                "parameters": {
                    "focus": "想优先查看的角度（如：'动作分布'、'协同单元'、'影响力差异'、'记忆回收'），可选"
                }
            }
        return tools

    def _get_simulation_context(self) -> Dict[str, Any]:
        if self._simulation_context_cache is None:
            self._simulation_context_cache = self.zep_tools.get_simulation_context(
                graph_id=self.graph_id,
                simulation_requirement=self.simulation_requirement,
                simulation_id=self.simulation_id,
                report_mode=self.report_mode,
            )
        return self._simulation_context_cache

    def _execute_tool(self, tool_name: str, parameters: Dict[str, Any], report_context: str = "") -> str:
        logger.info(f"执行工具: {tool_name}, 参数: {parameters}")

        try:
            if tool_name == "insight_forge":
                query = parameters.get("query", "")
                ctx = parameters.get("report_context", "") or report_context
                result = self.zep_tools.insight_forge(
                    graph_id=self.graph_id,
                    query=query,
                    simulation_requirement=self.simulation_requirement,
                    report_context=ctx
                )
                return result.to_text()

            elif tool_name == "panorama_search":

                query = parameters.get("query", "")
                include_expired = parameters.get("include_expired", True)
                if isinstance(include_expired, str):
                    include_expired = include_expired.lower() in ['true', '1', 'yes']
                result = self.zep_tools.panorama_search(
                    graph_id=self.graph_id,
                    query=query,
                    include_expired=include_expired
                )
                return result.to_text()

            elif tool_name == "quick_search":

                query = parameters.get("query", "")
                limit = parameters.get("limit", 10)
                if isinstance(limit, str):
                    limit = int(limit)
                result = self.zep_tools.quick_search(
                    graph_id=self.graph_id,
                    query=query,
                    limit=limit
                )
                return result.to_text()

            elif tool_name == "interview_agents":

                interview_topic = parameters.get("interview_topic", parameters.get("query", ""))
                max_agents = parameters.get("max_agents", 5)
                if isinstance(max_agents, str):
                    max_agents = int(max_agents)
                max_agents = min(max_agents, 10)
                result = self.zep_tools.interview_agents(
                    simulation_id=self.simulation_id,
                    interview_requirement=interview_topic,
                    simulation_requirement=self.simulation_requirement,
                    max_agents=max_agents
                )
                return result.to_text()

            elif tool_name == "review_simulation_state":
                summary = self.zep_tools.get_simulation_artifact_summary(self.simulation_id)
                return summary.to_text(mode=self.report_mode)


            elif tool_name == "search_graph":

                logger.info("search_graph 已重定向到 quick_search")
                return self._execute_tool("quick_search", parameters, report_context)

            elif tool_name == "get_graph_statistics":
                result = self.zep_tools.get_graph_statistics(self.graph_id)
                return json.dumps(result, ensure_ascii=False, indent=2)

            elif tool_name == "get_entity_summary":
                entity_name = parameters.get("entity_name", "")
                result = self.zep_tools.get_entity_summary(
                    graph_id=self.graph_id,
                    entity_name=entity_name
                )
                return json.dumps(result, ensure_ascii=False, indent=2)

            elif tool_name == "get_simulation_context":

                logger.info("get_simulation_context 已重定向到 insight_forge")
                query = parameters.get("query", self.simulation_requirement)
                return self._execute_tool("insight_forge", {"query": query}, report_context)

            elif tool_name == "get_entities_by_type":
                entity_type = parameters.get("entity_type", "")
                nodes = self.zep_tools.get_entities_by_type(
                    graph_id=self.graph_id,
                    entity_type=entity_type
                )
                result = [n.to_dict() for n in nodes]
                return json.dumps(result, ensure_ascii=False, indent=2)

            else:
                return (
                    f"未知工具: {tool_name}。请使用以下工具之一: "
                    + ", ".join(self.tools.keys())
                )

        except Exception as e:
            logger.error(f"工具执行失败: {tool_name}, 错误: {str(e)}")
            return f"工具执行失败: {str(e)}"

    def _parse_tool_calls(self, response: str) -> List[Dict[str, Any]]:
        tool_calls = []


        xml_pattern = r'<tool_call>\s*(\{.*?\})\s*</tool_call>'
        for match in re.finditer(xml_pattern, response, re.DOTALL):
            try:
                call_data = json.loads(match.group(1))
                tool_calls.append(call_data)
            except json.JSONDecodeError:
                pass

        if tool_calls:
            return tool_calls


        stripped = response.strip()
        if stripped.startswith('{') and stripped.endswith('}'):
            try:
                call_data = json.loads(stripped)
                if self._is_valid_tool_call(call_data):
                    tool_calls.append(call_data)
                    return tool_calls
            except json.JSONDecodeError:
                pass


        json_pattern = r'(\{"(?:name|tool)"\s*:.*?\})\s*$'
        match = re.search(json_pattern, stripped, re.DOTALL)
        if match:
            try:
                call_data = json.loads(match.group(1))
                if self._is_valid_tool_call(call_data):
                    tool_calls.append(call_data)
            except json.JSONDecodeError:
                pass

        return tool_calls

    def _is_valid_tool_call(self, data: dict) -> bool:

        tool_name = data.get("name") or data.get("tool")
        if tool_name and tool_name in self.tools:

            if "tool" in data:
                data["name"] = data.pop("tool")
            if "params" in data and "parameters" not in data:
                data["parameters"] = data.pop("params")
            return True
        return False

    def _get_tools_description(self) -> str:
        desc_parts = ["可用工具："]
        for name, tool in self.tools.items():
            params_desc = ", ".join([f"{k}: {v}" for k, v in tool["parameters"].items()])
            desc_parts.append(f"- {name}: {tool['description']}")
            if params_desc:
                desc_parts.append(f"  参数: {params_desc}")
        return "\n".join(desc_parts)

    def _get_mode_prompt_block(self) -> str:
        if self.is_public_report:
            return """【写作模式：public_report】
- 目标读者是普通公众、媒体读者和非技术决策者。
- 正文禁止直接使用或解释以下术语：cluster、unit、PPR、ppr_centrality、topology、memory、retrieval、delta、agent memory。
- 如果内部机制确实影响结论，必须翻译成大众语言，例如：
  - unit/cluster -> 传播群体、协同行动群
  - PPR/centrality -> 核心放大节点、关键传播枢纽
  - memory retrieval -> 前期叙事被反复调用、旧说法持续影响后续讨论
- 重点写“发生了什么、谁在推动、公众会怎么理解、风险在哪里”，不要写成技术复盘文档。
- 可以引用模拟中的原话，但你自己的分析语言必须自然、通俗、面向大众。"""
        return """【写作模式：technical_report】
- 目标读者是研究人员、模型开发者和实验复盘人员。
- 可以直接使用 unit、PPR、memory、retrieval、delta 等术语。
- 允许在正文中解释结构信号、影响力差异和跨轮次记忆机制。"""

    def _get_public_tool_guidance(self) -> str:
        if self.is_public_report:
            return (
                "\n【公众版额外要求】\n"
                "- 优先使用 insight_forge、panorama_search、quick_search、interview_agents。\n"
                "- 不要为了写作而主动追求内部机制指标；即便看到技术信号，也要改写成自然语言。\n"
            )
        return ""

    @staticmethod
    def _contains_any_keyword(text: str, keywords: tuple[str, ...]) -> bool:
        title = str(text or "").strip()
        return any(keyword in title for keyword in keywords)

    @classmethod
    def _default_outline(cls) -> ReportOutline:
        return ReportOutline(
            title="社会模拟演化预测报告",
            summary="基于历史事件与社会模拟结果，概括事件如何演化，并判断下一阶段最可能出现的走向与风险。",
            sections=[
                ReportSection(
                    title="历史事件回顾与演化起点",
                    description="概括过去已经发生的关键事件、核心冲突与触发点，解释为什么局势会进入当前模拟。"
                ),
                ReportSection(
                    title="社会模拟中的关键演化信号",
                    description="提炼模拟里最重要的人群反应、平台差异、关键Agent言行与扩散/协同机制。"
                ),
                ReportSection(
                    title="未来演化方向与情景预测",
                    description="基于前述历史与模拟信号，判断下一阶段最可能的演化路径、触发条件、风险与缓和因素。"
                ),
            ]
        )

    @classmethod
    def _normalize_outline(cls, outline: ReportOutline) -> ReportOutline:
        cleaned_sections: List[ReportSection] = []
        seen_titles = set()
        for section in outline.sections or []:
            title = str(section.title or "").strip()
            if not title or title in seen_titles:
                continue
            seen_titles.add(title)
            cleaned_sections.append(
                ReportSection(
                    title=title,
                    content=section.content,
                    description=str(getattr(section, "description", "") or "").strip(),
                )
            )

        default_outline = cls._default_outline()
        if not cleaned_sections:
            return default_outline

        history_default, evolution_default, forecast_default = default_outline.sections
        history_section = next(
            (section for section in cleaned_sections if cls._contains_any_keyword(section.title, cls.HISTORY_SECTION_KEYWORDS)),
            None,
        )
        evolution_section = next(
            (section for section in cleaned_sections if cls._contains_any_keyword(section.title, cls.EVOLUTION_SECTION_KEYWORDS)),
            None,
        )
        forecast_section = next(
            (section for section in cleaned_sections if cls._contains_any_keyword(section.title, cls.FORECAST_SECTION_KEYWORDS)),
            None,
        )

        normalized_sections: List[ReportSection] = []

        def append_unique(section: ReportSection):
            if all(id(existing) != id(section) for existing in normalized_sections):
                normalized_sections.append(section)

        append_unique(history_section or history_default)
        append_unique(evolution_section or evolution_default)

        reserved_ids = {id(section) for section in normalized_sections}
        if forecast_section is not None:
            reserved_ids.add(id(forecast_section))

        extras = [
            section for section in cleaned_sections
            if id(section) not in reserved_ids
        ]

        normalized_sections.extend(extras[:2])
        normalized_sections.append(
            ReportSection(
                title=str((forecast_section or forecast_default).title or forecast_default.title).strip() or forecast_default.title,
                content=(forecast_section or forecast_default).content,
                description=str(getattr((forecast_section or forecast_default), "description", "") or "").strip() or forecast_default.description,
            )
        )

        title = str(outline.title or "").strip() or default_outline.title
        summary = str(outline.summary or "").strip() or default_outline.summary
        if "预测" not in title and "演化" not in title:
            title = f"{title}：社会模拟演化预测报告"

        return ReportOutline(
            title=title,
            summary=summary,
            sections=normalized_sections[:5]
        )

    @classmethod
    def _build_section_focus_guidance(
        cls,
        section: ReportSection,
        section_index: int,
        total_sections: int,
    ) -> str:
        title = str(section.title or "").strip()
        lines = [
            "- 优先围绕“事件如何演化”来组织材料，不要写成零散观点堆叠。",
            "- 必须同时交代关键主体、关键动作以及这些动作如何改变后续局势。",
        ]

        if cls._contains_any_keyword(title, cls.HISTORY_SECTION_KEYWORDS) or section_index == 1:
            lines.extend([
                "- 本章优先总结过去已经发生的事件、冲突起点和触发点，为后续模拟演化建立时间起点。",
                "- 不能只写背景介绍，还要说明这些前置事件为什么会引出后续舆论或社会反应。",
            ])
        elif cls._contains_any_keyword(title, cls.FORECAST_SECTION_KEYWORDS) or section_index == total_sections:
            lines.extend([
                "- 本章必须输出未来演化方向的判断，不能只停留在当前局势总结。",
                "- 至少写清楚一个最可能路径，并尽量补充一到两个可能分叉、触发条件或缓和因素。",
                "- 需要回答：哪些群体会继续放大、转向、对冲或降温，以及风险会如何累积或释放。",
            ])
        else:
            lines.extend([
                "- 本章优先提炼模拟中已经出现的关键演化信号，如平台差异、群体协同、立场迁移、影响力放大。",
                "- 重点引用具有代表性的Agent原话或行为，体现LLM模拟出的关键社会反应。",
            ])

        return "\n".join(lines)

    def plan_outline(
        self,
        progress_callback: Optional[Callable] = None
    ) -> ReportOutline:
        logger.info("开始规划报告大纲...")

        if progress_callback:
            progress_callback("planning", 0, "正在分析模拟需求...")


        context = self._get_simulation_context()

        if self.report_logger:
            self.report_logger.log_planning_context(context)

        if progress_callback:
            progress_callback("planning", 30, "正在生成报告大纲...")

        system_prompt = PLAN_SYSTEM_PROMPT + "\n\n" + self._get_mode_prompt_block()
        user_prompt = PLAN_USER_PROMPT_TEMPLATE.format(
            simulation_requirement=self.simulation_requirement,
            total_nodes=context.get('graph_statistics', {}).get('total_nodes', 0),
            total_edges=context.get('graph_statistics', {}).get('total_edges', 0),
            entity_types=list(context.get('graph_statistics', {}).get('entity_types', {}).keys()),
            total_entities=context.get('total_entities', 0),
            related_facts_json=json.dumps(context.get('related_facts', [])[:10], ensure_ascii=False, indent=2),
            simulation_artifact_digest="\n".join(
                f"- {line}" for line in (context.get("simulation_artifact_digest", []) or [])
            ) or "（暂无运行摘要）",
        )

        try:
            response = self.llm.chat_json(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3
            )

            if progress_callback:
                progress_callback("planning", 80, "正在解析大纲结构...")


            sections = []
            for section_data in response.get("sections", []):
                sections.append(ReportSection(
                    title=section_data.get("title", ""),
                    content="",
                    description=section_data.get("description", ""),
                ))

            outline = self._normalize_outline(ReportOutline(
                title=response.get("title", "模拟分析报告"),
                summary=response.get("summary", ""),
                sections=sections
            ))

            if progress_callback:
                progress_callback("planning", 100, "大纲规划完成")

            logger.info(f"大纲规划完成: {len(outline.sections)} 个章节")
            return outline

        except Exception as e:
            logger.error(f"大纲规划失败: {str(e)}")
            return self._default_outline()

    def _generate_section_react(
        self,
        section: ReportSection,
        outline: ReportOutline,
        previous_sections: List[str],
        progress_callback: Optional[Callable] = None,
        section_index: int = 0
    ) -> str:
        logger.info(f"ReACT生成章节: {section.title}")


        if self.report_logger:
            self.report_logger.log_section_start(section.title, section_index)

        system_prompt = SECTION_SYSTEM_PROMPT_TEMPLATE.format(
            report_title=outline.title,
            report_summary=outline.summary,
            simulation_requirement=self.simulation_requirement,
            section_title=section.title,
            section_description=section.description or "请围绕该章节在报告中的职责来组织内容。",
            section_focus_guidance=self._build_section_focus_guidance(
                section=section,
                section_index=section_index,
                total_sections=len(outline.sections),
            ),
            tools_description=self._get_tools_description(),
            simulation_artifact_digest="\n".join(
                f"- {line}" for line in (self._get_simulation_context().get("simulation_artifact_digest", []) or [])
            ) or "（暂无运行摘要）",
        ) + "\n\n" + self._get_mode_prompt_block() + self._get_public_tool_guidance()
        if self.is_public_report:
            system_prompt = system_prompt.replace(
                "- review_simulation_state: 快速查看运行后动作、协同、影响力与记忆信号，再决定下一步深挖方向\n",
                "",
            )


        if previous_sections:
            previous_parts = []
            for sec in previous_sections:

                truncated = sec[:4000] + "..." if len(sec) > 4000 else sec
                previous_parts.append(truncated)
            previous_content = "\n\n---\n\n".join(previous_parts)
        else:
            previous_content = "（这是第一个章节）"

        user_prompt = SECTION_USER_PROMPT_TEMPLATE.format(
            previous_content=previous_content,
            section_title=section.title,
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]


        tool_calls_count = 0
        max_iterations = 5
        min_tool_calls = 3
        conflict_retries = 0
        used_tools = set()
        all_tools = set(self.tools.keys())


        artifact_digest = self._get_simulation_context().get("simulation_artifact_digest", []) or []
        report_context = (
            f"章节标题: {section.title}\n"
            f"模拟需求: {self.simulation_requirement}\n"
            f"运行摘要: {' | '.join(artifact_digest[:6])}"
        )

        for iteration in range(max_iterations):
            if progress_callback:
                progress_callback(
                    "generating",
                    int((iteration / max_iterations) * 100),
                    f"深度检索与撰写中 ({tool_calls_count}/{self.MAX_TOOL_CALLS_PER_SECTION})"
                )


            response = self.llm.chat(
                messages=messages,
                temperature=0.5,
                max_tokens=4096
            )


            if response is None:
                logger.warning(f"章节 {section.title} 第 {iteration + 1} 次迭代: LLM 返回 None")

                if iteration < max_iterations - 1:
                    messages.append({"role": "assistant", "content": "（响应为空）"})
                    messages.append({"role": "user", "content": "请继续生成内容。"})
                    continue

                break

            logger.debug(f"LLM响应: {response[:200]}...")


            tool_calls = self._parse_tool_calls(response)
            has_tool_calls = bool(tool_calls)
            has_final_answer = "Final Answer:" in response


            if has_tool_calls and has_final_answer:
                conflict_retries += 1
                logger.warning(
                    f"章节 {section.title} 第 {iteration+1} 轮: "
                    f"LLM 同时输出工具调用和 Final Answer（第 {conflict_retries} 次冲突）"
                )

                if conflict_retries <= 2:

                    messages.append({"role": "assistant", "content": response})
                    messages.append({
                        "role": "user",
                        "content": (
                            "【格式错误】你在一次回复中同时包含了工具调用和 Final Answer，这是不允许的。\n"
                            "每次回复只能做以下两件事之一：\n"
                            "- 调用一个工具（输出一个 <tool_call> 块，不要写 Final Answer）\n"
                            "- 输出最终内容（以 'Final Answer:' 开头，不要包含 <tool_call>）\n"
                            "请重新回复，只做其中一件事。"
                        ),
                    })
                    continue
                else:

                    logger.warning(
                        f"章节 {section.title}: 连续 {conflict_retries} 次冲突，"
                        "降级为截断执行第一个工具调用"
                    )
                    first_tool_end = response.find('</tool_call>')
                    if first_tool_end != -1:
                        response = response[:first_tool_end + len('</tool_call>')]
                        tool_calls = self._parse_tool_calls(response)
                        has_tool_calls = bool(tool_calls)
                    has_final_answer = False
                    conflict_retries = 0


            if self.report_logger:
                self.report_logger.log_llm_response(
                    section_title=section.title,
                    section_index=section_index,
                    response=response,
                    iteration=iteration + 1,
                    has_tool_calls=has_tool_calls,
                    has_final_answer=has_final_answer
                )


            if has_final_answer:

                if tool_calls_count < min_tool_calls:
                    messages.append({"role": "assistant", "content": response})
                    unused_tools = all_tools - used_tools
                    unused_hint = f"（这些工具还未使用，推荐用一下他们: {', '.join(unused_tools)}）" if unused_tools else ""
                    messages.append({
                        "role": "user",
                        "content": REACT_INSUFFICIENT_TOOLS_MSG.format(
                            tool_calls_count=tool_calls_count,
                            min_tool_calls=min_tool_calls,
                            unused_hint=unused_hint,
                        ),
                    })
                    continue


                final_answer = response.split("Final Answer:")[-1].strip()
                logger.info(f"章节 {section.title} 生成完成（工具调用: {tool_calls_count}次）")

                if self.report_logger:
                    self.report_logger.log_section_content(
                        section_title=section.title,
                        section_index=section_index,
                        content=final_answer,
                        tool_calls_count=tool_calls_count
                    )
                return final_answer


            if has_tool_calls:

                if tool_calls_count >= self.MAX_TOOL_CALLS_PER_SECTION:
                    messages.append({"role": "assistant", "content": response})
                    messages.append({
                        "role": "user",
                        "content": REACT_TOOL_LIMIT_MSG.format(
                            tool_calls_count=tool_calls_count,
                            max_tool_calls=self.MAX_TOOL_CALLS_PER_SECTION,
                        ),
                    })
                    continue


                call = tool_calls[0]
                if len(tool_calls) > 1:
                    logger.info(f"LLM 尝试调用 {len(tool_calls)} 个工具，只执行第一个: {call['name']}")

                if self.report_logger:
                    self.report_logger.log_tool_call(
                        section_title=section.title,
                        section_index=section_index,
                        tool_name=call["name"],
                        parameters=call.get("parameters", {}),
                        iteration=iteration + 1
                    )

                result = self._execute_tool(
                    call["name"],
                    call.get("parameters", {}),
                    report_context=report_context
                )

                if self.report_logger:
                    self.report_logger.log_tool_result(
                        section_title=section.title,
                        section_index=section_index,
                        tool_name=call["name"],
                        result=result,
                        iteration=iteration + 1
                    )

                tool_calls_count += 1
                used_tools.add(call['name'])


                unused_tools = all_tools - used_tools
                unused_hint = ""
                if unused_tools and tool_calls_count < self.MAX_TOOL_CALLS_PER_SECTION:
                    unused_hint = REACT_UNUSED_TOOLS_HINT.format(unused_list="、".join(unused_tools))

                messages.append({"role": "assistant", "content": response})
                messages.append({
                    "role": "user",
                    "content": REACT_OBSERVATION_TEMPLATE.format(
                        tool_name=call["name"],
                        result=result,
                        tool_calls_count=tool_calls_count,
                        max_tool_calls=self.MAX_TOOL_CALLS_PER_SECTION,
                        used_tools_str=", ".join(used_tools),
                        unused_hint=unused_hint,
                    ),
                })
                continue


            messages.append({"role": "assistant", "content": response})

            if tool_calls_count < min_tool_calls:

                unused_tools = all_tools - used_tools
                unused_hint = f"（这些工具还未使用，推荐用一下他们: {', '.join(unused_tools)}）" if unused_tools else ""

                messages.append({
                    "role": "user",
                    "content": REACT_INSUFFICIENT_TOOLS_MSG_ALT.format(
                        tool_calls_count=tool_calls_count,
                        min_tool_calls=min_tool_calls,
                        unused_hint=unused_hint,
                    ),
                })
                continue


            logger.info(f"章节 {section.title} 未检测到 'Final Answer:' 前缀，直接采纳LLM输出作为最终内容（工具调用: {tool_calls_count}次）")
            final_answer = response.strip()

            if self.report_logger:
                self.report_logger.log_section_content(
                    section_title=section.title,
                    section_index=section_index,
                    content=final_answer,
                    tool_calls_count=tool_calls_count
                )
            return final_answer


        logger.warning(f"章节 {section.title} 达到最大迭代次数，强制生成")
        messages.append({"role": "user", "content": REACT_FORCE_FINAL_MSG})

        response = self.llm.chat(
            messages=messages,
            temperature=0.5,
            max_tokens=4096
        )


        if response is None:
            logger.error(f"章节 {section.title} 强制收尾时 LLM 返回 None，使用默认错误提示")
            final_answer = f"（本章节生成失败：LLM 返回空响应，请稍后重试）"
        elif "Final Answer:" in response:
            final_answer = response.split("Final Answer:")[-1].strip()
        else:
            final_answer = response


        if self.report_logger:
            self.report_logger.log_section_content(
                section_title=section.title,
                section_index=section_index,
                content=final_answer,
                tool_calls_count=tool_calls_count
            )

        return final_answer

    def generate_report(
        self,
        progress_callback: Optional[Callable[[str, int, str], None]] = None,
        report_id: Optional[str] = None
    ) -> Report:
        import uuid


        if not report_id:
            report_id = f"report_{uuid.uuid4().hex[:12]}"
        start_time = datetime.now()

        report = Report(
            report_id=report_id,
            simulation_id=self.simulation_id,
            graph_id=self.graph_id,
            simulation_requirement=self.simulation_requirement,
            report_mode=self.report_mode,
            status=ReportStatus.PENDING,
            created_at=datetime.now().isoformat()
        )


        completed_section_titles = []

        try:

            ReportManager._ensure_report_folder(report_id)


            self.report_logger = ReportLogger(report_id)
            self.report_logger.log_start(
                simulation_id=self.simulation_id,
                graph_id=self.graph_id,
                simulation_requirement=self.simulation_requirement
            )


            self.console_logger = ReportConsoleLogger(report_id)

            ReportManager.update_progress(
                report_id, "pending", 0, "初始化报告...",
                completed_sections=[]
            )
            ReportManager.save_report(report)


            report.status = ReportStatus.PLANNING
            ReportManager.update_progress(
                report_id, "planning", 5, "开始规划报告大纲...",
                completed_sections=[]
            )


            self.report_logger.log_planning_start()

            if progress_callback:
                progress_callback("planning", 0, "开始规划报告大纲...")

            outline = self.plan_outline(
                progress_callback=lambda stage, prog, msg:
                    progress_callback(stage, prog // 5, msg) if progress_callback else None
            )
            report.outline = outline


            self.report_logger.log_planning_complete(outline.to_dict())


            ReportManager.save_outline(report_id, outline)
            ReportManager.update_progress(
                report_id, "planning", 15, f"大纲规划完成，共{len(outline.sections)}个章节",
                completed_sections=[]
            )
            ReportManager.save_report(report)

            logger.info(f"大纲已保存到文件: {report_id}/outline.json")


            report.status = ReportStatus.GENERATING

            total_sections = len(outline.sections)
            generated_sections = []

            for i, section in enumerate(outline.sections):
                section_num = i + 1
                base_progress = 20 + int((i / total_sections) * 70)


                ReportManager.update_progress(
                    report_id, "generating", base_progress,
                    f"正在生成章节: {section.title} ({section_num}/{total_sections})",
                    current_section=section.title,
                    completed_sections=completed_section_titles
                )

                if progress_callback:
                    progress_callback(
                        "generating",
                        base_progress,
                        f"正在生成章节: {section.title} ({section_num}/{total_sections})"
                    )


                section_content = self._generate_section_react(
                    section=section,
                    outline=outline,
                    previous_sections=generated_sections,
                    progress_callback=lambda stage, prog, msg:
                        progress_callback(
                            stage,
                            base_progress + int(prog * 0.7 / total_sections),
                            msg
                        ) if progress_callback else None,
                    section_index=section_num
                )

                section.content = section_content
                generated_sections.append(f"## {section.title}\n\n{section_content}")


                ReportManager.save_section(report_id, section_num, section)
                completed_section_titles.append(section.title)


                full_section_content = f"## {section.title}\n\n{section_content}"

                if self.report_logger:
                    self.report_logger.log_section_full_complete(
                        section_title=section.title,
                        section_index=section_num,
                        full_content=full_section_content.strip()
                    )

                logger.info(f"章节已保存: {report_id}/section_{section_num:02d}.md")


                ReportManager.update_progress(
                    report_id, "generating",
                    base_progress + int(70 / total_sections),
                    f"章节 {section.title} 已完成",
                    current_section=None,
                    completed_sections=completed_section_titles
                )


            if progress_callback:
                progress_callback("generating", 95, "正在组装完整报告...")

            ReportManager.update_progress(
                report_id, "generating", 95, "正在组装完整报告...",
                completed_sections=completed_section_titles
            )


            report.markdown_content = ReportManager.assemble_full_report(report_id, outline)
            report.status = ReportStatus.COMPLETED
            report.completed_at = datetime.now().isoformat()


            total_time_seconds = (datetime.now() - start_time).total_seconds()


            if self.report_logger:
                self.report_logger.log_report_complete(
                    total_sections=total_sections,
                    total_time_seconds=total_time_seconds
                )


            ReportManager.save_report(report)
            ReportManager.update_progress(
                report_id, "completed", 100, "报告生成完成",
                completed_sections=completed_section_titles
            )

            if progress_callback:
                progress_callback("completed", 100, "报告生成完成")

            logger.info(f"报告生成完成: {report_id}")


            if self.console_logger:
                self.console_logger.close()
                self.console_logger = None

            return report

        except Exception as e:
            logger.error(f"报告生成失败: {str(e)}")
            report.status = ReportStatus.FAILED
            report.error = str(e)


            if self.report_logger:
                self.report_logger.log_error(str(e), "failed")


            try:
                ReportManager.save_report(report)
                ReportManager.update_progress(
                    report_id, "failed", -1, f"报告生成失败: {str(e)}",
                    completed_sections=completed_section_titles
                )
            except Exception:
                pass


            if self.console_logger:
                self.console_logger.close()
                self.console_logger = None

            return report

    def chat(
        self,
        message: str,
        chat_history: List[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        logger.info(f"Report Agent对话: {message[:50]}...")

        chat_history = chat_history or []


        report_content = ""
        try:
            report = ReportManager.get_report_by_simulation(self.simulation_id)
            if report and report.markdown_content:

                report_content = report.markdown_content[:15000]
                if len(report.markdown_content) > 15000:
                    report_content += "\n\n... [报告内容已截断] ..."
        except Exception as e:
            logger.warning(f"获取报告内容失败: {e}")

        system_prompt = CHAT_SYSTEM_PROMPT_TEMPLATE.format(
            simulation_requirement=self.simulation_requirement,
            report_content=report_content if report_content else "（暂无报告）",
            tools_description=self._get_tools_description(),
        ) + "\n\n" + self._get_mode_prompt_block()


        messages = [{"role": "system", "content": system_prompt}]


        for h in chat_history[-10:]:
            messages.append(h)


        messages.append({
            "role": "user",
            "content": message
        })


        tool_calls_made = []
        max_iterations = 2

        for iteration in range(max_iterations):
            response = self.llm.chat(
                messages=messages,
                temperature=0.5
            )


            tool_calls = self._parse_tool_calls(response)

            if not tool_calls:

                clean_response = re.sub(r'<tool_call>.*?</tool_call>', '', response, flags=re.DOTALL)
                clean_response = re.sub(r'\[TOOL_CALL\].*?\)', '', clean_response)

                return {
                    "response": clean_response.strip(),
                    "tool_calls": tool_calls_made,
                    "sources": [tc.get("parameters", {}).get("query", "") for tc in tool_calls_made]
                }


            tool_results = []
            for call in tool_calls[:1]:
                if len(tool_calls_made) >= self.MAX_TOOL_CALLS_PER_CHAT:
                    break
                result = self._execute_tool(call["name"], call.get("parameters", {}))
                tool_results.append({
                    "tool": call["name"],
                    "result": result[:1500]
                })
                tool_calls_made.append(call)


            messages.append({"role": "assistant", "content": response})
            observation = "\n".join([f"[{r['tool']}结果]\n{r['result']}" for r in tool_results])
            messages.append({
                "role": "user",
                "content": observation + CHAT_OBSERVATION_SUFFIX
            })


        final_response = self.llm.chat(
            messages=messages,
            temperature=0.5
        )


        clean_response = re.sub(r'<tool_call>.*?</tool_call>', '', final_response, flags=re.DOTALL)
        clean_response = re.sub(r'\[TOOL_CALL\].*?\)', '', clean_response)

        return {
            "response": clean_response.strip(),
            "tool_calls": tool_calls_made,
            "sources": [tc.get("parameters", {}).get("query", "") for tc in tool_calls_made]
        }


class ReportManager:


    REPORTS_DIR = Config.REPORTS_DIR
    _repository = FileReportRepository(REPORTS_DIR)

    @classmethod
    def _get_repository(cls) -> FileReportRepository:
        if cls._repository.reports_dir != cls.REPORTS_DIR:
            cls._repository = FileReportRepository(cls.REPORTS_DIR)
        return cls._repository

    @classmethod
    def _ensure_reports_dir(cls):
        cls._get_repository().ensure_reports_dir()

    @classmethod
    def _get_report_folder(cls, report_id: str) -> str:
        return cls._get_repository().get_report_folder(report_id)

    @classmethod
    def _ensure_report_folder(cls, report_id: str) -> str:
        return cls._get_repository().get_report_folder(report_id, create=True)

    @classmethod
    def _get_report_path(cls, report_id: str) -> str:
        return cls._get_repository().get_report_path(report_id)

    @classmethod
    def _get_report_markdown_path(cls, report_id: str) -> str:
        return cls._get_repository().get_report_markdown_path(report_id)

    @classmethod
    def _get_outline_path(cls, report_id: str) -> str:
        return cls._get_repository().get_outline_path(report_id)

    @classmethod
    def _get_progress_path(cls, report_id: str) -> str:
        return cls._get_repository().get_progress_path(report_id)

    @classmethod
    def _get_section_path(cls, report_id: str, section_index: int) -> str:
        return cls._get_repository().get_section_path(report_id, section_index)

    @classmethod
    def _get_agent_log_path(cls, report_id: str) -> str:
        return cls._get_repository().get_agent_log_path(report_id)

    @classmethod
    def _get_console_log_path(cls, report_id: str) -> str:
        return cls._get_repository().get_console_log_path(report_id)

    @classmethod
    def get_console_log(cls, report_id: str, from_line: int = 0) -> Dict[str, Any]:
        return cls._get_repository().read_text_lines(
            cls._get_console_log_path(report_id),
            from_line=from_line,
        )

    @classmethod
    def get_console_log_stream(cls, report_id: str) -> List[str]:
        result = cls.get_console_log(report_id, from_line=0)
        return result["logs"]

    @classmethod
    def get_agent_log(cls, report_id: str, from_line: int = 0) -> Dict[str, Any]:
        return cls._get_repository().read_jsonl_lines(
            cls._get_agent_log_path(report_id),
            from_line=from_line,
        )

    @classmethod
    def get_agent_log_stream(cls, report_id: str) -> List[Dict[str, Any]]:
        result = cls.get_agent_log(report_id, from_line=0)
        return result["logs"]

    @classmethod
    def save_outline(cls, report_id: str, outline: ReportOutline) -> None:
        cls._get_repository().save_outline_payload(report_id, outline.to_dict())

        logger.info(f"大纲已保存: {report_id}")

    @classmethod
    def save_section(
        cls,
        report_id: str,
        section_index: int,
        section: ReportSection
    ) -> str:
        cls._ensure_report_folder(report_id)


        cleaned_content = cls._clean_section_content(section.content, section.title)
        md_content = f"## {section.title}\n\n"
        if cleaned_content:
            md_content += f"{cleaned_content}\n\n"


        file_path = cls._get_repository().save_section_markdown(report_id, section_index, md_content)
        logger.info(f"章节已保存: {report_id}/{os.path.basename(file_path)}")
        return file_path

    @classmethod
    def _clean_section_content(cls, content: str, section_title: str) -> str:
        import re

        if not content:
            return content

        content = content.strip()
        lines = content.split('\n')
        cleaned_lines = []
        skip_next_empty = False

        for i, line in enumerate(lines):
            stripped = line.strip()


            heading_match = re.match(r'^(#{1,6})\s+(.+)$', stripped)

            if heading_match:
                level = len(heading_match.group(1))
                title_text = heading_match.group(2).strip()


                if i < 5:
                    if title_text == section_title or title_text.replace(' ', '') == section_title.replace(' ', ''):
                        skip_next_empty = True
                        continue


                cleaned_lines.append(f"**{title_text}**")
                cleaned_lines.append("")
                continue


            if skip_next_empty and stripped == '':
                skip_next_empty = False
                continue

            skip_next_empty = False
            cleaned_lines.append(line)


        while cleaned_lines and cleaned_lines[0].strip() == '':
            cleaned_lines.pop(0)


        while cleaned_lines and cleaned_lines[0].strip() in ['---', '***', '___']:
            cleaned_lines.pop(0)

            while cleaned_lines and cleaned_lines[0].strip() == '':
                cleaned_lines.pop(0)

        return '\n'.join(cleaned_lines)

    @classmethod
    def update_progress(
        cls,
        report_id: str,
        status: str,
        progress: int,
        message: str,
        current_section: str = None,
        completed_sections: List[str] = None
    ) -> None:
        progress_data = {
            "status": status,
            "progress": progress,
            "message": message,
            "current_section": current_section,
            "completed_sections": completed_sections or [],
            "updated_at": datetime.now().isoformat()
        }

        cls._get_repository().save_progress_payload(report_id, progress_data)

    @classmethod
    def get_progress(cls, report_id: str) -> Optional[Dict[str, Any]]:
        return cls._get_repository().load_progress_payload(report_id)

    @classmethod
    def get_generated_sections(cls, report_id: str) -> List[Dict[str, Any]]:
        return cls._get_repository().load_generated_sections(report_id)

    @classmethod
    def assemble_full_report(cls, report_id: str, outline: ReportOutline) -> str:
        md_content = f"# {outline.title}\n\n"
        md_content += f"> {outline.summary}\n\n"
        md_content += f"---\n\n"


        sections = cls.get_generated_sections(report_id)
        for section_info in sections:
            md_content += section_info["content"]


        md_content = cls._post_process_report(md_content, outline)


        cls._get_repository().save_full_report_markdown(report_id, md_content)

        logger.info(f"完整报告已组装: {report_id}")
        return md_content

    @classmethod
    def _post_process_report(cls, content: str, outline: ReportOutline) -> str:
        import re

        lines = content.split('\n')
        processed_lines = []
        prev_was_heading = False


        section_titles = set()
        for section in outline.sections:
            section_titles.add(section.title)

        i = 0
        while i < len(lines):
            line = lines[i]
            stripped = line.strip()


            heading_match = re.match(r'^(#{1,6})\s+(.+)$', stripped)

            if heading_match:
                level = len(heading_match.group(1))
                title = heading_match.group(2).strip()


                is_duplicate = False
                for j in range(max(0, len(processed_lines) - 5), len(processed_lines)):
                    prev_line = processed_lines[j].strip()
                    prev_match = re.match(r'^(#{1,6})\s+(.+)$', prev_line)
                    if prev_match:
                        prev_title = prev_match.group(2).strip()
                        if prev_title == title:
                            is_duplicate = True
                            break

                if is_duplicate:

                    i += 1
                    while i < len(lines) and lines[i].strip() == '':
                        i += 1
                    continue


                if level == 1:
                    if title == outline.title:

                        processed_lines.append(line)
                        prev_was_heading = True
                    elif title in section_titles:

                        processed_lines.append(f"## {title}")
                        prev_was_heading = True
                    else:

                        processed_lines.append(f"**{title}**")
                        processed_lines.append("")
                        prev_was_heading = False
                elif level == 2:
                    if title in section_titles or title == outline.title:

                        processed_lines.append(line)
                        prev_was_heading = True
                    else:

                        processed_lines.append(f"**{title}**")
                        processed_lines.append("")
                        prev_was_heading = False
                else:

                    processed_lines.append(f"**{title}**")
                    processed_lines.append("")
                    prev_was_heading = False

                i += 1
                continue

            elif stripped == '---' and prev_was_heading:

                i += 1
                continue

            elif stripped == '' and prev_was_heading:

                if processed_lines and processed_lines[-1].strip() != '':
                    processed_lines.append(line)
                prev_was_heading = False

            else:
                processed_lines.append(line)
                prev_was_heading = False

            i += 1


        result_lines = []
        empty_count = 0
        for line in processed_lines:
            if line.strip() == '':
                empty_count += 1
                if empty_count <= 2:
                    result_lines.append(line)
            else:
                empty_count = 0
                result_lines.append(line)

        return '\n'.join(result_lines)

    @classmethod
    def save_report(cls, report: Report) -> None:
        cls._get_repository().save_report_payload(report.report_id, report.to_dict())


        if report.outline:
            cls.save_outline(report.report_id, report.outline)


        if report.markdown_content:
            cls._get_repository().save_full_report_markdown(report.report_id, report.markdown_content)

        logger.info(f"报告已保存: {report.report_id}")

    @classmethod
    def get_report(cls, report_id: str) -> Optional[Report]:
        data = cls._get_repository().load_report_payload(report_id)
        if data is None:
            return None


        outline = None
        if data.get('outline'):
            outline_data = data['outline']
            sections = []
            for s in outline_data.get('sections', []):
                sections.append(ReportSection(
                    title=s['title'],
                    content=s.get('content', ''),
                    description=s.get('description', '')
                ))
            outline = ReportOutline(
                title=outline_data['title'],
                summary=outline_data['summary'],
                sections=sections
            )


        markdown_content = data.get('markdown_content', '') or ""
        if not markdown_content:
            markdown_content = cls._get_repository().load_report_markdown(report_id) or ""

        return Report(
            report_id=data['report_id'],
            simulation_id=data['simulation_id'],
            graph_id=data['graph_id'],
            simulation_requirement=data['simulation_requirement'],
            report_mode=data.get('report_mode', ReportMode.PUBLIC.value),
            status=ReportStatus(data['status']),
            outline=outline,
            markdown_content=markdown_content,
            created_at=data.get('created_at', ''),
            completed_at=data.get('completed_at', ''),
            error=data.get('error')
        )

    @classmethod
    def get_report_by_simulation(cls, simulation_id: str) -> Optional[Report]:
        cls._ensure_reports_dir()

        for report_id in cls._get_repository().list_report_ids():
            report = cls.get_report(report_id)
            if report and report.simulation_id == simulation_id:
                return report

        return None

    @classmethod
    def list_reports(cls, simulation_id: Optional[str] = None, limit: int = 50) -> List[Report]:
        cls._ensure_reports_dir()

        reports = []
        for report_id in cls._get_repository().list_report_ids():
            report = cls.get_report(report_id)
            if report and (simulation_id is None or report.simulation_id == simulation_id):
                reports.append(report)


        reports.sort(key=lambda r: r.created_at, reverse=True)

        return reports[:limit]

    @classmethod
    def delete_report(cls, report_id: str) -> bool:
        deleted = cls._get_repository().delete_report(report_id)
        if deleted:
            logger.info(f"报告已删除: {report_id}")
        return deleted
