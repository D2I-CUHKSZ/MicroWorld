
import os
import json
import time
import re
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from lightworld.config.settings import Config
from lightworld.infrastructure.llm_client import LLMClient
from lightworld.infrastructure.llm_client_factory import LLMClientFactory
from lightworld.telemetry.logging_config import get_logger
from lightworld.storage.report_repository import FileReportRepository
from lightworld.graph.zep_tools import (
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
                "message": "Report generation task started"
            }
        )

    def log_planning_start(self):
        self.log(
            action="planning_start",
            stage="planning",
            details={"message": "Starting report outline planning"}
        )

    def log_planning_context(self, context: Dict[str, Any]):
        self.log(
            action="planning_context",
            stage="planning",
            details={
                "message": "Fetching simulation context",
                "context": context
            }
        )

    def log_planning_complete(self, outline_dict: Dict[str, Any]):
        self.log(
            action="planning_complete",
            stage="planning",
            details={
                "message": "Outline planning complete",
                "outline": outline_dict
            }
        )

    def log_section_start(self, section_title: str, section_index: int):
        self.log(
            action="section_start",
            stage="generating",
            section_title=section_title,
            section_index=section_index,
            details={"message": f"Starting section generation: {section_title}"}
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
                "message": f"ReACT round {iteration} thinking"
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
                "message": f"Calling tool: {tool_name}"
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
                "message": f"Tool {tool_name} returned result"
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
                "message": f"LLM response (tool_calls: {has_tool_calls}, final_answer: {has_final_answer})"
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
                "message": f"Section {section_title} content generation complete"
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
                "message": f"Section {section_title} generation complete"
            }
        )

    def log_report_complete(self, total_sections: int, total_time_seconds: float):
        self.log(
            action="report_complete",
            stage="completed",
            details={
                "total_sections": total_sections,
                "total_time_seconds": round(total_time_seconds, 2),
                "message": "Report generation complete"
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
                "message": f"Error occurred: {error_message}"
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

[Deep Insight Retrieval - Powerful retrieval tool]
A powerful retrieval function designed for deep analysis. It will:
1. Automatically decompose your question into sub-questions
2. Retrieve information from multiple dimensions in the simulation graph
3. Integrate results from semantic search, entity analysis, and relationship chain tracking
4. Return the most comprehensive, in-depth retrieval content

[Use Cases]
- Need in-depth analysis of a topic
- Need to understand multiple aspects of an event
- Need rich material to support report sections

[Returns]
- Related factual text (can be directly quoted)
- Core entity insights
- Relationship chain analysis"""

TOOL_DESC_PANORAMA_SEARCH = """\

[Panorama Search - Full picture view]
This tool retrieves the complete picture of simulation results, especially suitable for understanding event evolution. It will:
1. Retrieve all related nodes and relationships
2. Distinguish between current valid facts and historical/expired facts
3. Help you understand how public opinion evolved

[Use Cases]
- Need to understand the complete development timeline of events
- Need to compare opinion changes across different stages
- Need comprehensive entity and relationship information

[Returns]
- Current valid facts (latest simulation results)
- Historical/expired facts (evolution records)
- All involved entities"""

TOOL_DESC_QUICK_SEARCH = """\

[Quick Search - Fast retrieval]
Lightweight fast retrieval tool, suitable for simple, direct information queries.

[Use Cases]
- Need to quickly look up specific information
- Need to verify a fact
- Simple information retrieval

[Returns]
- List of facts most relevant to the query"""

TOOL_DESC_INTERVIEW_AGENTS = """\

[Deep Interview - Real Agent interviews (dual-platform)]
Calls the OASIS simulation environment interview API to conduct real interviews with running simulation agents!
This is not LLM simulation, but real interview API calls to get original agent responses.
By default interviews on both Twitter and Reddit platforms for more comprehensive perspectives.

Workflow:
1. Automatically reads persona files to understand all simulation agents
2. Intelligently selects agents most relevant to the interview topic (e.g. students, media, officials)
3. Automatically generates interview questions
4. Calls /api/simulation/interview/batch for dual-platform real interviews
5. Integrates all interview results for multi-perspective analysis

[Use Cases]
- Need different role perspectives on events (What do students think? Media? Officials?)
- Need to collect opinions and stances from multiple parties
- Need real agent responses (from OASIS simulation environment)
- Want to make reports more vivid with "interview transcripts"

[Returns]
- Identity info of interviewed agents
- Each agent's interview responses on both Twitter and Reddit
- Key quotes (can be directly cited)
- Interview summary and viewpoint comparison

[Important] Requires OASIS simulation environment to be running!"""

TOOL_DESC_REVIEW_SIMULATION_STATE = """\

[Run Summary - Supplementary post-simulation behavior and structural signals]
This tool reads post-run summary files from the simulation directory, returning concisely:
- Action distribution and active agents
- Representative samples of entity keywords
- Structural signals such as coordination units and influence differences
- Overview of memory writes and retrievals

Suitable for these scenarios:
- Need to quickly grasp the overall simulation direction
- Need to verify if a section shows obvious group coordination, influence asymmetry, or memory feedback
- Need to find clues for further retrieval from the run summary

Note:
- This is a run summary, not a substitute for graph retrieval and interviews
- Report writing should still primarily use retrieval results and interview content as main evidence"""


PLAN_SYSTEM_PROMPT = """\

You are an expert writer of "Social Simulation Evolution Prediction Reports", with a "god's-eye view" of the simulated world -- you can observe every agent's behavior, statements, and interactions.

[Core Concept]
We built a simulated world and injected a specific "simulation requirement" as a variable. The evolution results of the simulated world serve as predictions of what may happen in the future. You are observing not "experimental data", but a "rehearsal of the future".

[Your Task]
Write a "Social Simulation Evolution Prediction Report" that answers:
1. What happened in the past, and which historical events and triggers brought the situation into the current simulation scenario?
2. In the simulation, how did various agent groups react, spread, coordinate, counter, and drive the situation to continue evolving?
3. Based on this simulation, what are the most likely directions for future development?

[Report Positioning]
- ✅ This is a simulation-based social evolution prediction report, revealing "how the past led here, and what comes next"
- ✅ Focus on the evolution process: event origins, group reactions, key turning points, emergent phenomena, potential risks, future directions
- ✅ Agent behavior in the simulated world represents predictions of future crowd behavior
- ❌ Not an analysis of real-world current status
- ❌ Not a generic opinion survey summary

[Mandatory Section Structure]
- The report must cover these three layers, none can be omitted:
  1. Past event review / evolution starting point
  2. Key evolution signals in the simulation (platform differences, group behavior, key agent actions, structural changes)
  3. Future evolution direction predictions (next-stage trajectory, branching conditions, risk or mitigation factors)
- The last chapter should prioritize future predictions, not just summarize the current state
- If adding chapter 4 or 5, they must serve these three main threads

[Section Count Limit]
- Minimum 3 sections, maximum 5 sections
- No sub-sections needed, each section contains complete content directly
- Content should be concise, focused on core evolution findings and future judgments
- Section structure is designed by you based on prediction results

Please output the report outline in JSON format as follows:
{
    "title": "report title",
    "summary": "report summary (one sentence summarizing core prediction findings)",
    "sections": [
        {
            "title": "section title",
            "description": "section content description"
        }
    ]
}

Note: sections array must have minimum 3 and maximum 5 elements!"""

PLAN_USER_PROMPT_TEMPLATE = """\

[Prediction Scenario]
Variable injected into the simulated world (simulation requirement):{simulation_requirement}

[Simulation World Scale]
- Number of simulated entities: {total_nodes}
- Number of relationships between entities: {total_edges}
- Entity type distribution: {entity_types}
- Number of active agents: {total_entities}

[Sample of Future Facts Predicted by Simulation]
{related_facts_json}

[Supplementary Run Signals]
{simulation_artifact_digest}

Review this future rehearsal from a "god's-eye view":
1. What happened in the past, and which precursor events formed the starting point for subsequent evolution?
2. Under our set conditions, how did the various groups (agents) in the simulation react and act?
3. What next-stage evolution directions do these simulated behaviors reveal?

Design the most appropriate report section structure based on prediction results.

[Mandatory Requirements]
- The report must explicitly cover "past event review", "key simulation evolution signals", and "future evolution direction prediction"
- The last chapter should ideally directly answer: if current momentum continues, how will things evolve next
- Risk or intervention chapters can be added, but must not replace the future prediction chapter

[Reminder] Report section count: minimum 3, maximum 5, content should be concise and focused on core prediction findings."""


SECTION_SYSTEM_PROMPT_TEMPLATE = """\

You are an expert writer of "Social Simulation Evolution Prediction Reports", currently writing one chapter.

Report title: {report_title}
Report summary: {report_summary}
Prediction scenario (simulation requirement): {simulation_requirement}

Current chapter to write: {section_title}
Section planning notes: {section_description}
Section writing focus:
{section_focus_guidance}

Post-run summary (auxiliary clues, cannot directly substitute for retrieval evidence):
{simulation_artifact_digest}

═══════════════════════════════════════════════════════════════
[Core Concept]
═══════════════════════════════════════════════════════════════

The simulated world is a rehearsal of the future. We injected specific conditions (simulation requirements),
and the behavior and interactions of agents in the simulation represent predictions of future crowd behavior.

Your task is:
- Write from a clear social evolution perspective, not scattered fact listing
- Reveal what happened in the future under set conditions
- Explain how past events brought the situation into the current simulation stage
- Predict how various groups (agents) reacted and acted
- Identify noteworthy future trends, risks, and opportunities

❌ Do not write as an analysis of real-world current status
✅ Focus on "how the past led into the simulation, how the simulation evolved, what the future holds" -- simulation results represent the predicted future

═══════════════════════════════════════════════════════════════
[Most Important Rules - Must Follow]
═══════════════════════════════════════════════════════════════

1. [Must call tools to observe the simulated world]
   - You are observing the future rehearsal from a "god's-eye view"
   - All content must come from events and agent behavior in the simulated world
   - Do not use your own knowledge to write report content
   - Each chapter must call tools at least 3 times (max 5) to observe the simulated world, which represents the future
   - The run summary above only helps you find clues; main text arguments should primarily rely on tool results

2. [Must quote agents' original words and actions]
   - Agent statements and behaviors are predictions of future crowd behavior
   - Use quote format in the report to display these predictions, e.g.:
     > "A certain group would say: original content..."
   - These quotes are core evidence of simulation predictions

3. [Language consistency - quoted content must be translated to report language]
   - Tool-returned content may contain English or mixed Chinese-English expressions
   - If the simulation requirement and source material are in Chinese, the report must be entirely in Chinese
   - When quoting English or mixed content from tools, translate it into fluent Chinese before writing into the report
   - Maintain original meaning when translating, ensure natural and smooth expression
   - This rule applies to both body text and quote blocks (> format)

4. [Faithfully present prediction results]
   - Report content must reflect simulation results representing the future in the simulated world
   - Do not add information that does not exist in the simulation
   - If information is insufficient in some aspect, state it honestly
   - If making future judgments, clearly state this is an inference based on observed simulation results

═══════════════════════════════════════════════════════════════
[Format Guidelines - Extremely Important!]
═══════════════════════════════════════════════════════════════

[One Chapter = Minimum Content Unit]
- Each chapter is the minimum section unit of the report
- ❌ Do not use any Markdown headings (#, ##, ###, #### etc.) within a chapter
- ❌ Do not add a chapter title at the beginning of content
- ✅ Chapter titles are added automatically by the system, you only need to write body text
- ✅ Use **bold**, paragraph breaks, quotes, lists to organize content, but no headings

[Correct Example]
```
This chapter analyzes the event's opinion dissemination dynamics. Through deep analysis of simulation data, we found...

**Initial Detonation Phase**

Platform A served as the frontline for information release, taking on the core function of first-mover dissemination:

> "Platform A contributed 68% of the initial voice volume..."

**Emotion Amplification Phase**

Platform B further amplified the event's impact:

- Strong visual impact
- High emotional resonance
```

[Wrong Example]
```
## Executive Summary          <- Wrong! Do not add any headings
### 1. Initial Phase     <- Wrong! Do not use ### for sub-sections
#### 1.1 Detailed Analysis   <- Wrong! Do not use #### for sub-sections

This chapter analyzes...
```

═══════════════════════════════════════════════════════════════
[Available Retrieval Tools] (call 3-5 times per chapter)
═══════════════════════════════════════════════════════════════

{tools_description}

[Tool Usage Tips - Mix different tools, do not use only one]
- insight_forge: Deep insight analysis, automatically decomposes questions and retrieves facts and relationships from multiple dimensions
- panorama_search: Wide-angle panoramic search, understand the full picture, timeline, and evolution of events
- quick_search: Quickly verify a specific information point
- interview_agents: Interview simulation agents, get first-person perspectives and real reactions from different roles
- review_simulation_state: Quickly review post-run actions, coordination, influence, and memory signals to decide what to dig into next

═══════════════════════════════════════════════════════════════
[Workflow]
═══════════════════════════════════════════════════════════════

Each reply you can only do one of the following two things (not both):

Option A - Call a tool:
Output your thinking, then call a tool in the following format:
<tool_call>
{{"name": "tool_name", "parameters": {{"param_name": "param_value"}}}}
</tool_call>
The system will execute the tool and return results. You do not need to and cannot write tool results yourself.

Option B - Output final content:
When you have gathered sufficient information through tools, output chapter content starting with "Final Answer:".

Strictly forbidden:
- Do not include both tool calls and Final Answer in one reply
- Do not fabricate tool results (Observation), all tool results are injected by the system
- Each reply may call at most one tool

═══════════════════════════════════════════════════════════════
[Chapter Content Requirements]
═══════════════════════════════════════════════════════════════

1. Content must be based on simulation data retrieved by tools
2. Extensively quote original text to demonstrate simulation effects
3. Use Markdown format (but headings are forbidden):
   - Use **bold text** to mark key points (instead of sub-headings)
   - Use lists (- or 1.2.3.) to organize points
   - Use blank lines to separate paragraphs
   - ❌ Do not use #, ##, ###, #### or any heading syntax
4. [Quote Format - Must be standalone paragraphs]
   Quotes must be standalone paragraphs with a blank line before and after, not mixed into paragraphs:

   ✅ Correct format:
   ```
   The response was considered lacking substance.

   > "The response pattern appeared rigid and slow in the fast-changing social media environment."

   This assessment reflects widespread public dissatisfaction.
   ```

   ❌ Wrong format:
   ```
   The response was considered lacking. > "The response pattern..." This assessment...
   ```
5. Maintain logical coherence with other chapters
6. [Avoid Repetition] Carefully read the completed chapters below, do not repeat the same information
7. [Emphasis] Do not add any headings! Use **bold** instead of sub-headings
8. [If this is a prediction chapter] Must provide clear future evolution judgments, not just summarize current status"""

SECTION_USER_PROMPT_TEMPLATE = """\

Completed chapter content (read carefully to avoid repetition):
{previous_content}

═══════════════════════════════════════════════════════════════
[Current Task] Write chapter: {section_title}
═══════════════════════════════════════════════════════════════

[Important Reminders]
1. Carefully read completed chapters above, avoid repeating the same content!
2. Must call tools to get simulation data before starting
3. Mix different tools, do not use only one
4. Report content must come from retrieval results, do not use your own knowledge
5. If the chapter involves group coordination, dissemination differences, or cognitive changes, consider reviewing the run summary first, then decide what to dig into

[Format Warning - Must Follow]
- ❌ Do not write any headings (#, ##, ###, #### are all forbidden)
- ❌ Do not write "{section_title}" as the beginning
- ✅ Chapter title is added automatically by the system
- ✅ Write body text directly, use **bold** instead of sub-headings

Please begin:
1. First think (Thought) about what information this chapter needs
2. Then call tools (Action) to get simulation data
3. After collecting sufficient information, output Final Answer (pure body text, no headings)"""


REACT_OBSERVATION_TEMPLATE = """\

Observation (retrieval results):

=== Tool {tool_name} returned ===
{result}

═══════════════════════════════════════════════════════════════
Tools called {tool_calls_count}/{max_tool_calls} times (used: {used_tools_str}){unused_hint}
- If information is sufficient: output chapter content starting with "Final Answer:" (must quote the above text)
- If more information is needed: call a tool to continue retrieval
═══════════════════════════════════════════════════════════════"""

REACT_INSUFFICIENT_TOOLS_MSG = (
    "[Note] You have only called tools {tool_calls_count} times, at least {min_tool_calls} required. "
    "Please call more tools to get more simulation data before outputting Final Answer. {unused_hint}"
)

REACT_INSUFFICIENT_TOOLS_MSG_ALT = (
    "Currently only {tool_calls_count} tool calls made, at least {min_tool_calls} required. "
    "Please call tools to get simulation data. {unused_hint}"
)

REACT_TOOL_LIMIT_MSG = (
    "Tool call limit reached ({tool_calls_count}/{max_tool_calls}), no more tool calls allowed. "
    'Please immediately output chapter content starting with "Final Answer:" based on the information gathered.'
)

REACT_UNUSED_TOOLS_HINT = "\n💡 You haven't used: {unused_list} yet, try different tools for multi-angle info"

REACT_FORCE_FINAL_MSG = "Tool call limit reached, please output Final Answer: and generate chapter content now."


CHAT_SYSTEM_PROMPT_TEMPLATE = """\

You are a concise and efficient simulation prediction assistant.

[Background]
Prediction conditions: {simulation_requirement}

[Generated Analysis Report]
{report_content}

[Rules]
1. Prioritize answering based on the report content above
2. Answer directly, avoid lengthy reasoning
3. Only call tools for more data when the report is insufficient
4. Answers should be concise, clear, and organized

[Available Tools] (use only when needed, max 1-2 calls)
{tools_description}

[Tool Call Format]
<tool_call>
{{"name": "tool_name", "parameters": {{"param_name": "param_value"}}}}
</tool_call>

[Response Style]
- Concise and direct, no lengthy essays
- Use > format to quote key content
- Give conclusions first, then explain reasons"""

CHAT_OBSERVATION_SUFFIX = "\n\nPlease answer concisely."


class ReportAgent:


    MAX_TOOL_CALLS_PER_SECTION = 5


    MAX_REFLECTION_ROUNDS = 3


    MAX_TOOL_CALLS_PER_CHAT = 2

    HISTORY_SECTION_KEYWORDS = ("review", "background", "origin", "history", "past", "context", "genesis")
    EVOLUTION_SECTION_KEYWORDS = ("evolution", "dissemination", "spread", "action", "signal", "group", "platform", "simulation", "coordination", "dynamics")
    FORECAST_SECTION_KEYWORDS = ("prediction", "future", "direction", "outlook", "trend", "scenario", "branching", "follow-up", "next phase")

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
            f"ReportAgent initialized: graph_id={graph_id}, simulation_id={simulation_id}, "
            f"report_mode={self.report_mode}"
        )

    def _define_tools(self) -> Dict[str, Dict[str, Any]]:
        tools = {
            "insight_forge": {
                "name": "insight_forge",
                "description": TOOL_DESC_INSIGHT_FORGE,
                "parameters": {
                    "query": "Question or topic you want to deeply analyze",
                    "report_context": "Current report chapter context (optional, helps generate more precise sub-questions)"
                }
            },
            "panorama_search": {
                "name": "panorama_search",
                "description": TOOL_DESC_PANORAMA_SEARCH,
                "parameters": {
                    "query": "Search query for relevance ranking",
                    "include_expired": "Whether to include expired/historical content (default True)"
                }
            },
            "quick_search": {
                "name": "quick_search",
                "description": TOOL_DESC_QUICK_SEARCH,
                "parameters": {
                    "query": "Search query string",
                    "limit": "Number of results to return (optional, default 10)"
                }
            },
            "interview_agents": {
                "name": "interview_agents",
                "description": TOOL_DESC_INTERVIEW_AGENTS,
                "parameters": {
                    "interview_topic": "Interview topic or requirement description (e.g. 'understand students\' views on dormitory formaldehyde incident')",
                    "max_agents": "Maximum number of agents to interview (optional, default 5, max 10)"
                }
            }
        }
        if self.is_technical_report:
            tools["review_simulation_state"] = {
                "name": "review_simulation_state",
                "description": TOOL_DESC_REVIEW_SIMULATION_STATE,
                "parameters": {
                    "focus": "Preferred viewing angle (e.g. 'action distribution', 'coordination units', 'influence differences', 'memory retrieval'), optional"
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
        logger.info(f"Executing tool: {tool_name}, params: {parameters}")

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

                logger.info("search_graph redirected to quick_search")
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

                logger.info("get_simulation_context redirected to insight_forge")
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
                    f"Unknown tool: {tool_name}. Please use one of: "
                    + ", ".join(self.tools.keys())
                )

        except Exception as e:
            logger.error(f"Tool execution failed: {tool_name}, error: {str(e)}")
            return f"Tool execution failed: {str(e)}"

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
        desc_parts = ["Available tools:"]
        for name, tool in self.tools.items():
            params_desc = ", ".join([f"{k}: {v}" for k, v in tool["parameters"].items()])
            desc_parts.append(f"- {name}: {tool['description']}")
            if params_desc:
                desc_parts.append(f"  Parameters: {params_desc}")
        return "\n".join(desc_parts)

    def _get_mode_prompt_block(self) -> str:
        if self.is_public_report:
            return """[Writing Mode: public_report]
- Target readers are general public, media readers, and non-technical decision makers.
- Body text must not directly use or explain these terms: cluster, unit, PPR, ppr_centrality, topology, memory, retrieval, delta, agent memory.
- If internal mechanisms do affect conclusions, translate them into plain language, e.g.:
  - unit/cluster -> dissemination groups, coordinated action clusters
  - PPR/centrality -> core amplification nodes, key dissemination hubs
  - memory retrieval -> earlier narratives repeatedly referenced, old narratives continue to influence subsequent discussion
- Focus on "what happened, who is driving it, how the public perceives it, where the risks are", not a technical review.
- You may quote original agent words, but your own analysis language must be natural, accessible, and audience-facing."""
        return """[Writing Mode: technical_report]
- Target readers are researchers, model developers, and experiment reviewers.
- May directly use terms like unit, PPR, memory, retrieval, delta.
- Allowed to explain structural signals, influence differences, and cross-round memory mechanisms in body text."""

    def _get_public_tool_guidance(self) -> str:
        if self.is_public_report:
            return (
                "\n[Public Version Additional Requirements]\n"
                "- Prioritize using insight_forge, panorama_search, quick_search, interview_agents.\n"
                "- Do not actively pursue internal mechanism metrics for writing; even if you see technical signals, rewrite them in natural language.\n"
            )
        return ""

    @staticmethod
    def _contains_any_keyword(text: str, keywords: tuple[str, ...]) -> bool:
        title = str(text or "").strip()
        return any(keyword in title for keyword in keywords)

    @classmethod
    def _default_outline(cls) -> ReportOutline:
        return ReportOutline(
            title="Social Simulation Evolution Prediction Report",
            summary="Based on historical events and social simulation results, summarizes how events evolved and predicts the most likely direction and risks for the next stage.",
            sections=[
                ReportSection(
                    title="Historical Event Review and Evolution Starting Point",
                    description="Summarize key events, core conflicts, and triggers that have occurred, explaining why the situation entered the current simulation."
                ),
                ReportSection(
                    title="Key Evolution Signals in Social Simulation",
                    description="Distill the most important crowd reactions, platform differences, key agent behaviors, and dissemination/coordination mechanisms from the simulation."
                ),
                ReportSection(
                    title="Future Evolution Direction and Scenario Prediction",
                    description="Based on the aforementioned history and simulation signals, predict the most likely evolution paths, trigger conditions, risks, and mitigation factors for the next stage."
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
        if "prediction" not in title.lower() and "evolution" not in title.lower():
            title = f"{title}: Social Simulation Evolution Prediction Report"

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
            "- Prioritize organizing material around \"how events evolved\", not scattered opinion stacking.",
            "- Must address key actors, key actions, and how those actions changed subsequent dynamics.",
        ]

        if cls._contains_any_keyword(title, cls.HISTORY_SECTION_KEYWORDS) or section_index == 1:
            lines.extend([
                "- This chapter should prioritize summarizing past events, conflict origins, and triggers, establishing a timeline starting point for subsequent simulation evolution.",
                "- Do not just write background introductions; also explain why these precursor events led to subsequent public opinion or social reactions.",
            ])
        elif cls._contains_any_keyword(title, cls.FORECAST_SECTION_KEYWORDS) or section_index == total_sections:
            lines.extend([
                "- This chapter must output judgments on future evolution directions, not just summarize the current situation.",
                "- At least clearly describe one most likely path, and try to supplement one or two possible branches, trigger conditions, or mitigation factors.",
                "- Must answer: which groups will continue to amplify, pivot, counter, or cool down, and how risks will accumulate or release.",
            ])
        else:
            lines.extend([
                "- This chapter should prioritize distilling key evolution signals that have appeared in the simulation, such as platform differences, group coordination, stance migration, and influence amplification.",
                "- Focus on quoting representative agent original words or behaviors, reflecting key social reactions produced by the LLM simulation.",
            ])

        return "\n".join(lines)

    def plan_outline(
        self,
        progress_callback: Optional[Callable] = None
    ) -> ReportOutline:
        logger.info("Starting report outline planning...")

        if progress_callback:
            progress_callback("planning", 0, "Analyzing simulation requirements...")


        context = self._get_simulation_context()

        if self.report_logger:
            self.report_logger.log_planning_context(context)

        if progress_callback:
            progress_callback("planning", 30, "Generating report outline...")

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
            ) or "(No run summary available)",
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
                progress_callback("planning", 80, "Parsing outline structure...")


            sections = []
            for section_data in response.get("sections", []):
                sections.append(ReportSection(
                    title=section_data.get("title", ""),
                    content="",
                    description=section_data.get("description", ""),
                ))

            outline = self._normalize_outline(ReportOutline(
                title=response.get("title", "Simulation Analysis Report"),
                summary=response.get("summary", ""),
                sections=sections
            ))

            if progress_callback:
                progress_callback("planning", 100, "Outline planning complete")

            logger.info(f"Outline planning complete: {len(outline.sections)} sections")
            return outline

        except Exception as e:
            logger.error(f"Outline planning failed: {str(e)}")
            return self._default_outline()

    def _generate_section_react(
        self,
        section: ReportSection,
        outline: ReportOutline,
        previous_sections: List[str],
        progress_callback: Optional[Callable] = None,
        section_index: int = 0
    ) -> str:
        logger.info(f"ReACT generating section: {section.title}")


        if self.report_logger:
            self.report_logger.log_section_start(section.title, section_index)

        system_prompt = SECTION_SYSTEM_PROMPT_TEMPLATE.format(
            report_title=outline.title,
            report_summary=outline.summary,
            simulation_requirement=self.simulation_requirement,
            section_title=section.title,
            section_description=section.description or "Organize content around this chapter's role in the report.",
            section_focus_guidance=self._build_section_focus_guidance(
                section=section,
                section_index=section_index,
                total_sections=len(outline.sections),
            ),
            tools_description=self._get_tools_description(),
            simulation_artifact_digest="\n".join(
                f"- {line}" for line in (self._get_simulation_context().get("simulation_artifact_digest", []) or [])
            ) or "(No run summary available)",
        ) + "\n\n" + self._get_mode_prompt_block() + self._get_public_tool_guidance()
        if self.is_public_report:
            system_prompt = system_prompt.replace(
                "- review_simulation_state: Quickly review post-run actions, coordination, influence and memory signals to decide next steps\n",
                "",
            )


        if previous_sections:
            previous_parts = []
            for sec in previous_sections:

                truncated = sec[:4000] + "..." if len(sec) > 4000 else sec
                previous_parts.append(truncated)
            previous_content = "\n\n---\n\n".join(previous_parts)
        else:
            previous_content = "(This is the first chapter)"

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
            f"Section title: {section.title}\n"
            f"Simulation requirement: {self.simulation_requirement}\n"
            f"Run summary: {' | '.join(artifact_digest[:6])}"
        )

        for iteration in range(max_iterations):
            if progress_callback:
                progress_callback(
                    "generating",
                    int((iteration / max_iterations) * 100),
                    f"Deep retrieval and writing ({tool_calls_count}/{self.MAX_TOOL_CALLS_PER_SECTION})"
                )


            response = self.llm.chat(
                messages=messages,
                temperature=0.5,
                max_tokens=4096
            )


            if response is None:
                logger.warning(f"Section {section.title} iteration {iteration + 1}: LLM returned None")

                if iteration < max_iterations - 1:
                    messages.append({"role": "assistant", "content": "(Empty response)"})
                    messages.append({"role": "user", "content": "Please continue generating content."})
                    continue

                break

            logger.debug(f"LLM response: {response[:200]}...")


            tool_calls = self._parse_tool_calls(response)
            has_tool_calls = bool(tool_calls)
            has_final_answer = "Final Answer:" in response


            if has_tool_calls and has_final_answer:
                conflict_retries += 1
                logger.warning(
                    f"Section {section.title} round {iteration+1}: "
                    f"LLM output both tool call and Final Answer (conflict #{conflict_retries})"
                )

                if conflict_retries <= 2:

                    messages.append({"role": "assistant", "content": response})
                    messages.append({
                        "role": "user",
                        "content": (
                            "[Format Error] You included both a tool call and Final Answer in one reply, which is not allowed.\n"
                            "Each reply can only do one of the following:\n"
                            "- Call a tool (output a <tool_call> block, do not write Final Answer)\n"
                            "- Output final content (start with 'Final Answer:', do not include <tool_call>)\n"
                            "Please reply again, doing only one of these."
                        ),
                    })
                    continue
                else:

                    logger.warning(
                        f"Section {section.title}: {conflict_retries} consecutive conflicts, "
                        "falling back to truncated execution of first tool call"
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
                    unused_hint = f"(These tools haven't been used yet, try them: {', '.join(unused_tools)})" if unused_tools else ""
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
                logger.info(f"Section {section.title} generation complete (tool calls: {tool_calls_count})")

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
                    logger.info(f"LLM attempted {len(tool_calls)} tool calls, executing only the first: {call['name']}")

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
                unused_hint = f"(These tools haven't been used yet, try them: {', '.join(unused_tools)})" if unused_tools else ""

                messages.append({
                    "role": "user",
                    "content": REACT_INSUFFICIENT_TOOLS_MSG_ALT.format(
                        tool_calls_count=tool_calls_count,
                        min_tool_calls=min_tool_calls,
                        unused_hint=unused_hint,
                    ),
                })
                continue


            logger.info(f"Section {section.title} no 'Final Answer:' prefix detected, adopting LLM output as final content (tool calls: {tool_calls_count})")
            final_answer = response.strip()

            if self.report_logger:
                self.report_logger.log_section_content(
                    section_title=section.title,
                    section_index=section_index,
                    content=final_answer,
                    tool_calls_count=tool_calls_count
                )
            return final_answer


        logger.warning(f"Section {section.title} reached max iterations, forcing generation")
        messages.append({"role": "user", "content": REACT_FORCE_FINAL_MSG})

        response = self.llm.chat(
            messages=messages,
            temperature=0.5,
            max_tokens=4096
        )


        if response is None:
            logger.error(f"Section {section.title} LLM returned None during forced conclusion, using default error")
            final_answer = f"(This chapter generation failed: LLM returned empty response, please retry later)"
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
                report_id, "pending", 0, "Initializing report...",
                completed_sections=[]
            )
            ReportManager.save_report(report)


            report.status = ReportStatus.PLANNING
            ReportManager.update_progress(
                report_id, "planning", 5, "Starting report outline planning...",
                completed_sections=[]
            )


            self.report_logger.log_planning_start()

            if progress_callback:
                progress_callback("planning", 0, "Starting report outline planning...")

            outline = self.plan_outline(
                progress_callback=lambda stage, prog, msg:
                    progress_callback(stage, prog // 5, msg) if progress_callback else None
            )
            report.outline = outline


            self.report_logger.log_planning_complete(outline.to_dict())


            ReportManager.save_outline(report_id, outline)
            ReportManager.update_progress(
                report_id, "planning", 15, f"Outline planning complete, {len(outline.sections)} sections",
                completed_sections=[]
            )
            ReportManager.save_report(report)

            logger.info(f"Outline saved to file: {report_id}/outline.json")


            report.status = ReportStatus.GENERATING

            total_sections = len(outline.sections)
            generated_sections = []

            for i, section in enumerate(outline.sections):
                section_num = i + 1
                base_progress = 20 + int((i / total_sections) * 70)


                ReportManager.update_progress(
                    report_id, "generating", base_progress,
                    f"Generating section: {section.title} ({section_num}/{total_sections})",
                    current_section=section.title,
                    completed_sections=completed_section_titles
                )

                if progress_callback:
                    progress_callback(
                        "generating",
                        base_progress,
                        f"Generating section: {section.title} ({section_num}/{total_sections})"
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

                logger.info(f"Section saved: {report_id}/section_{section_num:02d}.md")


                ReportManager.update_progress(
                    report_id, "generating",
                    base_progress + int(70 / total_sections),
                    f"Section {section.title} completed",
                    current_section=None,
                    completed_sections=completed_section_titles
                )


            if progress_callback:
                progress_callback("generating", 95, "Assembling full report...")

            ReportManager.update_progress(
                report_id, "generating", 95, "Assembling full report...",
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
                report_id, "completed", 100, "Report generation complete",
                completed_sections=completed_section_titles
            )

            if progress_callback:
                progress_callback("completed", 100, "Report generation complete")

            logger.info(f"Report generation complete: {report_id}")


            if self.console_logger:
                self.console_logger.close()
                self.console_logger = None

            return report

        except Exception as e:
            logger.error(f"Report generation failed: {str(e)}")
            report.status = ReportStatus.FAILED
            report.error = str(e)


            if self.report_logger:
                self.report_logger.log_error(str(e), "failed")


            try:
                ReportManager.save_report(report)
                ReportManager.update_progress(
                    report_id, "failed", -1, f"Report generation failed: {str(e)}",
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
        logger.info(f"Report Agent chat: {message[:50]}...")

        chat_history = chat_history or []


        report_content = ""
        try:
            report = ReportManager.get_report_by_simulation(self.simulation_id)
            if report and report.markdown_content:

                report_content = report.markdown_content[:15000]
                if len(report.markdown_content) > 15000:
                    report_content += "\n\n... [Report content truncated] ..."
        except Exception as e:
            logger.warning(f"Failed to get report content: {e}")

        system_prompt = CHAT_SYSTEM_PROMPT_TEMPLATE.format(
            simulation_requirement=self.simulation_requirement,
            report_content=report_content if report_content else "(No report available)",
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
            observation = "\n".join([f"[{r['tool']} result]\n{r['result']}" for r in tool_results])
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

        logger.info(f"Outline saved: {report_id}")

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
        logger.info(f"Section saved: {report_id}/{os.path.basename(file_path)}")
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

        logger.info(f"Full report assembled: {report_id}")
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

        logger.info(f"Report saved: {report.report_id}")

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
            logger.info(f"Report deleted: {report_id}")
        return deleted
