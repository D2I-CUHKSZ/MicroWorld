
import os
import time
import json
from collections import Counter
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field

from zep_cloud.client import Zep

from microworld.config.settings import Config
from microworld.telemetry.logging_config import get_logger
from microworld.infrastructure.llm_client import LLMClient
from microworld.infrastructure.llm_client_factory import LLMClientFactory
from microworld.memory.zep_paging import fetch_all_nodes, fetch_all_edges

logger = get_logger('microworld.zep_tools')


@dataclass
class SearchResult:
    facts: List[str]
    edges: List[Dict[str, Any]]
    nodes: List[Dict[str, Any]]
    query: str
    total_count: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "facts": self.facts,
            "edges": self.edges,
            "nodes": self.nodes,
            "query": self.query,
            "total_count": self.total_count
        }

    def to_text(self) -> str:
        text_parts = [f"Search query: {self.query}", f"Found {self.total_count} relevant items"]

        if self.facts:
            text_parts.append("\n### Related facts:")
            for i, fact in enumerate(self.facts, 1):
                text_parts.append(f"{i}. {fact}")

        return "\n".join(text_parts)


@dataclass
class NodeInfo:
    uuid: str
    name: str
    labels: List[str]
    summary: str
    attributes: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "uuid": self.uuid,
            "name": self.name,
            "labels": self.labels,
            "summary": self.summary,
            "attributes": self.attributes
        }

    def to_text(self) -> str:
        entity_type = next((l for l in self.labels if l not in ["Entity", "Node"]), "Unknown")
        return f"Entity: {self.name} (type: {entity_type})\nSummary: {self.summary}"


@dataclass
class EdgeInfo:
    uuid: str
    name: str
    fact: str
    source_node_uuid: str
    target_node_uuid: str
    source_node_name: Optional[str] = None
    target_node_name: Optional[str] = None

    created_at: Optional[str] = None
    valid_at: Optional[str] = None
    invalid_at: Optional[str] = None
    expired_at: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "uuid": self.uuid,
            "name": self.name,
            "fact": self.fact,
            "source_node_uuid": self.source_node_uuid,
            "target_node_uuid": self.target_node_uuid,
            "source_node_name": self.source_node_name,
            "target_node_name": self.target_node_name,
            "created_at": self.created_at,
            "valid_at": self.valid_at,
            "invalid_at": self.invalid_at,
            "expired_at": self.expired_at
        }

    def to_text(self, include_temporal: bool = False) -> str:
        source = self.source_node_name or self.source_node_uuid[:8]
        target = self.target_node_name or self.target_node_uuid[:8]
        base_text = f"Relation: {source} --[{self.name}]--> {target}\nFact: {self.fact}"

        if include_temporal:
            valid_at = self.valid_at or "unknown"
            invalid_at = self.invalid_at or "present"
            base_text += f"\nValidity: {valid_at} - {invalid_at}"
            if self.expired_at:
                base_text += f" (expired: {self.expired_at})"

        return base_text

    @property
    def is_expired(self) -> bool:
        return self.expired_at is not None

    @property
    def is_invalid(self) -> bool:
        return self.invalid_at is not None


@dataclass
class InsightForgeResult:
    query: str
    simulation_requirement: str
    sub_queries: List[str]


    semantic_facts: List[str] = field(default_factory=list)
    entity_insights: List[Dict[str, Any]] = field(default_factory=list)
    relationship_chains: List[str] = field(default_factory=list)


    total_facts: int = 0
    total_entities: int = 0
    total_relationships: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "simulation_requirement": self.simulation_requirement,
            "sub_queries": self.sub_queries,
            "semantic_facts": self.semantic_facts,
            "entity_insights": self.entity_insights,
            "relationship_chains": self.relationship_chains,
            "total_facts": self.total_facts,
            "total_entities": self.total_entities,
            "total_relationships": self.total_relationships
        }

    def to_text(self) -> str:
        text_parts = [
            f"## Deep Predictive Analysis",
            f"Analysis query: {self.query}",
            f"Prediction scenario: {self.simulation_requirement}",
            f"\n### Prediction Data Statistics",
            f"- Related prediction facts: {self.total_facts}",
            f"- Entities involved: {self.total_entities}",
            f"- Relationship chains: {self.total_relationships}"
        ]


        if self.sub_queries:
            text_parts.append(f"\n### Sub-queries Analyzed")
            for i, sq in enumerate(self.sub_queries, 1):
                text_parts.append(f"{i}. {sq}")


        if self.semantic_facts:
            text_parts.append(f"\n### [Key Facts] (cite these in reports)")
            for i, fact in enumerate(self.semantic_facts, 1):
                text_parts.append(f"{i}. \"{fact}\"")


        if self.entity_insights:
            text_parts.append(f"\n### [Core Entities]")
            for entity in self.entity_insights:
                text_parts.append(f"- **{entity.get('name', 'Unknown')}** ({entity.get('type', 'Entity')})")
                if entity.get('summary'):
                    text_parts.append(f"  Summary: \"{entity.get('summary')}\"")
                if entity.get('related_facts'):
                    text_parts.append(f"  Related facts: {len(entity.get('related_facts', []))}")


        if self.relationship_chains:
            text_parts.append(f"\n### [Relationship Chains]")
            for chain in self.relationship_chains:
                text_parts.append(f"- {chain}")

        return "\n".join(text_parts)


@dataclass
class PanoramaResult:
    query: str


    all_nodes: List[NodeInfo] = field(default_factory=list)

    all_edges: List[EdgeInfo] = field(default_factory=list)

    active_facts: List[str] = field(default_factory=list)

    historical_facts: List[str] = field(default_factory=list)


    total_nodes: int = 0
    total_edges: int = 0
    active_count: int = 0
    historical_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "all_nodes": [n.to_dict() for n in self.all_nodes],
            "all_edges": [e.to_dict() for e in self.all_edges],
            "active_facts": self.active_facts,
            "historical_facts": self.historical_facts,
            "total_nodes": self.total_nodes,
            "total_edges": self.total_edges,
            "active_count": self.active_count,
            "historical_count": self.historical_count
        }

    def to_text(self) -> str:
        text_parts = [
            f"## Broad Search Results (Panoramic View)",
            f"Query: {self.query}",
            f"\n### Statistics",
            f"- Total nodes: {self.total_nodes}",
            f"- Total edges: {self.total_edges}",
            f"- Active facts: {self.active_count}",
            f"- Historical/expired facts: {self.historical_count}"
        ]


        if self.active_facts:
            text_parts.append(f"\n### [Active Facts] (simulation results)")
            for i, fact in enumerate(self.active_facts, 1):
                text_parts.append(f"{i}. \"{fact}\"")


        if self.historical_facts:
            text_parts.append(f"\n### [Historical/Expired Facts] (evolution records)")
            for i, fact in enumerate(self.historical_facts, 1):
                text_parts.append(f"{i}. \"{fact}\"")


        if self.all_nodes:
            text_parts.append(f"\n### [Entities Involved]")
            for node in self.all_nodes:
                entity_type = next((l for l in node.labels if l not in ["Entity", "Node"]), "Entity")
                text_parts.append(f"- **{node.name}** ({entity_type})")

        return "\n".join(text_parts)


@dataclass
class AgentInterview:
    agent_name: str
    agent_role: str
    agent_bio: str
    question: str
    response: str
    key_quotes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_name": self.agent_name,
            "agent_role": self.agent_role,
            "agent_bio": self.agent_bio,
            "question": self.question,
            "response": self.response,
            "key_quotes": self.key_quotes
        }

    def to_text(self) -> str:
        text = f"**{self.agent_name}** ({self.agent_role})\n"

        text += f"_Bio: {self.agent_bio}_\n\n"
        text += f"**Q:** {self.question}\n\n"
        text += f"**A:** {self.response}\n"
        if self.key_quotes:
            text += "\n**Key quotes:**\n"
            for quote in self.key_quotes:

                clean_quote = quote.replace('\u201c', '').replace('\u201d', '').replace('"', '')
                clean_quote = clean_quote.replace('\u300c', '').replace('\u300d', '')
                clean_quote = clean_quote.strip()

                while clean_quote and clean_quote[0] in '，,；;：:、。！？\n\r\t ':
                    clean_quote = clean_quote[1:]

                skip = False
                for d in '123456789':
                    if f'\u95ee\u9898{d}' in clean_quote:
                        skip = True
                        break
                if skip:
                    continue

                if len(clean_quote) > 150:
                    dot_pos = clean_quote.find('\u3002', 80)
                    if dot_pos > 0:
                        clean_quote = clean_quote[:dot_pos + 1]
                    else:
                        clean_quote = clean_quote[:147] + "..."
                if clean_quote and len(clean_quote) >= 10:
                    text += f'> "{clean_quote}"\n'
        return text


@dataclass
class InterviewResult:
    interview_topic: str
    interview_questions: List[str]


    selected_agents: List[Dict[str, Any]] = field(default_factory=list)

    interviews: List[AgentInterview] = field(default_factory=list)


    selection_reasoning: str = ""

    summary: str = ""


    total_agents: int = 0
    interviewed_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "interview_topic": self.interview_topic,
            "interview_questions": self.interview_questions,
            "selected_agents": self.selected_agents,
            "interviews": [i.to_dict() for i in self.interviews],
            "selection_reasoning": self.selection_reasoning,
            "summary": self.summary,
            "total_agents": self.total_agents,
            "interviewed_count": self.interviewed_count
        }

    def to_text(self) -> str:
        text_parts = [
            "## In-Depth Interview Report",
            f"**Interview topic:** {self.interview_topic}",
            f"**Interviewees:** {self.interviewed_count} / {self.total_agents} simulated agents",
            "\n### Selection Reasoning",
            self.selection_reasoning or "(auto-selected)",
            "\n---",
            "\n### Interview Transcripts",
        ]

        if self.interviews:
            for i, interview in enumerate(self.interviews, 1):
                text_parts.append(f"\n#### Interview #{i}: {interview.agent_name}")
                text_parts.append(interview.to_text())
                text_parts.append("\n---")
        else:
            text_parts.append("(No interview records)\n\n---")

        text_parts.append("\n### Interview Summary & Key Insights")
        text_parts.append(self.summary or "(No summary)")

        return "\n".join(text_parts)


@dataclass
class SimulationArtifactSummary:
    simulation_id: str
    entity_prompt_count: int = 0
    keyword_examples: List[Dict[str, Any]] = field(default_factory=list)
    action_summary: Dict[str, Any] = field(default_factory=dict)
    topology_summary: Dict[str, Any] = field(default_factory=dict)
    memory_summary: Dict[str, Any] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "simulation_id": self.simulation_id,
            "entity_prompt_count": self.entity_prompt_count,
            "keyword_examples": self.keyword_examples,
            "action_summary": self.action_summary,
            "topology_summary": self.topology_summary,
            "memory_summary": self.memory_summary,
            "notes": self.notes,
        }

    def to_digest_lines(self, mode: str = "technical_report") -> List[str]:
        if str(mode).strip().lower() == "public_report":
            return self._to_public_digest_lines()
        return self._to_technical_digest_lines()

    def _to_public_digest_lines(self) -> List[str]:
        lines: List[str] = []
        if self.entity_prompt_count > 0:
            lines.append("The system has compiled multi-role profiles (institutions, students, media, self-media, etc.) to support multi-perspective narrative.")

        for platform, summary in self.action_summary.items():
            total_actions = int(summary.get("total_actions", 0) or 0)
            if total_actions <= 0:
                continue
            top_actions = [str(item.get("action_type", "") or "").upper() for item in (summary.get("top_action_types", []) or [])[:3]]
            if platform == "twitter":
                if any(action in {"QUOTE_POST", "REPOST"} for action in top_actions):
                    lines.append("Twitter side is dominated by quotes, reposts, and remix content; opinion spreads faster with stronger emotional amplification.")
                else:
                    lines.append("Twitter side favors quick interactions; opinion rhythm leans toward instant diffusion.")
            elif platform == "reddit":
                if any(action in {"SEARCH_POSTS", "CREATE_POST", "CREATE_COMMENT"} for action in top_actions):
                    lines.append("Reddit side leans toward research, long-form posts, and structured discussion; institutional and evidence topics are more prominent.")
                else:
                    lines.append("Reddit side leans toward sustained discussion around factual materials.")

        if self.topology_summary:
            lines.append("Both platforms show stable propagation clusters and core amplification nodes, indicating coordinated group dynamics rather than scattered discussion.")

        if self.memory_summary:
            lines.append("Early narratives are repeatedly recalled in later rounds, indicating a persistent memory effect in public opinion rather than one-time spikes.")

        lines.extend(self.notes[:3])
        return lines[:8]

    def _to_technical_digest_lines(self) -> List[str]:
        lines: List[str] = []
        if self.entity_prompt_count > 0:
            lines.append(f"Entity semantic distillation complete, entity prompts count: {self.entity_prompt_count}.")

        for platform, summary in self.action_summary.items():
            total_actions = int(summary.get("total_actions", 0) or 0)
            if total_actions <= 0:
                continue
            top_actions = summary.get("top_action_types", []) or []
            action_text = ", ".join(
                f"{item.get('action_type')}:{item.get('count')}"
                for item in top_actions[:3]
            )
            lines.append(f"{platform} platform produced {total_actions} actions, top behaviors: {action_text}.")

        for platform, summary in self.topology_summary.items():
            unit_count = int(summary.get("unit_count", 0) or 0)
            if unit_count <= 0:
                continue
            avg_unit_size = summary.get("avg_unit_size", 0)
            top_pair = (summary.get("top_asymmetric_pairs", []) or [{}])[0]
            if top_pair:
                pair_text = (
                    f"{top_pair.get('dominant_source_agent_name')} -> "
                    f"{top_pair.get('dominant_target_agent_name')} "
                    f"(delta={top_pair.get('delta')})"
                )
            else:
                pair_text = "no significant asymmetric pair"
            lines.append(
                f"{platform} platform formed {unit_count} coordination units, avg unit size {avg_unit_size}, "
                f"most prominent influence asymmetric pair: {pair_text}."
            )

        for platform, summary in self.memory_summary.items():
            agents_with_memory = int(summary.get("agents_with_memory", 0) or 0)
            total_units = int(summary.get("total_agent_units", 0) or 0)
            retrieval_events = int(summary.get("retrieval_events", 0) or 0)
            if total_units <= 0:
                continue
            lines.append(
                f"{platform} platform has {agents_with_memory} agents with memory, {total_units} agent memory units, "
                f"{retrieval_events} retrieval events."
            )

        lines.extend(self.notes[:3])
        return lines[:10]

    def to_text(self, mode: str = "technical_report") -> str:
        if str(mode).strip().lower() == "public_report":
            return self._to_public_text()
        return self._to_technical_text()

    def _to_public_text(self) -> str:
        lines = [
            "## Execution Summary (Public Edition)",
            f"simulation_id: {self.simulation_id}",
            "",
        ]
        for line in self._to_public_digest_lines():
            lines.append(f"- {line}")
        return "\n".join(lines)

    def _to_technical_text(self) -> str:
        lines = [
            "## Simulation Execution Summary",
            f"simulation_id: {self.simulation_id}",
        ]

        if self.keyword_examples:
            lines.append("\n### Entity Semantic Clues")
            for row in self.keyword_examples[:6]:
                lines.append(
                    f"- {row.get('entity_name')} ({row.get('entity_type')}): "
                    f"{', '.join(row.get('keywords', [])[:6])}"
                )

        if self.action_summary:
            lines.append("\n### Behavior Overview")
            for platform, summary in self.action_summary.items():
                top_agents = ", ".join(
                    f"{item.get('agent_name')}:{item.get('actions')}"
                    for item in (summary.get("top_agents", []) or [])[:4]
                ) or "none"
                top_actions = ", ".join(
                    f"{item.get('action_type')}:{item.get('count')}"
                    for item in (summary.get("top_action_types", []) or [])[:5]
                ) or "none"
                lines.append(
                    f"- {platform}: total_actions={summary.get('total_actions', 0)}, "
                    f"top_actions={top_actions}, top_agents={top_agents}"
                )

        if self.topology_summary:
            lines.append("\n### Coordination & Influence Signals")
            for platform, summary in self.topology_summary.items():
                lines.append(
                    f"- {platform}: units={summary.get('unit_count', 0)}, "
                    f"avg_unit_size={summary.get('avg_unit_size', 0)}, "
                    f"largest_unit_size={summary.get('largest_unit_size', 0)}"
                )
                for pair in (summary.get("top_asymmetric_pairs", []) or [])[:3]:
                    lines.append(
                        f"  - asymmetric: {pair.get('dominant_source_agent_name')} -> "
                        f"{pair.get('dominant_target_agent_name')} "
                        f"(dominant={pair.get('dominant_weight')}, reverse={pair.get('reverse_weight')})"
                    )

        if self.memory_summary:
            lines.append("\n### Memory Signals")
            for platform, summary in self.memory_summary.items():
                sample = summary.get("sample_retrieval")
                lines.append(
                    f"- {platform}: agents_with_memory={summary.get('agents_with_memory', 0)}, "
                    f"total_agent_units={summary.get('total_agent_units', 0)}, "
                    f"world_units={summary.get('world_units', 0)}, "
                    f"retrieval_events={summary.get('retrieval_events', 0)}"
                )
                if sample:
                    lines.append(
                        f"  - sample_retrieval: {sample.get('agent_name')} | "
                        f"complexity={sample.get('complexity')} | "
                        f"selected={sample.get('selected_topics')}"
                    )

        if self.notes:
            lines.append("\n### Notes")
            for note in self.notes[:5]:
                lines.append(f"- {note}")

        return "\n".join(lines)


class ZepToolsService:


    MAX_RETRIES = 3
    RETRY_DELAY = 2.0

    def __init__(self, api_key: Optional[str] = None, llm_client: Optional[LLMClient] = None):
        self.api_key = api_key or Config.ZEP_API_KEY
        if not self.api_key:
            raise ValueError("ZEP_API_KEY is not configured")

        self.client = Zep(api_key=self.api_key)

        self._llm_client = llm_client
        logger.info("ZepToolsService initialized")

    @property
    def llm(self) -> LLMClient:
        if self._llm_client is None:
            self._llm_client = LLMClientFactory.get_shared_client()
        return self._llm_client

    def _call_with_retry(self, func, operation_name: str, max_retries: int = None):
        max_retries = max_retries or self.MAX_RETRIES
        last_exception = None
        delay = self.RETRY_DELAY

        for attempt in range(max_retries):
            try:
                return func()
            except Exception as e:
                last_exception = e
                if attempt < max_retries - 1:
                    logger.warning(
                        f"Zep {operation_name} attempt {attempt + 1} failed: {str(e)[:100]}, "
                        f"retrying in {delay:.1f}s..."
                    )
                    time.sleep(delay)
                    delay *= 2
                else:
                    logger.error(f"Zep {operation_name} failed after {max_retries} attempts: {str(e)}")

        raise last_exception

    def _get_simulation_dir(self, simulation_id: str) -> str:
        return os.path.join(Config.OASIS_SIMULATION_DATA_DIR, simulation_id)

    def _read_json_file(self, path: str, default: Any = None) -> Any:
        if not os.path.exists(path):
            return default
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return default

    def _read_jsonl_file(self, path: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        if not os.path.exists(path):
            return []
        rows: List[Dict[str, Any]] = []
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
                    if limit is not None and len(rows) >= limit:
                        break
        except Exception:
            return []
        return rows

    def _summarize_action_rows(self, rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        action_rows = [row for row in rows if row.get("agent_id") is not None and row.get("action_type")]
        by_type = Counter(str(row.get("action_type")) for row in action_rows)
        by_agent = Counter(str(row.get("agent_name", f"Agent_{row.get('agent_id')}")) for row in action_rows)
        return {
            "total_actions": len(action_rows),
            "top_action_types": [
                {"action_type": name, "count": count}
                for name, count in by_type.most_common(6)
            ],
            "top_agents": [
                {"agent_name": name, "actions": count}
                for name, count in by_agent.most_common(6)
            ],
        }

    def _pick_sample_retrieval(self, rows: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        for row in rows:
            selected_units = row.get("selected_units") or []
            if not selected_units:
                continue
            topic_parts = []
            for item in selected_units[:4]:
                topic_parts.append(
                    f"{item.get('scope')}:{item.get('source_agent_name')}:{item.get('topic')}"
                )
            return {
                "agent_name": row.get("agent_name"),
                "complexity": (row.get("plan") or {}).get("complexity"),
                "selected_topics": "; ".join(topic_parts),
            }
        return None

    def get_simulation_artifact_summary(self, simulation_id: str) -> SimulationArtifactSummary:
        summary = SimulationArtifactSummary(simulation_id=simulation_id)
        sim_dir = self._get_simulation_dir(simulation_id)

        if not os.path.exists(sim_dir):
            summary.notes.append("Simulation directory not found, execution summary unavailable.")
            return summary

        entity_prompts = self._read_json_file(os.path.join(sim_dir, "entity_prompts.json"), default=[]) or []
        if isinstance(entity_prompts, list):
            summary.entity_prompt_count = len(entity_prompts)
            for row in entity_prompts[:6]:
                if not isinstance(row, dict):
                    continue
                summary.keyword_examples.append({
                    "entity_name": row.get("entity_name"),
                    "entity_type": row.get("entity_type"),
                    "keywords": row.get("keywords", []) or [],
                })

        for platform in ["twitter", "reddit"]:
            actions_path = os.path.join(sim_dir, platform, "actions.jsonl")
            action_rows = self._read_jsonl_file(actions_path)
            if action_rows:
                summary.action_summary[platform] = self._summarize_action_rows(action_rows)

            topology_path = os.path.join(
                sim_dir, "artifacts", "topology", platform, "latest_topology.json"
            )
            topology = self._read_json_file(topology_path, default={}) or {}
            if isinstance(topology, dict) and topology:
                summary.topology_summary[platform] = {
                    "unit_count": topology.get("unit_count", 0),
                    "avg_unit_size": topology.get("avg_unit_size", 0),
                    "largest_unit_size": topology.get("largest_unit_size", 0),
                    "top_asymmetric_pairs": (topology.get("top_asymmetric_pairs", []) or [])[:5],
                }

            memory_path = os.path.join(
                sim_dir, "artifacts", "memory", platform, "latest_memory_state.json"
            )
            memory_state = self._read_json_file(memory_path, default={}) or {}
            retrieval_path = os.path.join(
                sim_dir, "artifacts", "memory", platform, "retrieval_trace.jsonl"
            )
            retrieval_rows = self._read_jsonl_file(retrieval_path, limit=400)
            if isinstance(memory_state, dict) and memory_state:
                summary.memory_summary[platform] = {
                    "agents_with_memory": memory_state.get("agents_with_memory", 0),
                    "total_agent_units": memory_state.get("total_agent_units", 0),
                    "world_units": memory_state.get("world_units", 0),
                    "retrieval_events": len(retrieval_rows),
                    "sample_retrieval": self._pick_sample_retrieval(retrieval_rows),
                }

        if not summary.action_summary:
            summary.notes.append("Action logs not found or empty; report will rely more on graph and retrieval evidence.")
        if not summary.topology_summary:
            summary.notes.append("No topology artifact detected; coordination units and influence asymmetry will be weakly represented.")
        if not summary.memory_summary:
            summary.notes.append("No memory artifact detected; cross-round memory signals unavailable.")
        return summary

    def search_graph(
        self,
        graph_id: str,
        query: str,
        limit: int = 10,
        scope: str = "edges"
    ) -> SearchResult:
        logger.info(f"Graph search: graph_id={graph_id}, query={query[:50]}...")


        try:
            search_results = self._call_with_retry(
                func=lambda: self.client.graph.search(
                    graph_id=graph_id,
                    query=query,
                    limit=limit,
                    scope=scope,
                    reranker="cross_encoder"
                ),
                operation_name=f"graph_search(graph={graph_id})"
            )

            facts = []
            edges = []
            nodes = []


            if hasattr(search_results, 'edges') and search_results.edges:
                for edge in search_results.edges:
                    if hasattr(edge, 'fact') and edge.fact:
                        facts.append(edge.fact)
                    edges.append({
                        "uuid": getattr(edge, 'uuid_', None) or getattr(edge, 'uuid', ''),
                        "name": getattr(edge, 'name', ''),
                        "fact": getattr(edge, 'fact', ''),
                        "source_node_uuid": getattr(edge, 'source_node_uuid', ''),
                        "target_node_uuid": getattr(edge, 'target_node_uuid', ''),
                    })


            if hasattr(search_results, 'nodes') and search_results.nodes:
                for node in search_results.nodes:
                    nodes.append({
                        "uuid": getattr(node, 'uuid_', None) or getattr(node, 'uuid', ''),
                        "name": getattr(node, 'name', ''),
                        "labels": getattr(node, 'labels', []),
                        "summary": getattr(node, 'summary', ''),
                    })

                    if hasattr(node, 'summary') and node.summary:
                        facts.append(f"[{node.name}]: {node.summary}")

            logger.info(f"Search complete: found {len(facts)} related facts")

            return SearchResult(
                facts=facts,
                edges=edges,
                nodes=nodes,
                query=query,
                total_count=len(facts)
            )

        except Exception as e:
            logger.warning(f"Zep Search API failed, falling back to local search: {str(e)}")

            return self._local_search(graph_id, query, limit, scope)

    def _local_search(
        self,
        graph_id: str,
        query: str,
        limit: int = 10,
        scope: str = "edges"
    ) -> SearchResult:
        logger.info(f"Using local search: query={query[:30]}...")

        facts = []
        edges_result = []
        nodes_result = []


        query_lower = query.lower()
        keywords = [w.strip() for w in query_lower.replace(',', ' ').replace('，', ' ').split() if len(w.strip()) > 1]

        def match_score(text: str) -> int:
            if not text:
                return 0
            text_lower = text.lower()

            if query_lower in text_lower:
                return 100

            score = 0
            for keyword in keywords:
                if keyword in text_lower:
                    score += 10
            return score

        try:
            if scope in ["edges", "both"]:

                all_edges = self.get_all_edges(graph_id)
                scored_edges = []
                for edge in all_edges:
                    score = match_score(edge.fact) + match_score(edge.name)
                    if score > 0:
                        scored_edges.append((score, edge))


                scored_edges.sort(key=lambda x: x[0], reverse=True)

                for score, edge in scored_edges[:limit]:
                    if edge.fact:
                        facts.append(edge.fact)
                    edges_result.append({
                        "uuid": edge.uuid,
                        "name": edge.name,
                        "fact": edge.fact,
                        "source_node_uuid": edge.source_node_uuid,
                        "target_node_uuid": edge.target_node_uuid,
                    })

            if scope in ["nodes", "both"]:

                all_nodes = self.get_all_nodes(graph_id)
                scored_nodes = []
                for node in all_nodes:
                    score = match_score(node.name) + match_score(node.summary)
                    if score > 0:
                        scored_nodes.append((score, node))

                scored_nodes.sort(key=lambda x: x[0], reverse=True)

                for score, node in scored_nodes[:limit]:
                    nodes_result.append({
                        "uuid": node.uuid,
                        "name": node.name,
                        "labels": node.labels,
                        "summary": node.summary,
                    })
                    if node.summary:
                        facts.append(f"[{node.name}]: {node.summary}")

            logger.info(f"Local search complete: found {len(facts)} related facts")

        except Exception as e:
            logger.error(f"Local search failed: {str(e)}")

        return SearchResult(
            facts=facts,
            edges=edges_result,
            nodes=nodes_result,
            query=query,
            total_count=len(facts)
        )

    def get_all_nodes(self, graph_id: str) -> List[NodeInfo]:
        logger.info(f"Fetching all nodes for graph {graph_id}...")

        nodes = fetch_all_nodes(self.client, graph_id)

        result = []
        for node in nodes:
            node_uuid = getattr(node, 'uuid_', None) or getattr(node, 'uuid', None) or ""
            result.append(NodeInfo(
                uuid=str(node_uuid) if node_uuid else "",
                name=node.name or "",
                labels=node.labels or [],
                summary=node.summary or "",
                attributes=node.attributes or {}
            ))

        logger.info(f"Fetched {len(result)} nodes")
        return result

    def get_all_edges(self, graph_id: str, include_temporal: bool = True) -> List[EdgeInfo]:
        logger.info(f"Fetching all edges for graph {graph_id}...")

        edges = fetch_all_edges(self.client, graph_id)

        result = []
        for edge in edges:
            edge_uuid = getattr(edge, 'uuid_', None) or getattr(edge, 'uuid', None) or ""
            edge_info = EdgeInfo(
                uuid=str(edge_uuid) if edge_uuid else "",
                name=edge.name or "",
                fact=edge.fact or "",
                source_node_uuid=edge.source_node_uuid or "",
                target_node_uuid=edge.target_node_uuid or ""
            )


            if include_temporal:
                edge_info.created_at = getattr(edge, 'created_at', None)
                edge_info.valid_at = getattr(edge, 'valid_at', None)
                edge_info.invalid_at = getattr(edge, 'invalid_at', None)
                edge_info.expired_at = getattr(edge, 'expired_at', None)

            result.append(edge_info)

        logger.info(f"Fetched {len(result)} edges")
        return result

    def get_node_detail(self, node_uuid: str) -> Optional[NodeInfo]:
        logger.info(f"Fetching node detail: {node_uuid[:8]}...")

        try:
            node = self._call_with_retry(
                func=lambda: self.client.graph.node.get(uuid_=node_uuid),
                operation_name=f"get_node_detail(uuid={node_uuid[:8]}...)"
            )

            if not node:
                return None

            return NodeInfo(
                uuid=getattr(node, 'uuid_', None) or getattr(node, 'uuid', ''),
                name=node.name or "",
                labels=node.labels or [],
                summary=node.summary or "",
                attributes=node.attributes or {}
            )
        except Exception as e:
            logger.error(f"Failed to get node detail: {str(e)}")
            return None

    def get_node_edges(self, graph_id: str, node_uuid: str) -> List[EdgeInfo]:
        logger.info(f"Fetching edges for node {node_uuid[:8]}...")

        try:

            all_edges = self.get_all_edges(graph_id)

            result = []
            for edge in all_edges:

                if edge.source_node_uuid == node_uuid or edge.target_node_uuid == node_uuid:
                    result.append(edge)

            logger.info(f"Found {len(result)} edges related to node")
            return result

        except Exception as e:
            logger.warning(f"Failed to get node edges: {str(e)}")
            return []

    def get_entities_by_type(
        self,
        graph_id: str,
        entity_type: str
    ) -> List[NodeInfo]:
        logger.info(f"Fetching entities of type {entity_type}...")

        all_nodes = self.get_all_nodes(graph_id)

        filtered = []
        for node in all_nodes:

            if entity_type in node.labels:
                filtered.append(node)

        logger.info(f"Found {len(filtered)} entities of type {entity_type}")
        return filtered

    def get_entity_summary(
        self,
        graph_id: str,
        entity_name: str
    ) -> Dict[str, Any]:
        logger.info(f"Fetching relationship summary for entity {entity_name}...")


        search_result = self.search_graph(
            graph_id=graph_id,
            query=entity_name,
            limit=20
        )


        all_nodes = self.get_all_nodes(graph_id)
        entity_node = None
        for node in all_nodes:
            if node.name.lower() == entity_name.lower():
                entity_node = node
                break

        related_edges = []
        if entity_node:

            related_edges = self.get_node_edges(graph_id, entity_node.uuid)

        return {
            "entity_name": entity_name,
            "entity_info": entity_node.to_dict() if entity_node else None,
            "related_facts": search_result.facts,
            "related_edges": [e.to_dict() for e in related_edges],
            "total_relations": len(related_edges)
        }

    def get_graph_statistics(self, graph_id: str) -> Dict[str, Any]:
        logger.info(f"Fetching statistics for graph {graph_id}...")

        nodes = self.get_all_nodes(graph_id)
        edges = self.get_all_edges(graph_id)


        entity_types = {}
        for node in nodes:
            for label in node.labels:
                if label not in ["Entity", "Node"]:
                    entity_types[label] = entity_types.get(label, 0) + 1


        relation_types = {}
        for edge in edges:
            relation_types[edge.name] = relation_types.get(edge.name, 0) + 1

        return {
            "graph_id": graph_id,
            "total_nodes": len(nodes),
            "total_edges": len(edges),
            "entity_types": entity_types,
            "relation_types": relation_types
        }

    def get_simulation_context(
        self,
        graph_id: str,
        simulation_requirement: str,
        simulation_id: str = "",
        limit: int = 30,
        report_mode: str = "technical_report",
    ) -> Dict[str, Any]:
        logger.info(f"Fetching simulation context: {simulation_requirement[:50]}...")


        search_result = self.search_graph(
            graph_id=graph_id,
            query=simulation_requirement,
            limit=limit
        )


        stats = self.get_graph_statistics(graph_id)


        all_nodes = self.get_all_nodes(graph_id)


        entities = []
        for node in all_nodes:
            custom_labels = [l for l in node.labels if l not in ["Entity", "Node"]]
            if custom_labels:
                entities.append({
                    "name": node.name,
                    "type": custom_labels[0],
                    "summary": node.summary
                })

        artifact_summary = None
        artifact_lines: List[str] = []
        if simulation_id:
            artifact_summary = self.get_simulation_artifact_summary(simulation_id)
            artifact_lines = artifact_summary.to_digest_lines(mode=report_mode)

        related_facts = list(search_result.facts)
        if artifact_lines and str(report_mode).strip().lower() == "technical_report":
            related_facts.extend([f"[Runtime signal] {line}" for line in artifact_lines])

        return {
            "simulation_requirement": simulation_requirement,
            "related_facts": related_facts,
            "graph_statistics": stats,
            "entities": entities[:limit],
            "total_entities": len(entities),
            "simulation_id": simulation_id,
            "simulation_artifact_summary": artifact_summary.to_dict() if artifact_summary else {},
            "simulation_artifact_digest": artifact_lines,
            "report_mode": report_mode,
        }


    def insight_forge(
        self,
        graph_id: str,
        query: str,
        simulation_requirement: str,
        report_context: str = "",
        max_sub_queries: int = 5
    ) -> InsightForgeResult:
        logger.info(f"InsightForge deep insight retrieval: {query[:50]}...")

        result = InsightForgeResult(
            query=query,
            simulation_requirement=simulation_requirement,
            sub_queries=[]
        )


        sub_queries = self._generate_sub_queries(
            query=query,
            simulation_requirement=simulation_requirement,
            report_context=report_context,
            max_queries=max_sub_queries
        )
        result.sub_queries = sub_queries
        logger.info(f"Generated {len(sub_queries)} sub-queries")


        all_facts = []
        all_edges = []
        seen_facts = set()

        for sub_query in sub_queries:
            search_result = self.search_graph(
                graph_id=graph_id,
                query=sub_query,
                limit=15,
                scope="edges"
            )

            for fact in search_result.facts:
                if fact not in seen_facts:
                    all_facts.append(fact)
                    seen_facts.add(fact)

            all_edges.extend(search_result.edges)


        main_search = self.search_graph(
            graph_id=graph_id,
            query=query,
            limit=20,
            scope="edges"
        )
        for fact in main_search.facts:
            if fact not in seen_facts:
                all_facts.append(fact)
                seen_facts.add(fact)

        result.semantic_facts = all_facts
        result.total_facts = len(all_facts)


        entity_uuids = set()
        for edge_data in all_edges:
            if isinstance(edge_data, dict):
                source_uuid = edge_data.get('source_node_uuid', '')
                target_uuid = edge_data.get('target_node_uuid', '')
                if source_uuid:
                    entity_uuids.add(source_uuid)
                if target_uuid:
                    entity_uuids.add(target_uuid)


        entity_insights = []
        node_map = {}

        for uuid in list(entity_uuids):
            if not uuid:
                continue
            try:

                node = self.get_node_detail(uuid)
                if node:
                    node_map[uuid] = node
                    entity_type = next((l for l in node.labels if l not in ["Entity", "Node"]), "Entity")


                    related_facts = [
                        f for f in all_facts
                        if node.name.lower() in f.lower()
                    ]

                    entity_insights.append({
                        "uuid": node.uuid,
                        "name": node.name,
                        "type": entity_type,
                        "summary": node.summary,
                        "related_facts": related_facts
                    })
            except Exception as e:
                logger.debug(f"Failed to get node {uuid}: {e}")
                continue

        result.entity_insights = entity_insights
        result.total_entities = len(entity_insights)


        relationship_chains = []
        for edge_data in all_edges:
            if isinstance(edge_data, dict):
                source_uuid = edge_data.get('source_node_uuid', '')
                target_uuid = edge_data.get('target_node_uuid', '')
                relation_name = edge_data.get('name', '')

                source_name = node_map.get(source_uuid, NodeInfo('', '', [], '', {})).name or source_uuid[:8]
                target_name = node_map.get(target_uuid, NodeInfo('', '', [], '', {})).name or target_uuid[:8]

                chain = f"{source_name} --[{relation_name}]--> {target_name}"
                if chain not in relationship_chains:
                    relationship_chains.append(chain)

        result.relationship_chains = relationship_chains
        result.total_relationships = len(relationship_chains)

        logger.info(f"InsightForge complete: {result.total_facts} facts, {result.total_entities} entities, {result.total_relationships} relations")
        return result

    def _generate_sub_queries(
        self,
        query: str,
        simulation_requirement: str,
        report_context: str = "",
        max_queries: int = 5
    ) -> List[str]:
        system_prompt = """You are a professional question analysis expert. Your task is to decompose a complex question into multiple sub-questions that can be independently observed in a simulated world.

Requirements:
1. Each sub-question should be specific enough to find related agent behaviors or events in the simulation
2. Sub-questions should cover different dimensions (who, what, why, how, when, where)
3. Sub-questions should be relevant to the simulation scenario
4. Return JSON format: {"sub_queries": ["sub-question 1", "sub-question 2", ...]}"""

        user_prompt = f"""Simulation requirement background:
{simulation_requirement}

{f"Report context: {report_context[:500]}" if report_context else ""}

Please decompose the following question into {max_queries} sub-questions:
{query}

Return a JSON list of sub-questions."""

        try:
            response = self.llm.chat_json(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3
            )

            sub_queries = response.get("sub_queries", [])

            return [str(sq) for sq in sub_queries[:max_queries]]

        except Exception as e:
            logger.warning(f"Failed to generate sub-queries: {str(e)}, using defaults")

            return [
                query,
                f"Key participants in {query}",
                f"Causes and impact of {query}",
                f"Development process of {query}"
            ][:max_queries]

    def panorama_search(
        self,
        graph_id: str,
        query: str,
        include_expired: bool = True,
        limit: int = 50
    ) -> PanoramaResult:
        logger.info(f"PanoramaSearch broad search: {query[:50]}...")

        result = PanoramaResult(query=query)


        all_nodes = self.get_all_nodes(graph_id)
        node_map = {n.uuid: n for n in all_nodes}
        result.all_nodes = all_nodes
        result.total_nodes = len(all_nodes)


        all_edges = self.get_all_edges(graph_id, include_temporal=True)
        result.all_edges = all_edges
        result.total_edges = len(all_edges)


        active_facts = []
        historical_facts = []

        for edge in all_edges:
            if not edge.fact:
                continue


            source_name = node_map.get(edge.source_node_uuid, NodeInfo('', '', [], '', {})).name or edge.source_node_uuid[:8]
            target_name = node_map.get(edge.target_node_uuid, NodeInfo('', '', [], '', {})).name or edge.target_node_uuid[:8]


            is_historical = edge.is_expired or edge.is_invalid

            if is_historical:

                valid_at = edge.valid_at or "unknown"
                invalid_at = edge.invalid_at or edge.expired_at or "unknown"
                fact_with_time = f"[{valid_at} - {invalid_at}] {edge.fact}"
                historical_facts.append(fact_with_time)
            else:

                active_facts.append(edge.fact)


        query_lower = query.lower()
        keywords = [w.strip() for w in query_lower.replace(',', ' ').replace('，', ' ').split() if len(w.strip()) > 1]

        def relevance_score(fact: str) -> int:
            fact_lower = fact.lower()
            score = 0
            if query_lower in fact_lower:
                score += 100
            for kw in keywords:
                if kw in fact_lower:
                    score += 10
            return score


        active_facts.sort(key=relevance_score, reverse=True)
        historical_facts.sort(key=relevance_score, reverse=True)

        result.active_facts = active_facts[:limit]
        result.historical_facts = historical_facts[:limit] if include_expired else []
        result.active_count = len(active_facts)
        result.historical_count = len(historical_facts)

        logger.info(f"PanoramaSearch complete: {result.active_count} active, {result.historical_count} historical")
        return result

    def quick_search(
        self,
        graph_id: str,
        query: str,
        limit: int = 10
    ) -> SearchResult:
        logger.info(f"QuickSearch: {query[:50]}...")


        result = self.search_graph(
            graph_id=graph_id,
            query=query,
            limit=limit,
            scope="edges"
        )

        logger.info(f"QuickSearch complete: {result.total_count} results")
        return result

    def interview_agents(
        self,
        simulation_id: str,
        interview_requirement: str,
        simulation_requirement: str = "",
        max_agents: int = 5,
        custom_questions: List[str] = None
    ) -> InterviewResult:
        from microworld.simulation.simulation_runner import SimulationRunner

        logger.info(f"InterviewAgents in-depth interview (real API): {interview_requirement[:50]}...")

        result = InterviewResult(
            interview_topic=interview_requirement,
            interview_questions=custom_questions or []
        )


        profiles = self._load_agent_profiles(simulation_id)

        if not profiles:
            logger.warning(f"Agent profile file not found for simulation {simulation_id}")
            result.summary = "No agent profile files available for interview"
            return result

        result.total_agents = len(profiles)
        logger.info(f"Loaded {len(profiles)} agent profiles")


        selected_agents, selected_indices, selection_reasoning = self._select_agents_for_interview(
            profiles=profiles,
            interview_requirement=interview_requirement,
            simulation_requirement=simulation_requirement,
            max_agents=max_agents
        )

        result.selected_agents = selected_agents
        result.selection_reasoning = selection_reasoning
        logger.info(f"Selected {len(selected_agents)} agents for interview: {selected_indices}")


        if not result.interview_questions:
            result.interview_questions = self._generate_interview_questions(
                interview_requirement=interview_requirement,
                simulation_requirement=simulation_requirement,
                selected_agents=selected_agents
            )
            logger.info(f"Generated {len(result.interview_questions)} interview questions")


        combined_prompt = "\n".join([f"{i+1}. {q}" for i, q in enumerate(result.interview_questions)])


        INTERVIEW_PROMPT_PREFIX = (
            "You are being interviewed. Please answer the following questions based on your persona, "
            "all past memories and actions, in plain text.\n"
            "Response requirements:\n"
            "1. Answer directly in natural language, do not call any tools\n"
            "2. Do not return JSON or tool call formats\n"
            "3. Do not use Markdown headings (e.g., #, ##, ###)\n"
            "4. Answer each question in order, starting with 'Question X:' (X is the question number)\n"
            "5. Separate answers with blank lines\n"
            "6. Provide substantive content, at least 2-3 sentences per question\n\n"
        )
        optimized_prompt = f"{INTERVIEW_PROMPT_PREFIX}{combined_prompt}"


        try:

            interviews_request = []
            for agent_idx in selected_indices:
                interviews_request.append({
                    "agent_id": agent_idx,
                    "prompt": optimized_prompt

                })

            logger.info(f"Calling batch interview API (dual platform): {len(interviews_request)} agents")


            api_result = SimulationRunner.interview_agents_batch(
                simulation_id=simulation_id,
                interviews=interviews_request,
                platform=None,
                timeout=180.0
            )

            logger.info(f"Interview API returned: {api_result.get('interviews_count', 0)} results, success={api_result.get('success')}")


            if not api_result.get("success", False):
                error_msg = api_result.get("error", "unknown error")
                logger.warning(f"Interview API returned failure: {error_msg}")
                result.summary = f"Interview API call failed: {error_msg}. Please check OASIS simulation environment status."
                return result


            api_data = api_result.get("result", {})
            results_dict = api_data.get("results", {}) if isinstance(api_data, dict) else {}

            for i, agent_idx in enumerate(selected_indices):
                agent = selected_agents[i]
                agent_name = agent.get("realname", agent.get("username", f"Agent_{agent_idx}"))
                agent_role = agent.get("profession", "Unknown")
                agent_bio = agent.get("bio", "")


                twitter_result = results_dict.get(f"twitter_{agent_idx}", {})
                reddit_result = results_dict.get(f"reddit_{agent_idx}", {})

                twitter_response = twitter_result.get("response", "")
                reddit_response = reddit_result.get("response", "")


                twitter_response = self._clean_tool_call_response(twitter_response)
                reddit_response = self._clean_tool_call_response(reddit_response)


                twitter_text = twitter_response if twitter_response else "(No response from this platform)"
                reddit_text = reddit_response if reddit_response else "(No response from this platform)"
                response_text = f"[Twitter Platform Response]\n{twitter_text}\n\n[Reddit Platform Response]\n{reddit_text}"


                import re
                combined_responses = f"{twitter_response} {reddit_response}"


                clean_text = re.sub(r'#{1,6}\s+', '', combined_responses)
                clean_text = re.sub(r'\{[^}]*tool_name[^}]*\}', '', clean_text)
                clean_text = re.sub(r'[*_`|>~\-]{2,}', '', clean_text)
                clean_text = re.sub(r'Question\s*\d+[：:]\s*', '', clean_text)
                clean_text = re.sub(r'【[^】]+】', '', clean_text)


                sentences = re.split(r'[。！？]', clean_text)
                meaningful = [
                    s.strip() for s in sentences
                    if 20 <= len(s.strip()) <= 150
                    and not re.match(r'^[\s\W，,；;：:、]+', s.strip())
                    and not s.strip().startswith(('{', 'Question'))
                ]
                meaningful.sort(key=len, reverse=True)
                key_quotes = [s + "。" for s in meaningful[:3]]


                if not key_quotes:
                    paired = re.findall(r'\u201c([^\u201c\u201d]{15,100})\u201d', clean_text)
                    paired += re.findall(r'\u300c([^\u300c\u300d]{15,100})\u300d', clean_text)
                    key_quotes = [q for q in paired if not re.match(r'^[，,；;：:、]', q)][:3]

                interview = AgentInterview(
                    agent_name=agent_name,
                    agent_role=agent_role,
                    agent_bio=agent_bio[:1000],
                    question=combined_prompt,
                    response=response_text,
                    key_quotes=key_quotes[:5]
                )
                result.interviews.append(interview)

            result.interviewed_count = len(result.interviews)

        except ValueError as e:

            logger.warning(f"Interview API call failed (environment not running?): {e}")
            result.summary = f"Interview failed: {str(e)}. Simulation environment may be shut down, please ensure OASIS is running."
            return result
        except Exception as e:
            logger.error(f"Interview API call error: {e}")
            import traceback
            logger.error(traceback.format_exc())
            result.summary = f"Error during interview: {str(e)}"
            return result


        if result.interviews:
            result.summary = self._generate_interview_summary(
                interviews=result.interviews,
                interview_requirement=interview_requirement
            )

        logger.info(f"InterviewAgents complete: interviewed {result.interviewed_count} agents (dual platform)")
        return result

    @staticmethod
    def _clean_tool_call_response(response: str) -> str:
        if not response or not response.strip().startswith('{'):
            return response
        text = response.strip()
        if 'tool_name' not in text[:80]:
            return response
        import re as _re
        try:
            data = json.loads(text)
            if isinstance(data, dict) and 'arguments' in data:
                for key in ('content', 'text', 'body', 'message', 'reply'):
                    if key in data['arguments']:
                        return str(data['arguments'][key])
        except (json.JSONDecodeError, KeyError, TypeError):
            match = _re.search(r'"content"\s*:\s*"((?:[^"\\]|\\.)*)"', text)
            if match:
                return match.group(1).replace('\\n', '\n').replace('\\"', '"')
        return response

    def _load_agent_profiles(self, simulation_id: str) -> List[Dict[str, Any]]:
        import os
        import csv


        sim_dir = os.path.join(
            Config.OASIS_SIMULATION_DATA_DIR,
            simulation_id
        )

        profiles = []


        reddit_profile_path = os.path.join(sim_dir, "reddit_profiles.json")
        if os.path.exists(reddit_profile_path):
            try:
                with open(reddit_profile_path, 'r', encoding='utf-8') as f:
                    profiles = json.load(f)
                logger.info(f"Loaded {len(profiles)} profiles from reddit_profiles.json")
                return profiles
            except Exception as e:
                logger.warning(f"Failed to read reddit_profiles.json: {e}")


        twitter_profile_path = os.path.join(sim_dir, "twitter_profiles.csv")
        if os.path.exists(twitter_profile_path):
            try:
                with open(twitter_profile_path, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in reader:

                        profiles.append({
                            "realname": row.get("name", ""),
                            "username": row.get("username", ""),
                            "bio": row.get("description", ""),
                            "persona": row.get("user_char", ""),
                            "profession": "Unknown"
                        })
                logger.info(f"Loaded {len(profiles)} profiles from twitter_profiles.csv")
                return profiles
            except Exception as e:
                logger.warning(f"Failed to read twitter_profiles.csv: {e}")

        return profiles

    def _select_agents_for_interview(
        self,
        profiles: List[Dict[str, Any]],
        interview_requirement: str,
        simulation_requirement: str,
        max_agents: int
    ) -> tuple:


        agent_summaries = []
        for i, profile in enumerate(profiles):
            summary = {
                "index": i,
                "name": profile.get("realname", profile.get("username", f"Agent_{i}")),
                "profession": profile.get("profession", "Unknown"),
                "bio": profile.get("bio", "")[:200],
                "interested_topics": profile.get("interested_topics", [])
            }
            agent_summaries.append(summary)

        system_prompt = """You are a professional interview planning expert. Your task is to select the most suitable interview subjects from the simulated agent list based on interview requirements.

Selection criteria:
1. Agent's identity/profession is relevant to the interview topic
2. Agent may hold unique or valuable perspectives
3. Select diverse viewpoints (e.g., supporters, opponents, neutral parties, professionals)
4. Prioritize roles directly related to the event

Return JSON format:
{
    "selected_indices": [list of selected agent indices],
    "reasoning": "explanation of selection reasoning"
}"""

        user_prompt = f"""Interview requirement:
{interview_requirement}

Simulation background:
{simulation_requirement if simulation_requirement else "Not provided"}

Available agent list ({len(agent_summaries)} total):
{json.dumps(agent_summaries, ensure_ascii=False, indent=2)}

Please select up to {max_agents} agents most suitable for interview, and explain the selection reasoning."""

        try:
            response = self.llm.chat_json(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3
            )

            selected_indices = response.get("selected_indices", [])[:max_agents]
            reasoning = response.get("reasoning", "Auto-selected based on relevance")


            selected_agents = []
            valid_indices = []
            for idx in selected_indices:
                if 0 <= idx < len(profiles):
                    selected_agents.append(profiles[idx])
                    valid_indices.append(idx)

            return selected_agents, valid_indices, reasoning

        except Exception as e:
            logger.warning(f"LLM agent selection failed, using default: {e}")

            selected = profiles[:max_agents]
            indices = list(range(min(max_agents, len(profiles))))
            return selected, indices, "Using default selection strategy"

    def _generate_interview_questions(
        self,
        interview_requirement: str,
        simulation_requirement: str,
        selected_agents: List[Dict[str, Any]]
    ) -> List[str]:

        agent_roles = [a.get("profession", "Unknown") for a in selected_agents]

        system_prompt = """You are a professional journalist/interviewer. Generate 3-5 in-depth interview questions based on the interview requirements.

Question requirements:
1. Open-ended questions that encourage detailed answers
2. Different roles may have different answers
3. Cover multiple dimensions: facts, opinions, feelings
4. Natural language, like a real interview
5. Keep each question concise (under 50 words)
6. Ask directly, no background or prefix

Return JSON format: {"questions": ["question 1", "question 2", ...]}"""

        user_prompt = f"""Interview requirement: {interview_requirement}

Simulation background: {simulation_requirement if simulation_requirement else "Not provided"}

Interviewee roles: {', '.join(agent_roles)}

Please generate 3-5 interview questions."""

        try:
            response = self.llm.chat_json(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.5
            )

            return response.get("questions", [f"What are your thoughts on {interview_requirement}?"])

        except Exception as e:
            logger.warning(f"Failed to generate interview questions: {e}")
            return [
                f"What is your perspective on {interview_requirement}?",
                "How does this affect you or the group you represent?",
                "How do you think this issue should be resolved or improved?"
            ]

    def _generate_interview_summary(
        self,
        interviews: List[AgentInterview],
        interview_requirement: str
    ) -> str:

        if not interviews:
            return "No interviews completed"


        interview_texts = []
        for interview in interviews:
            interview_texts.append(f"[{interview.agent_name} ({interview.agent_role})]\n{interview.response[:500]}")

        system_prompt = """You are a professional news editor. Based on the responses from multiple interviewees, generate an interview summary.

Summary requirements:
1. Extract key viewpoints from all parties
2. Identify consensus and disagreements
3. Highlight valuable quotes
4. Remain objective and neutral
5. Keep within 1000 words

Format constraints (must follow):
- Use plain text paragraphs, separate sections with blank lines
- Do not use Markdown headings (e.g., #, ##, ###)
- Do not use dividers (e.g., ---, ***)
- Use quotation marks when citing interviewee quotes
- May use **bold** for keywords, but no other Markdown syntax"""

        user_prompt = f"""Interview topic: {interview_requirement}

Interview content:
{"".join(interview_texts)}

Please generate an interview summary."""

        try:
            summary = self.llm.chat(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                max_tokens=800
            )
            return summary

        except Exception as e:
            logger.warning(f"Failed to generate interview summary: {e}")

            return f"Interviewed {len(interviews)} respondents, including: " + ", ".join([i.agent_name for i in interviews])
