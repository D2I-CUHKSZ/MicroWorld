
import json
import math
import re
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field, asdict
from datetime import datetime

from openai import OpenAI

from lightworld.config.settings import Config
from lightworld.telemetry.logging_config import get_logger
from lightworld.graph.zep_entity_reader import EntityNode, ZepEntityReader

logger = get_logger('lightworld.simulation_config')


CHINA_TIMEZONE_CONFIG = {

    "dead_hours": [0, 1, 2, 3, 4, 5],

    "morning_hours": [6, 7, 8],

    "work_hours": [9, 10, 11, 12, 13, 14, 15, 16, 17, 18],

    "peak_hours": [19, 20, 21, 22],

    "night_hours": [23],

    "activity_multipliers": {
        "dead": 0.05,
        "morning": 0.4,
        "work": 0.7,
        "peak": 1.5,
        "night": 0.5
    }
}


@dataclass
class AgentActivityConfig:
    agent_id: int
    entity_uuid: str
    entity_name: str
    entity_type: str


    activity_level: float = 0.5


    posts_per_hour: float = 1.0
    comments_per_hour: float = 2.0


    active_hours: List[int] = field(default_factory=lambda: list(range(8, 23)))


    response_delay_min: int = 5
    response_delay_max: int = 60


    sentiment_bias: float = 0.0


    stance: str = "neutral"


    influence_weight: float = 1.0


@dataclass
class TimeSimulationConfig:

    total_simulation_hours: int = 72


    minutes_per_round: int = 60


    agents_per_hour_min: int = 5
    agents_per_hour_max: int = 20


    peak_hours: List[int] = field(default_factory=lambda: [19, 20, 21, 22])
    peak_activity_multiplier: float = 1.5


    off_peak_hours: List[int] = field(default_factory=lambda: [0, 1, 2, 3, 4, 5])
    off_peak_activity_multiplier: float = 0.05


    morning_hours: List[int] = field(default_factory=lambda: [6, 7, 8])
    morning_activity_multiplier: float = 0.4


    work_hours: List[int] = field(default_factory=lambda: [9, 10, 11, 12, 13, 14, 15, 16, 17, 18])
    work_activity_multiplier: float = 0.7


@dataclass
class EventConfig:

    initial_posts: List[Dict[str, Any]] = field(default_factory=list)


    scheduled_events: List[Dict[str, Any]] = field(default_factory=list)


    hot_topics: List[str] = field(default_factory=list)


    narrative_direction: str = ""


@dataclass
class PlatformConfig:
    platform: str
    recsys_type: str = "reddit"
    refresh_rec_post_count: int = 1
    max_rec_post_len: int = 2
    following_post_count: int = 3
    rec_prob: float = 0.7
    trend_num_days: int = 7
    trend_top_k: int = 1
    report_threshold: int = 2
    show_score: bool = False
    allow_self_rating: bool = True
    use_openai_embedding: bool = False


@dataclass
class TopologyAwareConfig:
    enabled: bool = True
    coordination_enabled: bool = True
    differentiation_enabled: bool = True

    similarity_threshold: float = 0.9
    top_pairs_ratio: float = 0.03
    min_unit_size: int = 2
    extra_member_prob: float = 0.12
    importance_alpha: float = 0.7

    sentiment_diff_threshold: float = 0.35
    opinion_threshold: float = 0.5
    stubbornness_threshold: float = 0.5
    influence_threshold: float = 0.5

    ppr_alpha: float = 0.85
    ppr_eps: float = 1e-4

    semantic_threshold: float = 0.1
    keyword_jaccard_threshold: float = 0.12
    keyword_overlap_min: int = 1
    threshold_cluster_enabled: bool = False
    llm_keyword_cluster_enabled: bool = False
    cluster_mode: str = "disabled"


    graph_prior_similarity_boost: float = 0.35
    graph_prior_extra_ratio: float = 0.25


    dynamic_update_enabled: bool = True
    dynamic_update_interval: int = 4
    dynamic_update_min_events: int = 8
    dynamic_interaction_min_weight: float = 0.25
    dynamic_neighbors_per_agent: int = 6


    initial_follow_max_per_agent: int = 3
    initial_follow_max_total: int = 0


    social_link_sync_enabled: bool = True
    social_link_sync_interval: int = 6
    social_link_sync_max_total: int = 20


@dataclass
class SimpleMemConfig:
    enabled: bool = True
    max_units_per_agent: int = 120
    retrieval_topk: int = 5
    merge_jaccard_threshold: float = 0.45
    max_injected_chars: int = 1200
    recency_decay: float = 0.08
    counter_scope_max: int = 1
    counter_opinion_gap: float = 0.35
    novelty_lookback: int = 6
    unit_repeat_penalty: float = 0.35
    topic_repeat_penalty: float = 0.15


@dataclass
class LightModeConfig:
    enabled: bool = False
    agent_ratio: float = 0.6


@dataclass
class SimulationParameters:

    simulation_id: str
    project_id: str
    graph_id: str
    simulation_requirement: str


    time_config: TimeSimulationConfig = field(default_factory=TimeSimulationConfig)


    agent_configs: List[AgentActivityConfig] = field(default_factory=list)


    event_config: EventConfig = field(default_factory=EventConfig)


    twitter_config: Optional[PlatformConfig] = None
    reddit_config: Optional[PlatformConfig] = None


    topology_aware: TopologyAwareConfig = field(default_factory=TopologyAwareConfig)
    simplemem: SimpleMemConfig = field(default_factory=SimpleMemConfig)
    light_mode: LightModeConfig = field(default_factory=LightModeConfig)


    llm_model: str = ""
    llm_base_url: str = ""


    generated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    generation_reasoning: str = ""

    def to_dict(self) -> Dict[str, Any]:
        time_dict = asdict(self.time_config)
        return {
            "simulation_id": self.simulation_id,
            "project_id": self.project_id,
            "graph_id": self.graph_id,
            "simulation_requirement": self.simulation_requirement,
            "time_config": time_dict,
            "agent_configs": [asdict(a) for a in self.agent_configs],
            "event_config": asdict(self.event_config),
            "twitter_config": asdict(self.twitter_config) if self.twitter_config else None,
            "reddit_config": asdict(self.reddit_config) if self.reddit_config else None,
            "topology_aware": asdict(self.topology_aware),
            "simplemem": asdict(self.simplemem),
            "light_mode": asdict(self.light_mode),
            "llm_model": self.llm_model,
            "llm_base_url": self.llm_base_url,
            "generated_at": self.generated_at,
            "generation_reasoning": self.generation_reasoning,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)


class SimulationConfigGenerator:


    MAX_CONTEXT_LENGTH = 50000

    AGENTS_PER_BATCH = 15


    TIME_CONFIG_CONTEXT_LENGTH = 10000
    EVENT_CONFIG_CONTEXT_LENGTH = 8000
    ENTITY_SUMMARY_LENGTH = 300
    AGENT_SUMMARY_LENGTH = 300
    ENTITIES_PER_TYPE_DISPLAY = 20

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model_name: Optional[str] = None
    ):
        self.api_key = api_key or Config.LLM_API_KEY
        self.base_url = base_url or Config.LLM_BASE_URL
        self.model_name = model_name or Config.LLM_MODEL_NAME

        if not self.api_key:
            raise ValueError("LLM_API_KEY is not configured")

        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )

    def generate_config(
        self,
        simulation_id: str,
        project_id: str,
        graph_id: str,
        simulation_requirement: str,
        document_text: str,
        entities: List[EntityNode],
        enable_twitter: bool = True,
        enable_reddit: bool = True,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> SimulationParameters:
        logger.info(f"Starting intelligent simulation config generation: simulation_id={simulation_id}, entities={len(entities)}")


        num_batches = math.ceil(len(entities) / self.AGENTS_PER_BATCH)
        total_steps = 3 + num_batches
        current_step = 0

        def report_progress(step: int, message: str):
            nonlocal current_step
            current_step = step
            if progress_callback:
                progress_callback(step, total_steps, message)
            logger.info(f"[{step}/{total_steps}] {message}")


        context = self._build_context(
            simulation_requirement=simulation_requirement,
            document_text=document_text,
            entities=entities
        )

        reasoning_parts = []


        report_progress(1, "Generating time config...")
        num_entities = len(entities)
        time_config_result = self._generate_time_config(context, num_entities)
        time_config = self._parse_time_config(time_config_result, num_entities)
        reasoning_parts.append(f"Time config: {time_config_result.get('reasoning', 'success')}")


        report_progress(2, "Generating event config and hot topics...")
        event_config_result = self._generate_event_config(context, simulation_requirement, entities)
        event_config = self._parse_event_config(event_config_result)
        reasoning_parts.append(f"Event config: {event_config_result.get('reasoning', 'success')}")


        all_agent_configs = []
        for batch_idx in range(num_batches):
            start_idx = batch_idx * self.AGENTS_PER_BATCH
            end_idx = min(start_idx + self.AGENTS_PER_BATCH, len(entities))
            batch_entities = entities[start_idx:end_idx]

            report_progress(
                3 + batch_idx,
                f"Generating agent configs ({start_idx + 1}-{end_idx}/{len(entities)})..."
            )

            batch_configs = self._generate_agent_configs_batch(
                context=context,
                entities=batch_entities,
                start_idx=start_idx,
                simulation_requirement=simulation_requirement
            )
            all_agent_configs.extend(batch_configs)

        reasoning_parts.append(f"Agent configs: successfully generated {len(all_agent_configs)}")


        logger.info("Assigning appropriate poster agents to initial posts...")
        event_config = self._assign_initial_post_agents(event_config, all_agent_configs)
        assigned_count = len([p for p in event_config.initial_posts if p.get("poster_agent_id") is not None])
        reasoning_parts.append(f"Initial post assignment: {assigned_count} posts assigned to posters")


        report_progress(total_steps, "Generating platform config...")
        twitter_config = None
        reddit_config = None

        if enable_twitter:
            twitter_config = PlatformConfig(
                platform="twitter",
                recsys_type="twhin-bert",
                refresh_rec_post_count=2,
                max_rec_post_len=2,
                following_post_count=3,
                rec_prob=0.7,
                trend_num_days=7,
                trend_top_k=1,
                report_threshold=2,
                show_score=False,
                allow_self_rating=True,
                use_openai_embedding=False,
            )

        if enable_reddit:
            reddit_config = PlatformConfig(
                platform="reddit",
                recsys_type="reddit",
                refresh_rec_post_count=5,
                max_rec_post_len=100,
                following_post_count=3,
                rec_prob=0.7,
                trend_num_days=7,
                trend_top_k=1,
                report_threshold=2,
                show_score=True,
                allow_self_rating=True,
                use_openai_embedding=False,
            )


        params = SimulationParameters(
            simulation_id=simulation_id,
            project_id=project_id,
            graph_id=graph_id,
            simulation_requirement=simulation_requirement,
            time_config=time_config,
            agent_configs=all_agent_configs,
            event_config=event_config,
            twitter_config=twitter_config,
            reddit_config=reddit_config,
            llm_model=self.model_name,
            llm_base_url=self.base_url,
            generation_reasoning=" | ".join(reasoning_parts)
        )

        logger.info(f"Simulation config generation complete: {len(params.agent_configs)} agent configs")

        return params

    def _build_context(
        self,
        simulation_requirement: str,
        document_text: str,
        entities: List[EntityNode]
    ) -> str:


        entity_summary = self._summarize_entities(entities)


        context_parts = [
            f"## Simulation Requirements\n{simulation_requirement}",
            f"\n## Entity Information ({len(entities)} entities)\n{entity_summary}",
        ]

        current_length = sum(len(p) for p in context_parts)
        remaining_length = self.MAX_CONTEXT_LENGTH - current_length - 500

        if remaining_length > 0 and document_text:
            doc_text = document_text[:remaining_length]
            if len(document_text) > remaining_length:
                doc_text += "\n...(document truncated)"
            context_parts.append(f"\n## Original Document Content\n{doc_text}")

        return "\n".join(context_parts)

    def _summarize_entities(self, entities: List[EntityNode]) -> str:
        lines = []


        by_type: Dict[str, List[EntityNode]] = {}
        for e in entities:
            t = e.get_entity_type() or "Unknown"
            if t not in by_type:
                by_type[t] = []
            by_type[t].append(e)

        for entity_type, type_entities in by_type.items():
            lines.append(f"\n### {entity_type} ({len(type_entities)} entities)")

            display_count = self.ENTITIES_PER_TYPE_DISPLAY
            summary_len = self.ENTITY_SUMMARY_LENGTH
            for e in type_entities[:display_count]:
                summary_preview = (e.summary[:summary_len] + "...") if len(e.summary) > summary_len else e.summary
                lines.append(f"- {e.name}: {summary_preview}")
            if len(type_entities) > display_count:
                lines.append(f"  ... {len(type_entities) - display_count} more")

        return "\n".join(lines)

    def _call_llm_with_retry(self, prompt: str, system_prompt: str) -> Dict[str, Any]:
        import re

        max_attempts = 3
        last_error = None

        for attempt in range(max_attempts):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.7 - (attempt * 0.1)

                )

                content = response.choices[0].message.content
                finish_reason = response.choices[0].finish_reason


                if finish_reason == 'length':
                    logger.warning(f"LLM output truncated (attempt {attempt+1})")
                    content = self._fix_truncated_json(content)


                try:
                    return json.loads(content)
                except json.JSONDecodeError as e:
                    logger.warning(f"JSON parse failed (attempt {attempt+1}): {str(e)[:80]}")


                    fixed = self._try_fix_config_json(content)
                    if fixed:
                        return fixed

                    last_error = e

            except Exception as e:
                logger.warning(f"LLM call failed (attempt {attempt+1}): {str(e)[:80]}")
                last_error = e
                import time
                time.sleep(2 * (attempt + 1))

        raise last_error or Exception("LLM call failed")

    def _fix_truncated_json(self, content: str) -> str:
        content = content.strip()


        open_braces = content.count('{') - content.count('}')
        open_brackets = content.count('[') - content.count(']')


        if content and content[-1] not in '",}]':
            content += '"'


        content += ']' * open_brackets
        content += '}' * open_braces

        return content

    def _try_fix_config_json(self, content: str) -> Optional[Dict[str, Any]]:
        import re


        content = self._fix_truncated_json(content)


        json_match = re.search(r'\{[\s\S]*\}', content)
        if json_match:
            json_str = json_match.group()


            def fix_string(match):
                s = match.group(0)
                s = s.replace('\n', ' ').replace('\r', ' ')
                s = re.sub(r'\s+', ' ', s)
                return s

            json_str = re.sub(r'"[^"\\]*(?:\\.[^"\\]*)*"', fix_string, json_str)

            try:
                return json.loads(json_str)
            except:

                json_str = re.sub(r'[\x00-\x1f\x7f-\x9f]', ' ', json_str)
                json_str = re.sub(r'\s+', ' ', json_str)
                try:
                    return json.loads(json_str)
                except:
                    pass

        return None

    def _generate_time_config(self, context: str, num_entities: int) -> Dict[str, Any]:

        context_truncated = context[:self.TIME_CONFIG_CONTEXT_LENGTH]


        max_agents_allowed = max(1, int(num_entities * 0.9))

        prompt = f"""Based on the following simulation requirements, generate time simulation config.

{context_truncated}

## Task
Generate a time config JSON.

### Basic Principles (reference only, adjust based on specific events and participant groups):
- Target user group is Chinese, must follow Beijing timezone daily patterns
- 0-5am almost no activity (activity coefficient 0.05)
- 6-8am gradually active (activity coefficient 0.4)
- 9am-6pm work hours moderately active (activity coefficient 0.7)
- 7-10pm peak hours (activity coefficient 1.5)
- After 11pm activity declines (activity coefficient 0.5)
- General pattern: low activity at dawn, increasing in morning, moderate during work, peak in evening
- **Important**: Example values are reference only. Adjust based on event nature and participant characteristics
  - Example: Student groups may peak at 9-11pm; media active all day; official institutions only during work hours
  - Example: Breaking news may cause late-night discussions, off_peak_hours can be shortened

### Return JSON format (no markdown)

Example:
{{
    "total_simulation_hours": 72,
    "minutes_per_round": 60,
    "agents_per_hour_min": 5,
    "agents_per_hour_max": 50,
    "peak_hours": [19, 20, 21, 22],
    "off_peak_hours": [0, 1, 2, 3, 4, 5],
    "morning_hours": [6, 7, 8],
    "work_hours": [9, 10, 11, 12, 13, 14, 15, 16, 17, 18],
    "reasoning": "Time config explanation for this event"
}}

Field descriptions:
- total_simulation_hours (int): Total simulation duration, 24-168 hours, shorter for breaking events, longer for ongoing topics
- minutes_per_round (int): Duration per round, 30-120 minutes, recommended 60 minutes
- agents_per_hour_min (int): Min agents activated per hour (range: 1-{max_agents_allowed})
- agents_per_hour_max (int): Max agents activated per hour (range: 1-{max_agents_allowed})
- peak_hours (int array): Peak hours, adjust based on participant groups
- off_peak_hours (int array): Off-peak hours, typically late night/early morning
- morning_hours (int array): Morning hours
- work_hours (int array): Work hours
- reasoning (string): Brief explanation of config rationale"""

        system_prompt = "You are a social media simulation expert. Return pure JSON format. Time config must follow Chinese daily patterns."

        try:
            return self._call_llm_with_retry(prompt, system_prompt)
        except Exception as e:
            logger.warning(f"Time config LLM generation failed: {e}, using defaults")
            return self._get_default_time_config(num_entities)

    def _get_default_time_config(self, num_entities: int) -> Dict[str, Any]:
        return {
            "total_simulation_hours": 72,
            "minutes_per_round": 60,
            "agents_per_hour_min": max(1, num_entities // 15),
            "agents_per_hour_max": max(5, num_entities // 5),
            "peak_hours": [19, 20, 21, 22],
            "off_peak_hours": [0, 1, 2, 3, 4, 5],
            "morning_hours": [6, 7, 8],
            "work_hours": [9, 10, 11, 12, 13, 14, 15, 16, 17, 18],
            "reasoning": "Using default Chinese daily pattern config (1 hour per round)"
        }

    def _parse_time_config(self, result: Dict[str, Any], num_entities: int) -> TimeSimulationConfig:

        agents_per_hour_min = result.get("agents_per_hour_min", max(1, num_entities // 15))
        agents_per_hour_max = result.get("agents_per_hour_max", max(5, num_entities // 5))


        if agents_per_hour_min > num_entities:
            logger.warning(f"agents_per_hour_min ({agents_per_hour_min}) exceeds total agents ({num_entities}), corrected")
            agents_per_hour_min = max(1, num_entities // 10)

        if agents_per_hour_max > num_entities:
            logger.warning(f"agents_per_hour_max ({agents_per_hour_max}) exceeds total agents ({num_entities}), corrected")
            agents_per_hour_max = max(agents_per_hour_min + 1, num_entities // 2)


        if agents_per_hour_min >= agents_per_hour_max:
            agents_per_hour_min = max(1, agents_per_hour_max // 2)
            logger.warning(f"agents_per_hour_min >= max, corrected to {agents_per_hour_min}")

        return TimeSimulationConfig(
            total_simulation_hours=result.get("total_simulation_hours", 72),
            minutes_per_round=result.get("minutes_per_round", 60),
            agents_per_hour_min=agents_per_hour_min,
            agents_per_hour_max=agents_per_hour_max,
            peak_hours=result.get("peak_hours", [19, 20, 21, 22]),
            off_peak_hours=result.get("off_peak_hours", [0, 1, 2, 3, 4, 5]),
            off_peak_activity_multiplier=0.05,
            morning_hours=result.get("morning_hours", [6, 7, 8]),
            morning_activity_multiplier=0.4,
            work_hours=result.get("work_hours", list(range(9, 19))),
            work_activity_multiplier=0.7,
            peak_activity_multiplier=1.5
        )

    def _generate_event_config(
        self,
        context: str,
        simulation_requirement: str,
        entities: List[EntityNode]
    ) -> Dict[str, Any]:


        entity_types_available = list(set(
            e.get_entity_type() or "Unknown" for e in entities
        ))


        type_examples = {}
        for e in entities:
            etype = e.get_entity_type() or "Unknown"
            if etype not in type_examples:
                type_examples[etype] = []
            if len(type_examples[etype]) < 3:
                type_examples[etype].append(e.name)

        type_info = "\n".join([
            f"- {t}: {', '.join(examples)}"
            for t, examples in type_examples.items()
        ])


        context_truncated = context[:self.EVENT_CONFIG_CONTEXT_LENGTH]

        prompt = f"""Based on the following simulation requirements, generate event config.

Simulation requirement: {simulation_requirement}

{context_truncated}

## Available Entity Types and Examples
{type_info}

        ## Task
        Generate event config JSON:
        - Extract hot topic keywords
        - Describe opinion development direction
        - Design initial content with conflict and divergence, **each post must specify poster_type**
        - If mid-simulation event injection is needed, generate `scheduled_events`

        Cover the following social propagation structures:
        - At least 1 official/institutional announcement
        - At least 1 media follow-up or fact-check
        - At least 1 ordinary user/student emotional expression
        - At least 1 opposing stance or questioning response
        - At least 1 piece of new information that escalates or redirects the topic

        `scheduled_events` currently supports these event types:
        1. `create_post`
           - Must include `trigger_hour`
           - Must include `content`
           - Must include `poster_type`
        2. `create_comment`
           - Must include `trigger_hour`
           - Must include `content`
           - Must include `poster_type`
           - Must include `target_poster_type` or `target_post_strategy`
        3. `create_thread`
           - Must include `trigger_hour`
           - Must include `poster_type`
           - Must include `root_content`
           - Must include `replies` (1-3 strings)
        4. `hot_topics_update`
           - Must include `trigger_hour`
           - May include `hot_topics_add`
           - May include `hot_topics_remove`

**Important**: poster_type must be selected from the "Available Entity Types" above so initial posts can be assigned to appropriate agents.
Example: Official statements should be posted by Official/University types, news by MediaOutlet, student views by Student.
Avoid making all initial content read like press releases; at least half should have clear attitude, questioning, controversy, or emotional tension.

Return JSON format (no markdown):
        {{
            "hot_topics": ["keyword1", "keyword2", ...],
            "narrative_direction": "<opinion development direction description>",
            "initial_posts": [
                {{"content": "post content", "poster_type": "entity type (must be from available types)"}},
                ...
            ],
            "scheduled_events": [
                {{
                    "event_type": "create_post",
                    "trigger_hour": 6,
                    "content": "scheduled post content",
                    "poster_type": "entity type (must be from available types)"
                }},
                {{
                    "event_type": "create_comment",
                    "trigger_hour": 8,
                    "content": "response/questioning/supplementing existing post",
                    "poster_type": "entity type (must be from available types)",
                    "target_poster_type": "type of the poster being responded to",
                    "target_post_strategy": "latest_post_by_type"
                }},
                {{
                    "event_type": "create_thread",
                    "trigger_hour": 12,
                    "poster_type": "entity type (must be from available types)",
                    "root_content": "thread root post content",
                    "replies": ["supplement 1", "supplement 2"]
                }},
                {{
                    "event_type": "hot_topics_update",
                    "trigger_hour": 12,
                    "hot_topics_add": ["new topic 1"],
                    "hot_topics_remove": ["old topic 1"]
                }}
            ],
            "reasoning": "<brief explanation>"
        }}"""

        system_prompt = "You are an opinion analysis expert. Return pure JSON format. Note: poster_type must exactly match available entity types."

        try:
            return self._call_llm_with_retry(prompt, system_prompt)
        except Exception as e:
            logger.warning(f"Event config LLM generation failed: {e}, using defaults")
            return {
                "hot_topics": [],
                "narrative_direction": "",
                "initial_posts": [],
                "scheduled_events": [],
                "reasoning": "Using default config"
            }

    def _parse_event_config(self, result: Dict[str, Any]) -> EventConfig:
        scheduled_events = result.get("scheduled_events", [])
        if not isinstance(scheduled_events, list):
            scheduled_events = []

        normalized_scheduled_events = []
        for item in scheduled_events:
            if not isinstance(item, dict):
                continue

            normalized = dict(item)
            event_type = str(normalized.get("event_type", "") or "").strip().lower()
            if not event_type:
                if normalized.get("content"):
                    event_type = "create_post"
                elif normalized.get("hot_topics_add") or normalized.get("hot_topics_remove"):
                    event_type = "hot_topics_update"
                else:
                    continue
            normalized["event_type"] = event_type

            for key in ["trigger_hour", "hour", "hour_offset", "trigger_day", "trigger_round", "round_offset"]:
                if key not in normalized:
                    continue
                try:
                    normalized[key] = int(normalized[key])
                except Exception:
                    normalized.pop(key, None)

            for key in ["hot_topics_add", "hot_topics_remove"]:
                value = normalized.get(key, [])
                if isinstance(value, str):
                    value = [x.strip() for x in re.split(r"[，,;；\s]+", value) if x.strip()]
                if not isinstance(value, list):
                    value = []
                normalized[key] = [str(x).strip() for x in value if str(x).strip()]

            if event_type == "create_thread":
                replies = normalized.get("replies", [])
                if isinstance(replies, str):
                    replies = [x.strip() for x in re.split(r"[|｜\n]+", replies) if x.strip()]
                if not isinstance(replies, list):
                    replies = []
                normalized["replies"] = [str(x).strip() for x in replies if str(x).strip()][:3]
            if event_type == "create_comment":
                target_strategy = str(normalized.get("target_post_strategy", "") or "").strip()
                if not target_strategy:
                    if normalized.get("target_poster_type"):
                        target_strategy = "latest_post_by_type"
                    else:
                        target_strategy = "latest_hot_post"
                normalized["target_post_strategy"] = target_strategy

            normalized_scheduled_events.append(normalized)

        return EventConfig(
            initial_posts=result.get("initial_posts", []),
            scheduled_events=normalized_scheduled_events,
            hot_topics=result.get("hot_topics", []),
            narrative_direction=result.get("narrative_direction", "")
        )

    def _assign_posts_to_agents(
        self,
        posts: List[Dict[str, Any]],
        agent_configs: List[AgentActivityConfig]
    ) -> List[Dict[str, Any]]:
        if not posts:
            return []

        agents_by_type: Dict[str, List[AgentActivityConfig]] = {}
        for agent in agent_configs:
            etype = agent.entity_type.lower()
            if etype not in agents_by_type:
                agents_by_type[etype] = []
            agents_by_type[etype].append(agent)

        type_aliases = {
            "official": ["official", "university", "governmentagency", "government"],
            "university": ["university", "official"],
            "mediaoutlet": ["mediaoutlet", "media"],
            "student": ["student", "person"],
            "professor": ["professor", "expert", "teacher"],
            "alumni": ["alumni", "person"],
            "organization": ["organization", "ngo", "company", "group"],
            "person": ["person", "student", "alumni"],
        }

        used_indices: Dict[str, int] = {}
        updated_posts = []
        for post in posts:
            poster_type = str(post.get("poster_type", "") or "").lower()
            matched_agent_id = None

            if poster_type in agents_by_type:
                agents = agents_by_type[poster_type]
                idx = used_indices.get(poster_type, 0) % len(agents)
                matched_agent_id = agents[idx].agent_id
                used_indices[poster_type] = idx + 1
            else:
                for alias_key, aliases in type_aliases.items():
                    if poster_type in aliases or alias_key == poster_type:
                        for alias in aliases:
                            if alias in agents_by_type:
                                agents = agents_by_type[alias]
                                idx = used_indices.get(alias, 0) % len(agents)
                                matched_agent_id = agents[idx].agent_id
                                used_indices[alias] = idx + 1
                                break
                    if matched_agent_id is not None:
                        break

            if matched_agent_id is None:
                logger.warning(f"No matching agent found for type '{poster_type}', using highest influence agent")
                if agent_configs:
                    sorted_agents = sorted(agent_configs, key=lambda a: a.influence_weight, reverse=True)
                    matched_agent_id = sorted_agents[0].agent_id
                else:
                    matched_agent_id = 0

            updated_posts.append({
                **post,
                "poster_type": post.get("poster_type", "Unknown"),
                "poster_agent_id": matched_agent_id,
            })

        return updated_posts

    def _match_agent_id_by_type(
        self,
        poster_type: str,
        agent_configs: List[AgentActivityConfig],
        used_indices: Optional[Dict[str, int]] = None,
    ) -> Optional[int]:
        agents_by_type: Dict[str, List[AgentActivityConfig]] = {}
        for agent in agent_configs:
            etype = agent.entity_type.lower()
            agents_by_type.setdefault(etype, []).append(agent)

        type_aliases = {
            "official": ["official", "university", "governmentagency", "government"],
            "university": ["university", "official"],
            "mediaoutlet": ["mediaoutlet", "media"],
            "student": ["student", "person"],
            "professor": ["professor", "expert", "teacher"],
            "alumni": ["alumni", "person"],
            "organization": ["organization", "ngo", "company", "group"],
            "person": ["person", "student", "alumni"],
        }

        used_indices = used_indices if used_indices is not None else {}
        poster_type = str(poster_type or "").lower()
        if poster_type in agents_by_type:
            agents = agents_by_type[poster_type]
            idx = used_indices.get(poster_type, 0) % len(agents)
            used_indices[poster_type] = idx + 1
            return agents[idx].agent_id

        for alias_key, aliases in type_aliases.items():
            if poster_type in aliases or alias_key == poster_type:
                for alias in aliases:
                    if alias in agents_by_type:
                        agents = agents_by_type[alias]
                        idx = used_indices.get(alias, 0) % len(agents)
                        used_indices[alias] = idx + 1
                        return agents[idx].agent_id
        return None

    def _assign_initial_post_agents(
        self,
        event_config: EventConfig,
        agent_configs: List[AgentActivityConfig]
    ) -> EventConfig:
        event_config.initial_posts = self._assign_posts_to_agents(
            event_config.initial_posts,
            agent_configs,
        )
        for post in event_config.initial_posts:
            logger.info(
                f"Initial post assignment: poster_type='{str(post.get('poster_type', '')).lower()}' "
                f"-> agent_id={post.get('poster_agent_id')}"
            )

        scheduled_post_events = []
        passthrough_events = []
        for event in event_config.scheduled_events:
            if str(event.get("event_type", "")).lower() == "create_post" and event.get("content"):
                scheduled_post_events.append(event)
            else:
                passthrough_events.append(event)

        scheduled_post_events = self._assign_posts_to_agents(
            scheduled_post_events,
            agent_configs,
        )
        for event in scheduled_post_events:
            logger.info(
                f"Scheduled event assignment: event_type=create_post, "
                f"poster_type='{str(event.get('poster_type', '')).lower()}' "
                f"-> agent_id={event.get('poster_agent_id')}"
            )

        used_target_indices: Dict[str, int] = {}
        for event in passthrough_events:
            event_type = str(event.get("event_type", "")).lower()
            if event_type in {"create_comment", "create_thread"}:
                poster_agent_id = self._match_agent_id_by_type(
                    event.get("poster_type", ""),
                    agent_configs,
                )
                if poster_agent_id is not None:
                    event["poster_agent_id"] = poster_agent_id
                target_type = str(event.get("target_poster_type", "") or "")
                if target_type:
                    target_agent_id = self._match_agent_id_by_type(
                        target_type,
                        agent_configs,
                        used_indices=used_target_indices,
                    )
                    if target_agent_id is not None:
                        event["target_poster_agent_id"] = target_agent_id

        event_config.scheduled_events = scheduled_post_events + passthrough_events
        return event_config

    def _generate_agent_configs_batch(
        self,
        context: str,
        entities: List[EntityNode],
        start_idx: int,
        simulation_requirement: str
    ) -> List[AgentActivityConfig]:


        entity_list = []
        summary_len = self.AGENT_SUMMARY_LENGTH
        for i, e in enumerate(entities):
            entity_list.append({
                "agent_id": start_idx + i,
                "entity_name": e.name,
                "entity_type": e.get_entity_type() or "Unknown",
                "summary": e.summary[:summary_len] if e.summary else ""
            })

        prompt = f"""Based on the following information, generate social media activity config for each entity.

Simulation requirement: {simulation_requirement}

## Entity List
```json
{json.dumps(entity_list, ensure_ascii=False, indent=2)}
```

## Task
Generate activity config for each entity, noting:
- **Time follows Chinese daily patterns**: Almost no activity 0-5am, most active 7-10pm
- **Official institutions** (University/GovernmentAgency): Low activity (0.1-0.3), active during work hours (9-17), slow response (60-240 min), high influence (2.5-3.0)
- **Media** (MediaOutlet): Medium activity (0.4-0.6), active all day (8-23), fast response (5-30 min), high influence (2.0-2.5)
- **Individuals** (Student/Person/Alumni): High activity (0.6-0.9), mainly evening activity (18-23), fast response (1-15 min), low influence (0.8-1.2)
- **Public figures/Experts**: Medium activity (0.4-0.6), medium-high influence (1.5-2.0)

Return JSON format (no markdown):
{{
    "agent_configs": [
        {{
            "agent_id": <must match input>,
            "activity_level": <0.0-1.0>,
            "posts_per_hour": <posting frequency>,
            "comments_per_hour": <comment frequency>,
            "active_hours": [<active hours list, following Chinese daily patterns>],
            "response_delay_min": <min response delay in minutes>,
            "response_delay_max": <max response delay in minutes>,
            "sentiment_bias": <-1.0 to 1.0>,
            "stance": "<supportive/opposing/neutral/observer>",
            "influence_weight": <influence weight>
        }},
        ...
    ]
}}"""

        system_prompt = "You are a social media behavior analysis expert. Return pure JSON. Config must follow Chinese daily patterns."

        try:
            result = self._call_llm_with_retry(prompt, system_prompt)
            llm_configs = {cfg["agent_id"]: cfg for cfg in result.get("agent_configs", [])}
        except Exception as e:
            logger.warning(f"Agent config batch LLM generation failed: {e}, using rule-based generation")
            llm_configs = {}


        configs = []
        for i, entity in enumerate(entities):
            agent_id = start_idx + i
            cfg = llm_configs.get(agent_id, {})


            if not cfg:
                cfg = self._generate_agent_config_by_rule(entity)

            config = AgentActivityConfig(
                agent_id=agent_id,
                entity_uuid=entity.uuid,
                entity_name=entity.name,
                entity_type=entity.get_entity_type() or "Unknown",
                activity_level=cfg.get("activity_level", 0.5),
                posts_per_hour=cfg.get("posts_per_hour", 0.5),
                comments_per_hour=cfg.get("comments_per_hour", 1.0),
                active_hours=cfg.get("active_hours", list(range(9, 23))),
                response_delay_min=cfg.get("response_delay_min", 5),
                response_delay_max=cfg.get("response_delay_max", 60),
                sentiment_bias=cfg.get("sentiment_bias", 0.0),
                stance=cfg.get("stance", "neutral"),
                influence_weight=cfg.get("influence_weight", 1.0)
            )
            configs.append(config)

        return configs

    def _generate_agent_config_by_rule(self, entity: EntityNode) -> Dict[str, Any]:
        entity_type = (entity.get_entity_type() or "Unknown").lower()

        if entity_type in ["university", "governmentagency", "ngo"]:

            return {
                "activity_level": 0.2,
                "posts_per_hour": 0.1,
                "comments_per_hour": 0.05,
                "active_hours": list(range(9, 18)),
                "response_delay_min": 60,
                "response_delay_max": 240,
                "sentiment_bias": 0.0,
                "stance": "neutral",
                "influence_weight": 3.0
            }
        elif entity_type in ["mediaoutlet"]:

            return {
                "activity_level": 0.5,
                "posts_per_hour": 0.8,
                "comments_per_hour": 0.3,
                "active_hours": list(range(7, 24)),
                "response_delay_min": 5,
                "response_delay_max": 30,
                "sentiment_bias": 0.0,
                "stance": "observer",
                "influence_weight": 2.5
            }
        elif entity_type in ["professor", "expert", "official"]:

            return {
                "activity_level": 0.4,
                "posts_per_hour": 0.3,
                "comments_per_hour": 0.5,
                "active_hours": list(range(8, 22)),
                "response_delay_min": 15,
                "response_delay_max": 90,
                "sentiment_bias": 0.0,
                "stance": "neutral",
                "influence_weight": 2.0
            }
        elif entity_type in ["student"]:

            return {
                "activity_level": 0.8,
                "posts_per_hour": 0.6,
                "comments_per_hour": 1.5,
                "active_hours": [8, 9, 10, 11, 12, 13, 18, 19, 20, 21, 22, 23],
                "response_delay_min": 1,
                "response_delay_max": 15,
                "sentiment_bias": 0.0,
                "stance": "neutral",
                "influence_weight": 0.8
            }
        elif entity_type in ["alumni"]:

            return {
                "activity_level": 0.6,
                "posts_per_hour": 0.4,
                "comments_per_hour": 0.8,
                "active_hours": [12, 13, 19, 20, 21, 22, 23],
                "response_delay_min": 5,
                "response_delay_max": 30,
                "sentiment_bias": 0.0,
                "stance": "neutral",
                "influence_weight": 1.0
            }
        else:

            return {
                "activity_level": 0.7,
                "posts_per_hour": 0.5,
                "comments_per_hour": 1.2,
                "active_hours": [9, 10, 11, 12, 13, 18, 19, 20, 21, 22, 23],
                "response_delay_min": 2,
                "response_delay_max": 20,
                "sentiment_bias": 0.0,
                "stance": "neutral",
                "influence_weight": 1.0
            }
