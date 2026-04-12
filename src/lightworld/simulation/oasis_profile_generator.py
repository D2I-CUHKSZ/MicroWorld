
import json
import random
import re
import time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime

from openai import OpenAI
from zep_cloud.client import Zep

from lightworld.config.settings import Config
from lightworld.telemetry.logging_config import get_logger
from lightworld.graph.zep_entity_reader import EntityNode, ZepEntityReader

logger = get_logger('lightworld.oasis_profile')


@dataclass
class OasisAgentProfile:

    user_id: int
    user_name: str
    name: str
    bio: str
    persona: str


    karma: int = 1000


    friend_count: int = 100
    follower_count: int = 150
    statuses_count: int = 500


    age: Optional[int] = None
    gender: Optional[str] = None
    mbti: Optional[str] = None
    country: Optional[str] = None
    profession: Optional[str] = None
    interested_topics: List[str] = field(default_factory=list)


    source_entity_uuid: Optional[str] = None
    source_entity_type: Optional[str] = None

    created_at: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d"))

    def to_reddit_format(self) -> Dict[str, Any]:
        profile = {
            "user_id": self.user_id,
            "username": self.user_name,
            "name": self.name,
            "bio": self.bio,
            "persona": self.persona,
            "karma": self.karma,
            "created_at": self.created_at,
        }


        if self.age:
            profile["age"] = self.age
        if self.gender:
            profile["gender"] = self.gender
        if self.mbti:
            profile["mbti"] = self.mbti
        if self.country:
            profile["country"] = self.country
        if self.profession:
            profile["profession"] = self.profession
        if self.interested_topics:
            profile["interested_topics"] = self.interested_topics

        return profile

    def to_twitter_format(self) -> Dict[str, Any]:
        profile = {
            "user_id": self.user_id,
            "username": self.user_name,
            "name": self.name,
            "bio": self.bio,
            "persona": self.persona,
            "friend_count": self.friend_count,
            "follower_count": self.follower_count,
            "statuses_count": self.statuses_count,
            "created_at": self.created_at,
        }


        if self.age:
            profile["age"] = self.age
        if self.gender:
            profile["gender"] = self.gender
        if self.mbti:
            profile["mbti"] = self.mbti
        if self.country:
            profile["country"] = self.country
        if self.profession:
            profile["profession"] = self.profession
        if self.interested_topics:
            profile["interested_topics"] = self.interested_topics

        return profile

    def to_dict(self) -> Dict[str, Any]:
        return {
            "user_id": self.user_id,
            "user_name": self.user_name,
            "name": self.name,
            "bio": self.bio,
            "persona": self.persona,
            "karma": self.karma,
            "friend_count": self.friend_count,
            "follower_count": self.follower_count,
            "statuses_count": self.statuses_count,
            "age": self.age,
            "gender": self.gender,
            "mbti": self.mbti,
            "country": self.country,
            "profession": self.profession,
            "interested_topics": self.interested_topics,
            "source_entity_uuid": self.source_entity_uuid,
            "source_entity_type": self.source_entity_type,
            "created_at": self.created_at,
        }


class OasisProfileGenerator:


    MBTI_TYPES = [
        "INTJ", "INTP", "ENTJ", "ENTP",
        "INFJ", "INFP", "ENFJ", "ENFP",
        "ISTJ", "ISFJ", "ESTJ", "ESFJ",
        "ISTP", "ISFP", "ESTP", "ESFP"
    ]


    COUNTRIES = [
        "China", "US", "UK", "Japan", "Germany", "France",
        "Canada", "Australia", "Brazil", "India", "South Korea"
    ]


    INDIVIDUAL_ENTITY_TYPES = [
        "student", "alumni", "professor", "person", "publicfigure",
        "expert", "faculty", "official", "journalist", "activist"
    ]


    GROUP_ENTITY_TYPES = [
        "university", "governmentagency", "organization", "ngo",
        "mediaoutlet", "company", "institution", "group", "community"
    ]
    PERSONA_CHAR_MIN = 300
    PERSONA_CHAR_MAX = 600
    BIO_CHAR_MAX = 120

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model_name: Optional[str] = None,
        zep_api_key: Optional[str] = None,
        graph_id: Optional[str] = None
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


        self.zep_api_key = zep_api_key or Config.ZEP_API_KEY
        self.zep_client = None
        self.graph_id = graph_id

        if self.zep_api_key:
            try:
                self.zep_client = Zep(api_key=self.zep_api_key)
            except Exception as e:
                logger.warning(f"Failed to initialize Zep client: {e}")

    def generate_profile_from_entity(
        self,
        entity: EntityNode,
        user_id: int,
        use_llm: bool = True
    ) -> OasisAgentProfile:
        entity_type = entity.get_entity_type() or "Entity"


        name = entity.name
        user_name = self._generate_username(name)


        context = self._build_entity_context(entity)

        is_synthetic_population = bool((entity.attributes or {}).get("synthetic_population"))

        if is_synthetic_population:
            profile_data = self._generate_synthetic_population_profile(
                entity_name=name,
                entity_type=entity_type,
                entity_summary=entity.summary,
                entity_attributes=entity.attributes,
            )
        elif use_llm:

            profile_data = self._generate_profile_with_llm(
                entity_name=name,
                entity_type=entity_type,
                entity_summary=entity.summary,
                entity_attributes=entity.attributes,
                context=context
            )
        else:

            profile_data = self._generate_profile_rule_based(
                entity_name=name,
                entity_type=entity_type,
                entity_summary=entity.summary,
                entity_attributes=entity.attributes
            )
        profile_data = self._normalize_profile_data(
            entity_name=name,
            entity_type=entity_type,
            entity_summary=entity.summary,
            entity_attributes=entity.attributes,
            profile_data=profile_data,
        )

        return OasisAgentProfile(
            user_id=user_id,
            user_name=user_name,
            name=name,
            bio=profile_data.get("bio", f"{entity_type}: {name}"),
            persona=profile_data.get("persona", entity.summary or f"A {entity_type} named {name}."),
            karma=profile_data.get("karma", random.randint(500, 5000)),
            friend_count=profile_data.get("friend_count", random.randint(50, 500)),
            follower_count=profile_data.get("follower_count", random.randint(100, 1000)),
            statuses_count=profile_data.get("statuses_count", random.randint(100, 2000)),
            age=profile_data.get("age"),
            gender=profile_data.get("gender"),
            mbti=profile_data.get("mbti"),
            country=profile_data.get("country"),
            profession=profile_data.get("profession"),
            interested_topics=profile_data.get("interested_topics", []),
            source_entity_uuid=entity.uuid,
            source_entity_type=entity_type,
        )

    def _generate_username(self, name: str) -> str:

        username = name.lower().replace(" ", "_")
        username = ''.join(c for c in username if c.isalnum() or c == '_')


        suffix = random.randint(100, 999)
        return f"{username}_{suffix}"

    def _search_zep_for_entity(self, entity: EntityNode) -> Dict[str, Any]:
        import concurrent.futures

        if not self.zep_client:
            return {"facts": [], "node_summaries": [], "context": ""}

        entity_name = entity.name

        results = {
            "facts": [],
            "node_summaries": [],
            "context": ""
        }


        if not self.graph_id:
            logger.debug(f"Skipping Zep search: graph_id not set")
            return results

        comprehensive_query = f"All information, activities, events, relationships and background about {entity_name}"

        def search_edges():
            max_retries = 3
            last_exception = None
            delay = 2.0

            for attempt in range(max_retries):
                try:
                    return self.zep_client.graph.search(
                        query=comprehensive_query,
                        graph_id=self.graph_id,
                        limit=30,
                        scope="edges",
                        reranker="rrf"
                    )
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        logger.debug(f"Zep edge search attempt {attempt + 1} failed: {str(e)[:80]}, retrying...")
                        time.sleep(delay)
                        delay *= 2
                    else:
                        logger.debug(f"Zep edge search failed after {max_retries} attempts: {e}")
            return None

        def search_nodes():
            max_retries = 3
            last_exception = None
            delay = 2.0

            for attempt in range(max_retries):
                try:
                    return self.zep_client.graph.search(
                        query=comprehensive_query,
                        graph_id=self.graph_id,
                        limit=20,
                        scope="nodes",
                        reranker="rrf"
                    )
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        logger.debug(f"Zep node search attempt {attempt + 1} failed: {str(e)[:80]}, retrying...")
                        time.sleep(delay)
                        delay *= 2
                    else:
                        logger.debug(f"Zep node search failed after {max_retries} attempts: {e}")
            return None

        try:

            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                edge_future = executor.submit(search_edges)
                node_future = executor.submit(search_nodes)


                edge_result = edge_future.result(timeout=30)
                node_result = node_future.result(timeout=30)


            all_facts = set()
            if edge_result and hasattr(edge_result, 'edges') and edge_result.edges:
                for edge in edge_result.edges:
                    if hasattr(edge, 'fact') and edge.fact:
                        all_facts.add(edge.fact)
            results["facts"] = list(all_facts)


            all_summaries = set()
            if node_result and hasattr(node_result, 'nodes') and node_result.nodes:
                for node in node_result.nodes:
                    if hasattr(node, 'summary') and node.summary:
                        all_summaries.add(node.summary)
                    if hasattr(node, 'name') and node.name and node.name != entity_name:
                        all_summaries.add(f"Related entity: {node.name}")
            results["node_summaries"] = list(all_summaries)


            context_parts = []
            if results["facts"]:
                context_parts.append("Facts:\n" + "\n".join(f"- {f}" for f in results["facts"][:20]))
            if results["node_summaries"]:
                context_parts.append("Related entities:\n" + "\n".join(f"- {s}" for s in results["node_summaries"][:10]))
            results["context"] = "\n\n".join(context_parts)

            logger.info(f"Zep hybrid search complete: {entity_name}, retrieved {len(results['facts'])} facts, {len(results['node_summaries'])} related nodes")

        except concurrent.futures.TimeoutError:
            logger.warning(f"Zep search timed out ({entity_name})")
        except Exception as e:
            logger.warning(f"Zep search failed ({entity_name}): {e}")

        return results

    def _build_entity_context(self, entity: EntityNode) -> str:
        context_parts = []


        if entity.attributes:
            attrs = []
            for key, value in entity.attributes.items():
                if value and str(value).strip():
                    attrs.append(f"- {key}: {value}")
            if attrs:
                context_parts.append("### Entity Attributes\n" + "\n".join(attrs))


        existing_facts = set()
        if entity.related_edges:
            relationships = []
            for edge in entity.related_edges:
                fact = edge.get("fact", "")
                edge_name = edge.get("edge_name", "")
                direction = edge.get("direction", "")

                if fact:
                    relationships.append(f"- {fact}")
                    existing_facts.add(fact)
                elif edge_name:
                    if direction == "outgoing":
                        relationships.append(f"- {entity.name} --[{edge_name}]--> (related entity)")
                    else:
                        relationships.append(f"- (related entity) --[{edge_name}]--> {entity.name}")

            if relationships:
                context_parts.append("### Related Facts and Relationships\n" + "\n".join(relationships))


        if entity.related_nodes:
            related_info = []
            for node in entity.related_nodes:
                node_name = node.get("name", "")
                node_labels = node.get("labels", [])
                node_summary = node.get("summary", "")


                custom_labels = [l for l in node_labels if l not in ["Entity", "Node"]]
                label_str = f" ({', '.join(custom_labels)})" if custom_labels else ""

                if node_summary:
                    related_info.append(f"- **{node_name}**{label_str}: {node_summary}")
                else:
                    related_info.append(f"- **{node_name}**{label_str}")

            if related_info:
                context_parts.append("### Related Entity Info\n" + "\n".join(related_info))


        zep_results = self._search_zep_for_entity(entity)

        if zep_results.get("facts"):

            new_facts = [f for f in zep_results["facts"] if f not in existing_facts]
            if new_facts:
                context_parts.append("### Facts Retrieved from Zep\n" + "\n".join(f"- {f}" for f in new_facts[:15]))

        if zep_results.get("node_summaries"):
            context_parts.append("### Related Nodes Retrieved from Zep\n" + "\n".join(f"- {s}" for s in zep_results["node_summaries"][:10]))

        return "\n\n".join(context_parts)

    def _is_individual_entity(self, entity_type: str) -> bool:
        return entity_type.lower() in self.INDIVIDUAL_ENTITY_TYPES

    def _is_group_entity(self, entity_type: str) -> bool:
        return entity_type.lower() in self.GROUP_ENTITY_TYPES

    def _generate_profile_with_llm(
        self,
        entity_name: str,
        entity_type: str,
        entity_summary: str,
        entity_attributes: Dict[str, Any],
        context: str
    ) -> Dict[str, Any]:

        is_individual = self._is_individual_entity(entity_type)

        if is_individual:
            prompt = self._build_individual_persona_prompt(
                entity_name, entity_type, entity_summary, entity_attributes, context
            )
        else:
            prompt = self._build_group_persona_prompt(
                entity_name, entity_type, entity_summary, entity_attributes, context
            )


        max_attempts = 3
        last_error = None

        for attempt in range(max_attempts):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": self._get_system_prompt(is_individual)},
                        {"role": "user", "content": prompt}
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.7 - (attempt * 0.1)

                )

                content = response.choices[0].message.content


                finish_reason = response.choices[0].finish_reason
                if finish_reason == 'length':
                    logger.warning(f"LLM output truncated (attempt {attempt+1}), attempting fix...")
                    content = self._fix_truncated_json(content)


                try:
                    result = json.loads(content)


                    if "bio" not in result or not result["bio"]:
                        result["bio"] = entity_summary[:200] if entity_summary else f"{entity_type}: {entity_name}"
                    if "persona" not in result or not result["persona"]:
                        result["persona"] = entity_summary or f"{entity_name} is a {entity_type}."

                    return result

                except json.JSONDecodeError as je:
                    logger.warning(f"JSON parse failed (attempt {attempt+1}): {str(je)[:80]}")


                    result = self._try_fix_json(content, entity_name, entity_type, entity_summary)
                    if result.get("_fixed"):
                        del result["_fixed"]
                        return result

                    last_error = je

            except Exception as e:
                logger.warning(f"LLM call failed (attempt {attempt+1}): {str(e)[:80]}")
                last_error = e
                import time
                time.sleep(1 * (attempt + 1))

        logger.warning(f"LLM profile generation failed ({max_attempts} attempts): {last_error}, falling back to rule-based")
        return self._generate_profile_rule_based(
            entity_name, entity_type, entity_summary, entity_attributes
        )

    def _fix_truncated_json(self, content: str) -> str:
        import re


        content = content.strip()


        open_braces = content.count('{') - content.count('}')
        open_brackets = content.count('[') - content.count(']')


        if content and content[-1] not in '",}]':

            content += '"'


        content += ']' * open_brackets
        content += '}' * open_braces

        return content

    def _try_fix_json(self, content: str, entity_name: str, entity_type: str, entity_summary: str = "") -> Dict[str, Any]:
        import re


        content = self._fix_truncated_json(content)


        json_match = re.search(r'\{[\s\S]*\}', content)
        if json_match:
            json_str = json_match.group()


            def fix_string_newlines(match):
                s = match.group(0)

                s = s.replace('\n', ' ').replace('\r', ' ')

                s = re.sub(r'\s+', ' ', s)
                return s


            json_str = re.sub(r'"[^"\\]*(?:\\.[^"\\]*)*"', fix_string_newlines, json_str)


            try:
                result = json.loads(json_str)
                result["_fixed"] = True
                return result
            except json.JSONDecodeError as e:

                try:

                    json_str = re.sub(r'[\x00-\x1f\x7f-\x9f]', ' ', json_str)

                    json_str = re.sub(r'\s+', ' ', json_str)
                    result = json.loads(json_str)
                    result["_fixed"] = True
                    return result
                except:
                    pass


        bio_match = re.search(r'"bio"\s*:\s*"([^"]*)"', content)
        persona_match = re.search(r'"persona"\s*:\s*"([^"]*)', content)

        bio = bio_match.group(1) if bio_match else (entity_summary[:200] if entity_summary else f"{entity_type}: {entity_name}")
        persona = persona_match.group(1) if persona_match else (entity_summary or f"{entity_name} is a {entity_type}.")


        if bio_match or persona_match:
            logger.info(f"Extracted partial info from corrupted JSON")
            return {
                "bio": bio,
                "persona": persona,
                "_fixed": True
            }


        logger.warning(f"JSON repair failed, returning basic structure")
        return {
            "bio": entity_summary[:200] if entity_summary else f"{entity_type}: {entity_name}",
            "persona": entity_summary or f"{entity_name} is a {entity_type}."
        }

    def _get_system_prompt(self, is_individual: bool) -> str:
        base_prompt = (
            "You are a social media user profile generation expert. "
            "The goal is not to write a biography, but to generate a short, stable persona card that drives behavioral differentiation. "
            "Avoid fabricating precise document numbers, detailed resumes, overly long backstories, or details unsupported by context. "
            "You must return valid JSON. All string values must not contain unescaped newlines. Use Chinese for content."
        )
        return base_prompt

    def _build_individual_persona_prompt(
        self,
        entity_name: str,
        entity_type: str,
        entity_summary: str,
        entity_attributes: Dict[str, Any],
        context: str
    ) -> str:

        attrs_str = json.dumps(entity_attributes, ensure_ascii=False) if entity_attributes else "None"
        context_str = context[:3000] if context else "No additional context"

        return f"""Generate a personal account persona card for opinion simulation. Requirements: realistic, restrained, distinguishable. Do not write a long biography.

Entity name: {entity_name}
Entity type: {entity_type}
Entity summary: {entity_summary}
Entity attributes: {attrs_str}

Context:
{context_str}

Generate JSON with the following fields:

1. bio: Social media bio, 60-120 characters
2. persona: 300-600 characters of plain text, focusing on:
   - Identity and role positioning
   - Existing stance/preference on events
   - Speaking style and common expressions
   - Interaction habits (prefers posting, replying, reposting, fact-checking, or lurking)
   - What triggers them, what makes them silent
   - 1-2 memory anchors related to the event
3. age: Age as integer
4. gender: Must be English: "male" or "female"
5. mbti: MBTI type (e.g. INTJ, ENFP)
6. country: Country (use Chinese, e.g. "中国")
7. profession: Profession
8. interested_topics: Array of interested topics (4-7 items)

Important:
- All field values must be strings or numbers, no newlines
- persona must be a coherent text description
- Use Chinese for content (except gender must be English male/female)
- Content must be consistent with entity information
- age must be a valid integer, gender must be "male" or "female"
- Do not fabricate detailed education/family/resume details
- Do not turn the account into a legal commentary essay generator
"""

    def _build_group_persona_prompt(
        self,
        entity_name: str,
        entity_type: str,
        entity_summary: str,
        entity_attributes: Dict[str, Any],
        context: str
    ) -> str:

        attrs_str = json.dumps(entity_attributes, ensure_ascii=False) if entity_attributes else "None"
        context_str = context[:3000] if context else "No additional context"

        return f"""Generate an account persona for an organization/group entity for opinion simulation. Requirements: professional, restrained, clearly distinguishable. Do not write a long institutional history.

Entity name: {entity_name}
Entity type: {entity_type}
Entity summary: {entity_summary}
Entity attributes: {attrs_str}

Context:
{context_str}

Generate JSON with the following fields:

1. bio: Official account bio, 60-120 characters, professional and appropriate
2. persona: 300-600 characters of plain text, focusing on:
   - Account positioning and target audience
   - Speaking style and taboo expressions
   - Default approach to controversies
   - Prefers publishing, reposting, responding, or staying silent
   - 1-2 institutional memory anchors related to the event
3. age: Fixed at 30 (virtual age for institutional accounts)
4. gender: Fixed as "other" (institutional accounts use other for non-personal)
5. mbti: MBTI type to describe account style, e.g. ISTJ for rigorous and conservative
6. country: Country (use Chinese, e.g. "中国")
7. profession: Institutional function description
8. interested_topics: Array of focus areas (4-7 items)

Important:
- All field values must be strings or numbers, no null values
- persona must be a coherent text description, no newlines
- Use Chinese for content (except gender must be English "other")
- age must be integer 30, gender must be string "other"
- Institutional account speech must match its identity positioning
- Do not write long institutional commentary or fabricate complete organizational history"""

    def _normalize_profile_data(
        self,
        entity_name: str,
        entity_type: str,
        entity_summary: str,
        entity_attributes: Dict[str, Any],
        profile_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        result = dict(profile_data or {})

        def clean_text(value: Any) -> str:
            return re.sub(r"\s+", " ", str(value or "")).strip()

        bio = clean_text(result.get("bio", ""))
        persona = clean_text(result.get("persona", ""))
        if not bio:
            bio = clean_text(entity_summary[: self.BIO_CHAR_MAX] or f"{entity_type}: {entity_name}")
        if len(bio) > self.BIO_CHAR_MAX:
            bio = bio[: self.BIO_CHAR_MAX].rstrip("，,。；; ")

        if not persona:
            persona = clean_text(
                entity_summary
                or f"{entity_name} is a participant related to {entity_type}, with a cautious speaking style, focused on event developments."
            )
        if len(persona) < self.PERSONA_CHAR_MIN:
            extra_bits = [
                f"Account positioned as {clean_text(result.get('profession', '') or entity_type)}.",
                "Prefers concise expression, avoids lengthy background details.",
                "When facing controversy, prioritizes factual sources, interaction feedback, and personal stance boundaries.",
            ]
            persona = clean_text(persona + " " + " ".join(extra_bits))
        if len(persona) > self.PERSONA_CHAR_MAX:
            persona = persona[: self.PERSONA_CHAR_MAX].rstrip("，,。；; ")

        topics = result.get("interested_topics", []) or []
        if isinstance(topics, str):
            topics = [x.strip() for x in re.split(r"[，,;；\s]+", topics) if x.strip()]
        if not isinstance(topics, list):
            topics = []
        dedup_topics: List[str] = []
        seen = set()
        for item in topics:
            text = clean_text(item)
            if not text:
                continue
            key = text.lower()
            if key in seen:
                continue
            seen.add(key)
            dedup_topics.append(text[:24])
            if len(dedup_topics) >= 7:
                break
        if not dedup_topics:
            dedup_topics = self._infer_topics_from_entity(entity_name, entity_summary, entity_attributes)

        result["bio"] = bio
        result["persona"] = persona
        result["interested_topics"] = dedup_topics[:7]
        if result.get("profession") is None:
            result["profession"] = entity_type
        return result

    def _infer_topics_from_entity(
        self,
        entity_name: str,
        entity_summary: str,
        entity_attributes: Dict[str, Any],
    ) -> List[str]:
        candidates = [entity_name]
        candidates.extend(re.findall(r"[\u4e00-\u9fff]{2,8}", entity_summary or ""))
        for value in (entity_attributes or {}).values():
            if isinstance(value, str):
                candidates.extend(re.findall(r"[\u4e00-\u9fff]{2,8}", value))
        dedup: List[str] = []
        seen = set()
        for item in candidates:
            text = str(item or "").strip()
            if len(text) < 2:
                continue
            if text in {"实体", "相关信息", "事件相关", "entity", "related info", "event related"}:
                continue
            if text.lower() in seen:
                continue
            seen.add(text.lower())
            dedup.append(text)
            if len(dedup) >= 6:
                break
        return dedup or ["public affairs", "fact-checking", "platform interaction"]

    def _generate_synthetic_population_profile(
        self,
        entity_name: str,
        entity_type: str,
        entity_summary: str,
        entity_attributes: Dict[str, Any],
    ) -> Dict[str, Any]:
        segment = str((entity_attributes or {}).get("population_segment", "ordinary_user"))
        stance = str((entity_attributes or {}).get("stance_anchor", "neutral"))
        topics = list((entity_attributes or {}).get("topic_hints", []) or [])[:5]
        profession_map = {
            "campus_observer": "ordinary campus student",
            "fact_checker": "ordinary netizen",
            "emotional_bystander": "bystander-type social media user",
            "alumni_like": "alumni-type observer",
            "parent_view": "parent-perspective user",
            "amplifier": "trending-repost-type user",
        }
        speech_map = {
            "campus_observer": "posts short comments, follows threads, and reposts peer opinions",
            "fact_checker": "prefers questioning sources, timelines, and evidence screenshots",
            "emotional_bystander": "more emotional in expression, quick to take sides",
            "alumni_like": "proposes institutional-level suggestions, relatively restrained in expression",
            "parent_view": "more concerned about safety and order, sensitive to cyberbullying",
            "amplifier": "values trending speed, willing to spread but not necessarily dig deep",
        }
        bio = f"{profession_map.get(segment, 'ordinary social media user')}, follows {(', '.join(topics[:3]) or 'event developments')}."
        persona = (
            f"{entity_name} is a {profession_map.get(segment, 'ordinary social media user')}, "
            f"with a default stance leaning {stance} on the current event. "
            f"Their posts are usually short, more like a real passerby on the platform: {speech_map.get(segment, 'reacts instantly based on posts seen')}. "
            f"When controversy escalates, they tend to follow trending posts and peer discussions rather than independently writing long arguments; "
            f"if they see content involving {', '.join(topics[:2]) or 'procedural justice'}, they are more willing to reply, repost, or follow up. "
            "Memory anchors mainly come from trending posts, peer interactions, and personal emotional reactions, not systematic information gathering."
        )
        return {
            "bio": bio,
            "persona": persona,
            "age": random.randint(20, 38),
            "gender": random.choice(["male", "female"]),
            "mbti": random.choice(self.MBTI_TYPES),
            "country": "China",
            "profession": profession_map.get(segment, entity_type),
            "interested_topics": topics or ["event developments", "platform discussion", "fact-checking"],
            "friend_count": random.randint(30, 220),
            "follower_count": random.randint(20, 260),
            "statuses_count": random.randint(80, 1200),
            "karma": random.randint(80, 1600),
        }

    def _generate_profile_rule_based(
        self,
        entity_name: str,
        entity_type: str,
        entity_summary: str,
        entity_attributes: Dict[str, Any]
    ) -> Dict[str, Any]:


        entity_type_lower = entity_type.lower()

        if entity_type_lower in ["student", "alumni"]:
            return {
                "bio": f"{entity_name}, {entity_type} identity, follows campus issues and public discussions.",
                "persona": f"{entity_name} participates in discussions as a {entity_type}, expresses directly without over-elaboration, prefers speaking around personal experience, peer feedback, and event developments, often replies or reposts when seeing content related to education, fairness, or public opinion.",
                "age": random.randint(18, 30),
                "gender": random.choice(["male", "female"]),
                "mbti": random.choice(self.MBTI_TYPES),
                "country": "China",
                "profession": "student",
                "interested_topics": ["educational equity", "campus issues", "public discussion"],
            }

        elif entity_type_lower in ["publicfigure", "expert", "faculty"]:
            return {
                "bio": f"{entity_name}, has stable influence on public issues.",
                "persona": f"{entity_name} acts more like a professional commentator or opinion leader, usually approaches events from a professional angle, good at distilling viewpoints and setting agendas, but maintains some judgment distance, doesn't turn every post into a long essay.",
                "age": random.randint(35, 60),
                "gender": random.choice(["male", "female"]),
                "mbti": random.choice(["ENTJ", "INTJ", "ENTP", "INTP"]),
                "country": "China",
                "profession": entity_attributes.get("occupation", "expert"),
                "interested_topics": ["public affairs", "institutional discussion", "media commentary"],
            }

        elif entity_type_lower in ["mediaoutlet", "socialmediaplatform"]:
            return {
                "bio": f"{entity_name} official/media account, responsible for publishing news, forwarding developments, and brief commentary.",
                "persona": f"{entity_name}'s posts lean towards news bulletins, factual updates, and reference propagation, usually without deep personal emotion, preferring quick publishing, follow-ups, citations, and limited responses, intensifying references and tracking when topics trend.",
                "age": 30,
                "gender": "other",
                "mbti": "ISTJ",
                "country": "China",
                "profession": "media account",
                "interested_topics": ["news updates", "event developments", "public affairs"],
            }

        elif entity_type_lower in ["university", "governmentagency", "ngo", "organization"]:
            return {
                "bio": f"{entity_name} official account, publishes formal statements, announcements, and institutional positions.",
                "persona": f"{entity_name} as an institutional account prefers formal expression, announcement-style communication, and limited responses. When facing controversy, typically emphasizes procedures, rules, and follow-up arrangements, with conservative interaction tempo and minimal emotional confrontation.",
                "age": 30,
                "gender": "other",
                "mbti": "ISTJ",
                "country": "China",
                "profession": entity_type,
                "interested_topics": ["official announcements", "institutional response", "governance"],
            }

        else:

            return {
                "bio": entity_summary[:150] if entity_summary else f"{entity_type}: {entity_name}",
                "persona": entity_summary or f"{entity_name} participates in discussions around topics related to themselves, with concise expression, mainly reacting based on posts seen and limited context.",
                "age": random.randint(25, 50),
                "gender": random.choice(["male", "female"]),
                "mbti": random.choice(self.MBTI_TYPES),
                "country": "China",
                "profession": entity_type,
                "interested_topics": ["public discussion", "event developments", "social issues"],
            }

    def set_graph_id(self, graph_id: str):
        self.graph_id = graph_id

    def generate_profiles_from_entities(
        self,
        entities: List[EntityNode],
        use_llm: bool = True,
        progress_callback: Optional[callable] = None,
        graph_id: Optional[str] = None,
        parallel_count: int = 5,
        realtime_output_path: Optional[str] = None,
        output_platform: str = "reddit"
    ) -> List[OasisAgentProfile]:
        import concurrent.futures
        from threading import Lock


        if graph_id:
            self.graph_id = graph_id

        total = len(entities)
        profiles = [None] * total
        completed_count = [0]
        lock = Lock()


        def save_profiles_realtime():
            if not realtime_output_path:
                return

            with lock:

                existing_profiles = [p for p in profiles if p is not None]
                if not existing_profiles:
                    return

                try:
                    if output_platform == "reddit":

                        profiles_data = [p.to_reddit_format() for p in existing_profiles]
                        with open(realtime_output_path, 'w', encoding='utf-8') as f:
                            json.dump(profiles_data, f, ensure_ascii=False, indent=2)
                    else:

                        import csv
                        profiles_data = [p.to_twitter_format() for p in existing_profiles]
                        if profiles_data:
                            fieldnames = list(profiles_data[0].keys())
                            with open(realtime_output_path, 'w', encoding='utf-8', newline='') as f:
                                writer = csv.DictWriter(f, fieldnames=fieldnames)
                                writer.writeheader()
                                writer.writerows(profiles_data)
                except Exception as e:
                    logger.warning(f"Failed to save profiles in real-time: {e}")

        def generate_single_profile(idx: int, entity: EntityNode) -> tuple:
            entity_type = entity.get_entity_type() or "Entity"

            try:
                profile = self.generate_profile_from_entity(
                    entity=entity,
                    user_id=idx,
                    use_llm=use_llm
                )


                self._print_generated_profile(entity.name, entity_type, profile)

                return idx, profile, None

            except Exception as e:
                logger.error(f"Failed to generate profile for entity {entity.name}: {str(e)}")

                fallback_profile = OasisAgentProfile(
                    user_id=idx,
                    user_name=self._generate_username(entity.name),
                    name=entity.name,
                    bio=f"{entity_type}: {entity.name}",
                    persona=entity.summary or f"A participant in social discussions.",
                    source_entity_uuid=entity.uuid,
                    source_entity_type=entity_type,
                )
                return idx, fallback_profile, str(e)

        logger.info(f"Starting parallel profile generation for {total} agents (parallelism: {parallel_count})...")
        print(f"\n{'='*60}")
        print(f"Starting agent profile generation - {total} entities, parallelism: {parallel_count}")
        print(f"{'='*60}\n")


        with concurrent.futures.ThreadPoolExecutor(max_workers=parallel_count) as executor:

            future_to_entity = {
                executor.submit(generate_single_profile, idx, entity): (idx, entity)
                for idx, entity in enumerate(entities)
            }


            for future in concurrent.futures.as_completed(future_to_entity):
                idx, entity = future_to_entity[future]
                entity_type = entity.get_entity_type() or "Entity"

                try:
                    result_idx, profile, error = future.result()
                    profiles[result_idx] = profile

                    with lock:
                        completed_count[0] += 1
                        current = completed_count[0]


                    save_profiles_realtime()

                    if progress_callback:
                        progress_callback(
                            current,
                            total,
                            f"Completed {current}/{total}: {entity.name} ({entity_type})"
                        )

                    if error:
                        logger.warning(f"[{current}/{total}] {entity.name} using fallback profile: {error}")
                    else:
                        logger.info(f"[{current}/{total}] Successfully generated profile: {entity.name} ({entity_type})")

                except Exception as e:
                    logger.error(f"Exception processing entity {entity.name}: {str(e)}")
                    with lock:
                        completed_count[0] += 1
                    profiles[idx] = OasisAgentProfile(
                        user_id=idx,
                        user_name=self._generate_username(entity.name),
                        name=entity.name,
                        bio=f"{entity_type}: {entity.name}",
                        persona=entity.summary or "A participant in social discussions.",
                        source_entity_uuid=entity.uuid,
                        source_entity_type=entity_type,
                    )

                    save_profiles_realtime()

        print(f"\n{'='*60}")
        print(f"Profile generation complete! Generated {len([p for p in profiles if p])} agents")
        print(f"{'='*60}\n")

        return profiles

    def _print_generated_profile(self, entity_name: str, entity_type: str, profile: OasisAgentProfile):
        separator = "-" * 70


        topics_str = ', '.join(profile.interested_topics) if profile.interested_topics else 'None'

        output_lines = [
            f"\n{separator}",
            f"[Generated] {entity_name} ({entity_type})",
            f"{separator}",
            f"Username: {profile.user_name}",
            f"",
            f"[Bio]",
            f"{profile.bio}",
            f"",
            f"[Detailed Persona]",
            f"{profile.persona}",
            f"",
            f"[Basic Attributes]",
            f"Age: {profile.age} | Gender: {profile.gender} | MBTI: {profile.mbti}",
            f"Profession: {profile.profession} | Country: {profile.country}",
            f"Interested Topics: {topics_str}",
            separator
        ]

        output = "\n".join(output_lines)


        print(output)

    def save_profiles(
        self,
        profiles: List[OasisAgentProfile],
        file_path: str,
        platform: str = "reddit"
    ):
        if platform == "twitter":
            self._save_twitter_csv(profiles, file_path)
        else:
            self._save_reddit_json(profiles, file_path)

    def _save_twitter_csv(self, profiles: List[OasisAgentProfile], file_path: str):
        import csv


        if not file_path.endswith('.csv'):
            file_path = file_path.replace('.json', '.csv')

        def clean_text(value: str) -> str:
            return str(value or "").replace('\n', ' ').replace('\r', ' ').strip()

        def build_user_char(profile: OasisAgentProfile) -> str:
            parts: List[str] = []
            if profile.bio:
                parts.append(clean_text(profile.bio))
            if profile.persona and profile.persona != profile.bio:
                parts.append(clean_text(profile.persona))

            structured_bits: List[str] = []
            if profile.age:
                structured_bits.append(f"age={profile.age}")
            if profile.gender:
                structured_bits.append(f"gender={profile.gender}")
            if profile.mbti:
                structured_bits.append(f"mbti={profile.mbti}")
            if profile.country:
                structured_bits.append(f"country={profile.country}")
            if profile.profession:
                structured_bits.append(f"profession={profile.profession}")
            if profile.interested_topics:
                structured_bits.append(
                    "interested_topics=" + ", ".join(str(x) for x in profile.interested_topics[:8])
                )
            structured_bits.append(f"follower_count={profile.follower_count}")
            structured_bits.append(f"friend_count={profile.friend_count}")
            structured_bits.append(f"statuses_count={profile.statuses_count}")
            if profile.source_entity_uuid:
                structured_bits.append(f"source_entity_uuid={profile.source_entity_uuid}")
            if profile.source_entity_type:
                structured_bits.append(f"source_entity_type={profile.source_entity_type}")

            if structured_bits:
                parts.append("Structured profile: " + "; ".join(structured_bits) + ".")

            return clean_text(" ".join(p for p in parts if p))[:1600]

        with open(file_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)


            headers = [
                'user_id',
                'name',
                'username',
                'user_char',
                'description',
                'bio',
                'persona',
                'age',
                'gender',
                'mbti',
                'country',
                'profession',
                'interested_topics',
                'friend_count',
                'follower_count',
                'statuses_count',
                'source_entity_uuid',
                'source_entity_type',
            ]
            writer.writerow(headers)


            for idx, profile in enumerate(profiles):
                user_char = build_user_char(profile)
                description = clean_text(profile.bio)

                row = [
                    idx,
                    profile.name,
                    profile.user_name,
                    user_char,
                    description,
                    clean_text(profile.bio),
                    clean_text(profile.persona),
                    profile.age or "",
                    profile.gender or "",
                    profile.mbti or "",
                    profile.country or "",
                    profile.profession or "",
                    json.dumps(profile.interested_topics, ensure_ascii=False),
                    profile.friend_count,
                    profile.follower_count,
                    profile.statuses_count,
                    profile.source_entity_uuid or "",
                    profile.source_entity_type or "",
                ]
                writer.writerow(row)

        logger.info(f"Saved {len(profiles)} Twitter profiles to {file_path} (OASIS CSV format)")

    def _normalize_gender(self, gender: Optional[str]) -> str:
        if not gender:
            return "other"

        gender_lower = gender.lower().strip()


        gender_map = {
            "男": "male",    # Chinese: male
            "女": "female",  # Chinese: female
            "机构": "other", # Chinese: institution
            "其他": "other", # Chinese: other

            "male": "male",
            "female": "female",
            "other": "other",
        }

        return gender_map.get(gender_lower, "other")

    def _save_reddit_json(self, profiles: List[OasisAgentProfile], file_path: str):
        data = []
        for idx, profile in enumerate(profiles):

            item = {
                "user_id": profile.user_id if profile.user_id is not None else idx,
                "username": profile.user_name,
                "name": profile.name,
                "bio": profile.bio[:150] if profile.bio else f"{profile.name}",
                "persona": profile.persona or f"{profile.name} is a participant in social discussions.",
                "karma": profile.karma if profile.karma else 1000,
                "created_at": profile.created_at,

                "age": profile.age if profile.age else 30,
                "gender": self._normalize_gender(profile.gender),
                "mbti": profile.mbti if profile.mbti else "ISTJ",
                "country": profile.country if profile.country else "China",
            }


            if profile.profession:
                item["profession"] = profile.profession
            if profile.interested_topics:
                item["interested_topics"] = profile.interested_topics

            data.append(item)

        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        logger.info(f"Saved {len(profiles)} Reddit profiles to {file_path} (JSON format with user_id field)")


    def save_profiles_to_json(
        self,
        profiles: List[OasisAgentProfile],
        file_path: str,
        platform: str = "reddit"
    ):
        logger.warning("save_profiles_to_json is deprecated, use save_profiles instead")
        self.save_profiles(profiles, file_path, platform)
