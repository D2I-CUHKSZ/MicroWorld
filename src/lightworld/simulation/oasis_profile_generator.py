
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
            raise ValueError("LLM_API_KEY 未配置")

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
                logger.warning(f"Zep客户端初始化失败: {e}")

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
            logger.debug(f"跳过Zep检索：未设置graph_id")
            return results

        comprehensive_query = f"关于{entity_name}的所有信息、活动、事件、关系和背景"

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
                        logger.debug(f"Zep边搜索第 {attempt + 1} 次失败: {str(e)[:80]}, 重试中...")
                        time.sleep(delay)
                        delay *= 2
                    else:
                        logger.debug(f"Zep边搜索在 {max_retries} 次尝试后仍失败: {e}")
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
                        logger.debug(f"Zep节点搜索第 {attempt + 1} 次失败: {str(e)[:80]}, 重试中...")
                        time.sleep(delay)
                        delay *= 2
                    else:
                        logger.debug(f"Zep节点搜索在 {max_retries} 次尝试后仍失败: {e}")
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
                        all_summaries.add(f"相关实体: {node.name}")
            results["node_summaries"] = list(all_summaries)


            context_parts = []
            if results["facts"]:
                context_parts.append("事实信息:\n" + "\n".join(f"- {f}" for f in results["facts"][:20]))
            if results["node_summaries"]:
                context_parts.append("相关实体:\n" + "\n".join(f"- {s}" for s in results["node_summaries"][:10]))
            results["context"] = "\n\n".join(context_parts)

            logger.info(f"Zep混合检索完成: {entity_name}, 获取 {len(results['facts'])} 条事实, {len(results['node_summaries'])} 个相关节点")

        except concurrent.futures.TimeoutError:
            logger.warning(f"Zep检索超时 ({entity_name})")
        except Exception as e:
            logger.warning(f"Zep检索失败 ({entity_name}): {e}")

        return results

    def _build_entity_context(self, entity: EntityNode) -> str:
        context_parts = []


        if entity.attributes:
            attrs = []
            for key, value in entity.attributes.items():
                if value and str(value).strip():
                    attrs.append(f"- {key}: {value}")
            if attrs:
                context_parts.append("### 实体属性\n" + "\n".join(attrs))


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
                        relationships.append(f"- {entity.name} --[{edge_name}]--> (相关实体)")
                    else:
                        relationships.append(f"- (相关实体) --[{edge_name}]--> {entity.name}")

            if relationships:
                context_parts.append("### 相关事实和关系\n" + "\n".join(relationships))


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
                context_parts.append("### 关联实体信息\n" + "\n".join(related_info))


        zep_results = self._search_zep_for_entity(entity)

        if zep_results.get("facts"):

            new_facts = [f for f in zep_results["facts"] if f not in existing_facts]
            if new_facts:
                context_parts.append("### Zep检索到的事实信息\n" + "\n".join(f"- {f}" for f in new_facts[:15]))

        if zep_results.get("node_summaries"):
            context_parts.append("### Zep检索到的相关节点\n" + "\n".join(f"- {s}" for s in zep_results["node_summaries"][:10]))

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
                    logger.warning(f"LLM输出被截断 (attempt {attempt+1}), 尝试修复...")
                    content = self._fix_truncated_json(content)


                try:
                    result = json.loads(content)


                    if "bio" not in result or not result["bio"]:
                        result["bio"] = entity_summary[:200] if entity_summary else f"{entity_type}: {entity_name}"
                    if "persona" not in result or not result["persona"]:
                        result["persona"] = entity_summary or f"{entity_name}是一个{entity_type}。"

                    return result

                except json.JSONDecodeError as je:
                    logger.warning(f"JSON解析失败 (attempt {attempt+1}): {str(je)[:80]}")


                    result = self._try_fix_json(content, entity_name, entity_type, entity_summary)
                    if result.get("_fixed"):
                        del result["_fixed"]
                        return result

                    last_error = je

            except Exception as e:
                logger.warning(f"LLM调用失败 (attempt {attempt+1}): {str(e)[:80]}")
                last_error = e
                import time
                time.sleep(1 * (attempt + 1))

        logger.warning(f"LLM生成人设失败（{max_attempts}次尝试）: {last_error}, 使用规则生成")
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
        persona = persona_match.group(1) if persona_match else (entity_summary or f"{entity_name}是一个{entity_type}。")


        if bio_match or persona_match:
            logger.info(f"从损坏的JSON中提取了部分信息")
            return {
                "bio": bio,
                "persona": persona,
                "_fixed": True
            }


        logger.warning(f"JSON修复失败，返回基础结构")
        return {
            "bio": entity_summary[:200] if entity_summary else f"{entity_type}: {entity_name}",
            "persona": entity_summary or f"{entity_name}是一个{entity_type}。"
        }

    def _get_system_prompt(self, is_individual: bool) -> str:
        base_prompt = (
            "你是社交媒体用户画像生成专家。"
            "目标不是写人物小传，而是生成短而稳定、可驱动行为差异的人设卡。"
            "请避免臆造精确文号、详细履历、过长背景故事和无法从上下文支持的细节。"
            "必须返回有效JSON，所有字符串值不能包含未转义换行符。使用中文。"
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

        attrs_str = json.dumps(entity_attributes, ensure_ascii=False) if entity_attributes else "无"
        context_str = context[:3000] if context else "无额外上下文"

        return f"""为实体生成用于舆论模拟的个人账号人设卡，要求真实、克制、可区分，不要写成长篇传记。

实体名称: {entity_name}
实体类型: {entity_type}
实体摘要: {entity_summary}
实体属性: {attrs_str}

上下文信息:
{context_str}

请生成JSON，包含以下字段:

1. bio: 社交媒体简介，60-120字
2. persona: 300-600字的纯文本，重点包含：
   - 身份与角色定位
   - 对事件的既有立场/偏好
   - 发言风格与常见表达方式
   - 互动习惯（更爱发帖、回复、转发、求证还是围观）
   - 容易被什么触发，看到什么会沉默
   - 与事件相关的1-2个记忆锚点
3. age: 年龄数字（必须是整数）
4. gender: 性别，必须是英文: "male" 或 "female"
5. mbti: MBTI类型（如INTJ、ENFP等）
6. country: 国家（使用中文，如"中国"）
7. profession: 职业
8. interested_topics: 感兴趣话题数组（4-7个）

重要:
- 所有字段值必须是字符串或数字，不要使用换行符
- persona必须是一段连贯的文字描述
- 使用中文（除了gender字段必须用英文male/female）
- 内容要与实体信息保持一致
- age必须是有效的整数，gender必须是"male"或"female"
- 不要虚构过细的教育/家庭/履历细节
- 不要把账号写成法律评论长文生成器
"""

    def _build_group_persona_prompt(
        self,
        entity_name: str,
        entity_type: str,
        entity_summary: str,
        entity_attributes: Dict[str, Any],
        context: str
    ) -> str:

        attrs_str = json.dumps(entity_attributes, ensure_ascii=False) if entity_attributes else "无"
        context_str = context[:3000] if context else "无额外上下文"

        return f"""为机构/群体实体生成用于舆论模拟的账号设定，要求专业、克制、区分度清晰，不要写成长篇机构史。

实体名称: {entity_name}
实体类型: {entity_type}
实体摘要: {entity_summary}
实体属性: {attrs_str}

上下文信息:
{context_str}

请生成JSON，包含以下字段:

1. bio: 官方账号简介，60-120字，专业得体
2. persona: 300-600字的纯文本，重点包含:
   - 账号定位与面向人群
   - 发言风格与禁忌表达
   - 对争议的默认处理方式
   - 更偏好发布、转引、回应还是沉默
   - 与当前事件相关的1-2个机构记忆锚点
3. age: 固定填30（机构账号的虚拟年龄）
4. gender: 固定填"other"（机构账号使用other表示非个人）
5. mbti: MBTI类型，用于描述账号风格，如ISTJ代表严谨保守
6. country: 国家（使用中文，如"中国"）
7. profession: 机构职能描述
8. interested_topics: 关注领域数组（4-7个）

重要:
- 所有字段值必须是字符串或数字，不允许null值
- persona必须是一段连贯的文字描述，不要使用换行符
- 使用中文（除了gender字段必须用英文"other"）
- age必须是整数30，gender必须是字符串"other"
- 机构账号发言要符合其身份定位
- 不要写成长篇制度评论或虚构完整组织史"""

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
                or f"{entity_name}是与{entity_type}相关的参与者，发言风格谨慎，关注事件进展。"
            )
        if len(persona) < self.PERSONA_CHAR_MIN:
            extra_bits = [
                f"账号定位是{clean_text(result.get('profession', '') or entity_type)}。",
                "表达倾向简洁，不会长篇堆砌背景细节。",
                "面对争议更看重事实来源、互动反馈和自身立场边界。",
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
            if text in {"实体", "相关信息", "事件相关"}:
                continue
            if text.lower() in seen:
                continue
            seen.add(text.lower())
            dedup.append(text)
            if len(dedup) >= 6:
                break
        return dedup or ["公共议题", "事实核查", "平台互动"]

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
            "campus_observer": "普通在校学生",
            "fact_checker": "普通网民",
            "emotional_bystander": "围观型社交媒体用户",
            "alumni_like": "校友型观察者",
            "parent_view": "家长视角用户",
            "amplifier": "热点转发型用户",
        }
        speech_map = {
            "campus_observer": "会短评、跟帖，也会转发同学观点",
            "fact_checker": "偏好追问信源、时间线和证据截图",
            "emotional_bystander": "表达更情绪化，容易快速站队",
            "alumni_like": "会提出制度层面的建议，表达相对克制",
            "parent_view": "更关心安全与秩序，对网暴敏感",
            "amplifier": "重视热点速度，愿意扩散但不一定深挖",
        }
        bio = f"{profession_map.get(segment, '普通社交平台用户')}，关注{('、'.join(topics[:3]) or '事件进展')}。"
        persona = (
            f"{entity_name}是一个{profession_map.get(segment, '普通社交平台用户')}，"
            f"围绕当前事件的默认立场偏{stance}。"
            f"其发言通常不长，更像平台里的真实路人：{speech_map.get(segment, '根据看到的帖子做即时反应')}。"
            f"在争议升级时，他/她会优先跟随热帖和熟人讨论，而不是独立展开长篇论证；"
            f"如果看到涉及{'、'.join(topics[:2]) or '程序正义'}的内容，会更愿意参与回复、转发或追问。"
            "记忆锚点主要来自平台上的高热帖子、同伴互动和个人情绪反应，而不是系统化资料整理。"
        )
        return {
            "bio": bio,
            "persona": persona,
            "age": random.randint(20, 38),
            "gender": random.choice(["male", "female"]),
            "mbti": random.choice(self.MBTI_TYPES),
            "country": "中国",
            "profession": profession_map.get(segment, entity_type),
            "interested_topics": topics or ["事件进展", "平台讨论", "事实核查"],
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
                "bio": f"{entity_name}，{entity_type}身份，关注校园议题与公共讨论。",
                "persona": f"{entity_name}以{entity_type}身份参与讨论，表达直接但不过度铺陈，更愿意围绕身边经验、同伴反馈和事件进展发言，常在看到与教育、公平或舆情相关内容时参与回复或转发。",
                "age": random.randint(18, 30),
                "gender": random.choice(["male", "female"]),
                "mbti": random.choice(self.MBTI_TYPES),
                "country": "中国",
                "profession": "学生",
                "interested_topics": ["教育公平", "校园议题", "公共讨论"],
            }

        elif entity_type_lower in ["publicfigure", "expert", "faculty"]:
            return {
                "bio": f"{entity_name}，在公共议题上有稳定发言影响力。",
                "persona": f"{entity_name}更像专业评论者或意见领袖，通常从专业角度切入事件，善于提炼观点、设置议题，但会保留一定判断距离，不会把每次发言都写成长文。",
                "age": random.randint(35, 60),
                "gender": random.choice(["male", "female"]),
                "mbti": random.choice(["ENTJ", "INTJ", "ENTP", "INTP"]),
                "country": "中国",
                "profession": entity_attributes.get("occupation", "专家"),
                "interested_topics": ["公共议题", "制度讨论", "媒体评论"],
            }

        elif entity_type_lower in ["mediaoutlet", "socialmediaplatform"]:
            return {
                "bio": f"{entity_name}官方/媒体账号，负责发布消息、转载进展与简短评论。",
                "persona": f"{entity_name}的发言更偏新闻通报、事实更新和引用传播，通常不会深写个人情绪，更倾向于快速发布、跟进、引用和有限回应，在热点升高时会强化转引与追踪。",
                "age": 30,
                "gender": "other",
                "mbti": "ISTJ",
                "country": "中国",
                "profession": "媒体账号",
                "interested_topics": ["新闻更新", "事件进展", "公共议题"],
            }

        elif entity_type_lower in ["university", "governmentagency", "ngo", "organization"]:
            return {
                "bio": f"{entity_name}官方账号，发布正式说明、公告和机构立场。",
                "persona": f"{entity_name}作为机构账号，更偏好正式表达、公告式沟通和有限回应。面对争议时通常强调程序、规则和后续安排，互动节奏保守，少做情绪化对抗。",
                "age": 30,
                "gender": "other",
                "mbti": "ISTJ",
                "country": "中国",
                "profession": entity_type,
                "interested_topics": ["官方公告", "制度回应", "机构治理"],
            }

        else:

            return {
                "bio": entity_summary[:150] if entity_summary else f"{entity_type}: {entity_name}",
                "persona": entity_summary or f"{entity_name}会围绕与自身相关的话题参与讨论，表达方式较简洁，更多依据看到的帖子和有限上下文做出反应。",
                "age": random.randint(25, 50),
                "gender": random.choice(["male", "female"]),
                "mbti": random.choice(self.MBTI_TYPES),
                "country": "中国",
                "profession": entity_type,
                "interested_topics": ["公共讨论", "事件进展", "社会议题"],
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
                    logger.warning(f"实时保存 profiles 失败: {e}")

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
                logger.error(f"生成实体 {entity.name} 的人设失败: {str(e)}")

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

        logger.info(f"开始并行生成 {total} 个Agent人设（并行数: {parallel_count}）...")
        print(f"\n{'='*60}")
        print(f"开始生成Agent人设 - 共 {total} 个实体，并行数: {parallel_count}")
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
                            f"已完成 {current}/{total}: {entity.name}（{entity_type}）"
                        )

                    if error:
                        logger.warning(f"[{current}/{total}] {entity.name} 使用备用人设: {error}")
                    else:
                        logger.info(f"[{current}/{total}] 成功生成人设: {entity.name} ({entity_type})")

                except Exception as e:
                    logger.error(f"处理实体 {entity.name} 时发生异常: {str(e)}")
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
        print(f"人设生成完成！共生成 {len([p for p in profiles if p])} 个Agent")
        print(f"{'='*60}\n")

        return profiles

    def _print_generated_profile(self, entity_name: str, entity_type: str, profile: OasisAgentProfile):
        separator = "-" * 70


        topics_str = ', '.join(profile.interested_topics) if profile.interested_topics else '无'

        output_lines = [
            f"\n{separator}",
            f"[已生成] {entity_name} ({entity_type})",
            f"{separator}",
            f"用户名: {profile.user_name}",
            f"",
            f"【简介】",
            f"{profile.bio}",
            f"",
            f"【详细人设】",
            f"{profile.persona}",
            f"",
            f"【基本属性】",
            f"年龄: {profile.age} | 性别: {profile.gender} | MBTI: {profile.mbti}",
            f"职业: {profile.profession} | 国家: {profile.country}",
            f"兴趣话题: {topics_str}",
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

        logger.info(f"已保存 {len(profiles)} 个Twitter Profile到 {file_path} (OASIS CSV格式)")

    def _normalize_gender(self, gender: Optional[str]) -> str:
        if not gender:
            return "other"

        gender_lower = gender.lower().strip()


        gender_map = {
            "男": "male",
            "女": "female",
            "机构": "other",
            "其他": "other",

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
                "country": profile.country if profile.country else "中国",
            }


            if profile.profession:
                item["profession"] = profile.profession
            if profile.interested_topics:
                item["interested_topics"] = profile.interested_topics

            data.append(item)

        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        logger.info(f"已保存 {len(profiles)} 个Reddit Profile到 {file_path} (JSON格式，包含user_id字段)")


    def save_profiles_to_json(
        self,
        profiles: List[OasisAgentProfile],
        file_path: str,
        platform: str = "reddit"
    ):
        logger.warning("save_profiles_to_json已废弃，请使用save_profiles方法")
        self.save_profiles(profiles, file_path, platform)
