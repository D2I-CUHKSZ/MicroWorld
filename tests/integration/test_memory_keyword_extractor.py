from lightworld.simulation.memory_keywords import MemoryKeywordExtractor


class DummyTopologyRuntime:
    def __init__(self):
        self.agent_id_by_name = {
            "nine thirty podcast": 0,
            "wuhan university": 1,
            "campus witness user": 2,
        }
        self.agent_entity_name = {
            0: "Nine Thirty Podcast",
            1: "Wuhan University",
            2: "Campus Witness User",
        }
        self.agent_semantic_keywords = {
            0: {"timeline", "fact check", "procedural justice"},
            1: {"official bulletin", "student rights", "university governance"},
            2: {"video evidence", "campus safety", "procedural justice"},
        }
        self.profile_by_agent_id = {
            0: {"interested_topics": ["media ethics", "public incidents"]},
            1: {"interested_topics": ["university governance", "student rights"]},
            2: {"interested_topics": ["campus safety", "evidence chain"]},
        }

    def _normalize_agent_name(self, value):
        return str(value or "").strip().lower()


def test_memory_keywords_use_structured_sources():
    extractor = MemoryKeywordExtractor(
        config={
            "event_config": {
                "hot_topics": ["Wuhan University library incident", "procedural justice"],
            }
        },
        topology_runtime=DummyTopologyRuntime(),
    )
    action = {
        "agent_id": 0,
        "agent_name": "Nine Thirty Podcast",
        "action_type": "CREATE_COMMENT",
        "action_args": {
            "content": (
                "Procedural justice cannot rely on bulletins alone; "
                "video evidence from Campus Witness User should also be verified in public."
            ),
            "post_author_name": "Wuhan University",
            "target_user_name": "Campus Witness User",
        },
    }

    keywords = extractor.extract(
        action,
        summary="CREATE_COMMENT | content:procedural justice cannot rely on bulletins alone",
        target_agent_ids=[1, 2],
        max_count=8,
    )

    assert "Nine Thirty Podcast" in keywords
    assert "Wuhan University" in keywords
    assert "Campus Witness User" in keywords
    assert any(
        item in keywords
        for item in ("procedural justice", "video evidence", "university governance", "campus safety")
    )


def test_memory_keywords_filter_noise_tokens():
    extractor = MemoryKeywordExtractor(
        config={"event_config": {"hot_topics": ["procedural justice"]}},
        topology_runtime=DummyTopologyRuntime(),
    )
    action = {
        "agent_id": 0,
        "agent_name": "Nine Thirty Podcast",
        "action_type": "QUOTE_POST",
        "action_args": {
            "content": (
                "See https://example.edu/report/2025/ref-xx — the post_content field must not become a keyword."
            ),
            "quote_content": (
                "Pages at www.hkbu.edu.hk and the 2025 00:17 timestamp should not enter keywords."
            ),
            "original_author_name": "Wuhan University",
        },
    }

    keywords = extractor.extract(
        action,
        summary="QUOTE_POST | post_content:see https://example.edu/report/2025/ref-xx",
        target_agent_ids=[1],
        max_count=10,
    )

    assert "www" not in keywords
    assert "hk" not in keywords
    assert "2025" not in keywords
    assert "post_content" not in keywords
    assert "ref-xx" not in keywords
    assert "Nine Thirty Podcast" in keywords
    assert "Wuhan University" in keywords
