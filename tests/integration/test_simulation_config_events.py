from microworld.simulation.simulation_config_generator import AgentActivityConfig, SimulationConfigGenerator


def test_parse_event_config_supports_comment_and_thread_events():
    generator = SimulationConfigGenerator()
    event_config = generator._parse_event_config(
        {
            "hot_topics": ["procedural justice"],
            "initial_posts": [{"content": "Official bulletin", "poster_type": "University"}],
            "scheduled_events": [
                {
                    "event_type": "create_comment",
                    "trigger_hour": 3,
                    "content": "This bulletin still does not answer the core question",
                    "poster_type": "Student",
                    "target_poster_type": "University",
                },
                {
                    "event_type": "create_thread",
                    "trigger_hour": 5,
                    "poster_type": "MediaOutlet",
                    "root_content": "Timeline recap",
                    "replies": ["Supplement A", "Supplement B"],
                },
            ],
        }
    )

    assert len(event_config.scheduled_events) == 2
    assert event_config.scheduled_events[0]["event_type"] == "create_comment"
    assert event_config.scheduled_events[0]["target_post_strategy"] == "latest_post_by_type"
    assert event_config.scheduled_events[1]["event_type"] == "create_thread"
    assert event_config.scheduled_events[1]["replies"] == ["Supplement A", "Supplement B"]


def test_assign_initial_post_agents_maps_comment_targets():
    generator = SimulationConfigGenerator()
    agents = [
        AgentActivityConfig(
            agent_id=0,
            entity_uuid="u-1",
            entity_name="Wuhan University",
            entity_type="University",
        ),
        AgentActivityConfig(
            agent_id=1,
            entity_uuid="u-2",
            entity_name="Bystander Student 01",
            entity_type="Student",
        ),
    ]
    event_config = generator._parse_event_config(
        {
            "initial_posts": [{"content": "School notice", "poster_type": "University"}],
            "scheduled_events": [
                {
                    "event_type": "create_comment",
                    "trigger_hour": 2,
                    "content": "I do not accept this explanation",
                    "poster_type": "Student",
                    "target_poster_type": "University",
                }
            ],
        }
    )

    updated = generator._assign_initial_post_agents(event_config, agents)
    comment_event = updated.scheduled_events[0]
    assert comment_event["poster_agent_id"] == 1
    assert comment_event["target_poster_agent_id"] == 0
