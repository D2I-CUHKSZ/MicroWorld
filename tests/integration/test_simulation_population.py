from microworld.simulation.simulation_population import SimulationPopulationBuilder
from microworld.graph.zep_entity_reader import EntityNode


def make_entity(name: str, label: str, summary: str = "", attributes=None, related_nodes=None):
    return EntityNode(
        uuid=f"uuid_{name}",
        name=name,
        labels=["Entity", label],
        summary=summary,
        attributes=attributes or {},
        related_edges=[],
        related_nodes=related_nodes or [],
    )


def test_merge_alias_entities_merges_university_aliases():
    builder = SimulationPopulationBuilder()
    entities = [
        make_entity("Example Media", "University", "Official institution primary listing"),
        make_entity("ExampleMedia", "University", "Alternate compact spelling same org"),
        make_entity("Plain Press", "MediaOutlet", "Media node"),
    ]

    merged, alias_map, groups = builder.merge_alias_entities(entities)

    names = [entity.name for entity in merged]
    assert "Example Media" in names
    assert "ExampleMedia" not in names
    assert alias_map["uuid_ExampleMedia"] == "uuid_Example Media"
    assert any(group["canonical_name"] == "Example Media" for group in groups)


def test_add_ordinary_users_increases_non_elite_population():
    builder = SimulationPopulationBuilder(use_llm_topic_hints=False)
    entities = [
        make_entity(f"press_outlet_{i}", "MediaOutlet", "Press coverage")
        for i in range(10)
    ] + [
        make_entity(f"university_{i}", "University", "Official notice")
        for i in range(4)
    ] + [
        make_entity("involved_student", "Student", "Student perspective")
    ]

    augmented, synthetic = builder.add_ordinary_users(
        entities,
        simulation_requirement="Simulate technical controversy, fact-checking, and impact assessment",
        ordinary_ratio_target=0.5,
        max_synthetic_entities=12,
    )

    assert len(synthetic) > 0
    assert len(augmented) > len(entities)
    assert all(item.attributes.get("synthetic_population") for item in synthetic)
    assert all("校园" not in item.summary for item in synthetic)
    assert all("校友" not in item.name for item in synthetic)
