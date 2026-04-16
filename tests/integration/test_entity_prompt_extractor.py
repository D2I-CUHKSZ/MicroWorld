from microworld.tools.entity_prompt_extractor import EntityPromptExtractor
from microworld.graph.zep_entity_reader import EntityNode


def make_entity(**overrides) -> EntityNode:
    payload = {
        "uuid": "entity-1",
        "name": "BeijingNewsWireServiceDesk",
        "labels": ["Entity", "MediaOutlet"],
        "summary": (
            "On 2025-08-01 BeijingNewsWireServiceDesk reported that WuhanUniversityOfficialCampusGovernanceBody "
            "decided to open a full disciplinary review of StudentXiaoIncidentReportedSubjectMatter and a thesis audit for Yang."
        ),
        "attributes": {
            "media_type": "NationalDailyNewspaperPrintedEditionFormat",
            "jurisdiction": "NorthernCapitalCityMunicipalRegionBoundary",
        },
        "related_edges": [],
        "related_nodes": [
            {"name": "WuhanUniversityOfficialCampusGovernanceBody"},
            {"name": "StudentXiaoIncidentReportedSubjectMatter"},
        ],
    }
    payload.update(overrides)
    return EntityNode(**payload)


def test_fallback_keywords_filter_schema_noise_and_dates():
    extractor = EntityPromptExtractor(use_llm=False)
    result = extractor.extract_prompt_for_entity(make_entity())

    keywords = result["keywords"]
    assert "BeijingNewsWireServiceDesk" in keywords
    assert any(
        item in keywords
        for item in (
            "NationalDailyNewspaperPrintedEditionFormat",
            "NorthernCapitalCityMunicipalRegionBoundary",
            "WuhanUniversityOfficialCampusGovernanceBody",
        )
    )
    assert "null" not in keywords
    assert "name" not in keywords
    assert "media_type" not in keywords
    assert "08" not in keywords
    assert "2025" not in keywords


def test_normalize_cleans_llm_noise_and_backfills_heuristic_keywords():
    extractor = EntityPromptExtractor(use_llm=False)
    entity = make_entity(
        name="CounselForInvolvedMaleStudentRepresentation",
        labels=["Entity", "LegalProfessional"],
        summary=(
            "The counsel kept speaking publicly on procedural justice, hearings, "
            "and university accountability."
        ),
        attributes={
            "case_role": "DefenseCounselForStudentXiaoLongIdentifier",
            "bar_association": "HubeiProvinceLicensedBarAssociationChapter",
        },
        related_nodes=[
            {"name": "StudentXiaoIncidentReportedSubjectMatter"},
            {"name": "WuhanUniversityOfficialCampusGovernanceBody"},
        ],
    )

    result = extractor._normalize(
        entity,
        {
            "keywords": [
                "null",
                "name",
                "case_role",
                "08",
                "ProceduralJusticeNormativeStandardPhrase",
            ],
            "topic_tags": ["2025", "UniversityAccountabilityOversightTopic", "bar_association"],
            "description": "",
            "semantic_prompt": "",
        },
    )

    keywords = result["keywords"]
    assert keywords[0] == "CounselForInvolvedMaleStudentRepresentation"
    assert "ProceduralJusticeNormativeStandardPhrase" in keywords
    assert any(
        item in keywords
        for item in (
            "UniversityAccountabilityOversightTopic",
            "DefenseCounselForStudentXiaoLongIdentifier",
            "HubeiProvinceLicensedBarAssociationChapter",
        )
    )
    assert "null" not in keywords
    assert "name" not in keywords
    assert "case_role" not in keywords
    assert "08" not in keywords
    assert "2025" not in result["topic_tags"]
