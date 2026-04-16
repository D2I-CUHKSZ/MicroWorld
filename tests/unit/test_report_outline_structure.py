# -*- coding: utf-8 -*-
from microworld.reporting.report_agent import ReportAgent, ReportOutline, ReportSection


def test_normalize_outline_adds_history_and_forecast_sections():
    default = ReportAgent._default_outline()
    outline = ReportOutline(
        title="Wuhan University sentiment analysis",
        summary="Test summary",
        sections=[
            ReportSection(title="Event ignition and cross-platform divergence"),
            ReportSection(title="Actor action map: from silence to coordinated pressure"),
            ReportSection(title="Asymmetric influence reveals structural risk"),
        ],
    )

    normalized = ReportAgent._normalize_outline(outline)

    assert len(normalized.sections) >= 3
    assert normalized.sections[0].title == default.sections[0].title
    assert any(
        ReportAgent._contains_any_keyword(section.title, ReportAgent.EVOLUTION_SECTION_KEYWORDS)
        for section in normalized.sections
    )
    assert normalized.sections[-1].title == default.sections[2].title


def test_normalize_outline_moves_existing_forecast_to_last_section():
    hist_kw = ReportAgent.HISTORY_SECTION_KEYWORDS[0]
    evo_kw = ReportAgent.EVOLUTION_SECTION_KEYWORDS[0]
    fc_kw = ReportAgent.FORECAST_SECTION_KEYWORDS[0]
    outline = ReportOutline(
        title="Campus incident report",
        summary="Test summary",
        sections=[
            ReportSection(title=f"Outlook section {fc_kw}", description="Pre-existing forecast section"),
            ReportSection(title=f"Review section {hist_kw}"),
            ReportSection(title=f"Spread section {evo_kw}"),
        ],
    )

    normalized = ReportAgent._normalize_outline(outline)

    assert normalized.sections[-1].title == f"Outlook section {fc_kw}"
    assert normalized.sections[-1].description == "Pre-existing forecast section"
    assert ReportAgent._contains_any_keyword(
        normalized.sections[0].title, ReportAgent.HISTORY_SECTION_KEYWORDS
    )
