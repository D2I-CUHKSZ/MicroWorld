from PIL import Image

from microworld.ingestion.multimodal_ingestion import (
    MultimodalIngestionService,
    _VideoSegmentPayload,
)


def test_ingest_text_and_image_files(tmp_path):
    text_path = tmp_path / "brief.txt"
    text_path.write_text(
        "Wuhan University responded to online sentiment; alumni and media kept following up.",
        encoding="utf-8",
    )

    image_path = tmp_path / "poster.png"
    Image.new("RGB", (32, 24), color=(255, 255, 255)).save(image_path)

    service = MultimodalIngestionService()
    service._analyze_image_with_llm = lambda *args, **kwargs: {
        "summary": "Poster shows a university statement and media screenshot.",
        "visible_text": "Wuhan University official statement",
        "key_entities": ["Wuhan University", "media"],
        "social_signals": ["official response", "sentiment spread"],
        "evidence_text": "Image contains statement cues and media circulation hints.",
    }

    result = service.ingest_files(
        [
            {"path": str(text_path), "display_name": "brief.txt"},
            {"path": str(image_path), "display_name": "poster.png"},
        ],
        simulation_requirement="Simulate university brand sentiment spread",
    )

    assert len(result["document_texts"]) == 2
    assert result["manifest"]["file_count"] == 2
    assert result["manifest"]["block_count"] == 2
    assert "document" in result["manifest"]["modalities"]
    assert "image" in result["manifest"]["modalities"]
    assert "Wuhan University responded to online sentiment" in result["all_text"]
    assert "[IMAGE][source=poster.png]" in result["all_text"]
    assert "VisibleText: Wuhan University official statement" in result["all_text"]


def test_ingest_video_file_uses_segment_blocks(tmp_path):
    video_path = tmp_path / "news.mp4"
    video_path.write_bytes(b"fake-video")

    frame_a = tmp_path / "frame_a.jpg"
    frame_b = tmp_path / "frame_b.jpg"
    Image.new("RGB", (16, 16), color=(10, 10, 10)).save(frame_a)
    Image.new("RGB", (16, 16), color=(20, 20, 20)).save(frame_b)

    service = MultimodalIngestionService()
    service._extract_video_segments = lambda *_args, **_kwargs: [
        _VideoSegmentPayload(
            segment_index=0,
            start=0.0,
            end=30.0,
            frame_paths=[str(frame_a), str(frame_b)],
            audio_path=None,
        )
    ]
    service._transcribe_audio = lambda *_args, **_kwargs: (
        "A student posted video; outlets reposted it afterward."
    )
    service._analyze_video_segment_with_llm = lambda *args, **kwargs: {
        "summary": "Clip shows campus statement release and media follow-up.",
        "visible_text": "Official statement",
        "key_entities": ["student", "media", "university"],
        "social_signals": ["statement release", "media republication"],
        "evidence_text": "Video and transcript show the incident entered public spread.",
    }

    result = service.ingest_files(
        [{"path": str(video_path), "display_name": "news.mp4"}],
        simulation_requirement="Simulate campus incident diffusion",
    )

    assert result["manifest"]["file_count"] == 1
    assert result["manifest"]["block_count"] == 1
    assert result["manifest"]["modalities"] == ["video"]
    assert "[VIDEO_SEGMENT][source=news.mp4][00:00:00-00:00:30]" in result["all_text"]
    assert "Transcript: A student posted video; outlets reposted it afterward." in result["all_text"]
    assert "KeyEntities: student, media, university" in result["all_text"]
