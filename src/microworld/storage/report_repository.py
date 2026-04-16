import json
import os
import shutil
from typing import Any, Dict, List, Optional


class FileReportRepository:
    def __init__(self, reports_dir: str):
        self.reports_dir = reports_dir

    def ensure_reports_dir(self):
        os.makedirs(self.reports_dir, exist_ok=True)

    def get_report_folder(self, report_id: str, create: bool = False) -> str:
        folder = os.path.join(self.reports_dir, report_id)
        if create:
            os.makedirs(folder, exist_ok=True)
        return folder

    def get_report_path(self, report_id: str) -> str:
        return os.path.join(self.get_report_folder(report_id), "meta.json")

    def get_report_markdown_path(self, report_id: str) -> str:
        return os.path.join(self.get_report_folder(report_id), "full_report.md")

    def get_outline_path(self, report_id: str) -> str:
        return os.path.join(self.get_report_folder(report_id), "outline.json")

    def get_progress_path(self, report_id: str) -> str:
        return os.path.join(self.get_report_folder(report_id), "progress.json")

    def get_section_path(self, report_id: str, section_index: int) -> str:
        return os.path.join(self.get_report_folder(report_id), f"section_{section_index:02d}.md")

    def get_agent_log_path(self, report_id: str) -> str:
        return os.path.join(self.get_report_folder(report_id), "agent_log.jsonl")

    def get_console_log_path(self, report_id: str) -> str:
        return os.path.join(self.get_report_folder(report_id), "console_log.txt")

    def get_legacy_report_json_path(self, report_id: str) -> str:
        return os.path.join(self.reports_dir, f"{report_id}.json")

    def get_legacy_report_markdown_path(self, report_id: str) -> str:
        return os.path.join(self.reports_dir, f"{report_id}.md")

    def save_json(self, path: str, payload: Dict[str, Any]):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    def load_json(self, path: str) -> Optional[Dict[str, Any]]:
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None

    def save_text(self, path: str, content: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)

    def load_text(self, path: str) -> Optional[str]:
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    def read_text_lines(self, path: str, from_line: int = 0) -> Dict[str, Any]:
        if not os.path.exists(path):
            return {
                "logs": [],
                "total_lines": 0,
                "from_line": 0,
                "has_more": False,
            }

        logs: List[str] = []
        total_lines = 0
        with open(path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                total_lines = idx + 1
                if idx >= from_line:
                    logs.append(line.rstrip("\n\r"))

        return {
            "logs": logs,
            "total_lines": total_lines,
            "from_line": from_line,
            "has_more": False,
        }

    def read_jsonl_lines(self, path: str, from_line: int = 0) -> Dict[str, Any]:
        if not os.path.exists(path):
            return {
                "logs": [],
                "total_lines": 0,
                "from_line": 0,
                "has_more": False,
            }

        logs: List[Dict[str, Any]] = []
        total_lines = 0
        with open(path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                total_lines = idx + 1
                if idx < from_line:
                    continue
                try:
                    logs.append(json.loads(line.strip()))
                except json.JSONDecodeError:
                    continue

        return {
            "logs": logs,
            "total_lines": total_lines,
            "from_line": from_line,
            "has_more": False,
        }

    def load_generated_sections(self, report_id: str) -> List[Dict[str, Any]]:
        folder = self.get_report_folder(report_id)
        if not os.path.exists(folder):
            return []

        sections: List[Dict[str, Any]] = []
        for filename in sorted(os.listdir(folder)):
            if not (filename.startswith("section_") and filename.endswith(".md")):
                continue
            path = os.path.join(folder, filename)
            if not os.path.isfile(path):
                continue
            content = self.load_text(path) or ""
            try:
                section_index = int(filename.replace(".md", "").split("_")[1])
            except (IndexError, ValueError):
                continue
            sections.append(
                {
                    "filename": filename,
                    "section_index": section_index,
                    "content": content,
                }
            )
        return sections

    def save_report_payload(self, report_id: str, payload: Dict[str, Any]):
        self.get_report_folder(report_id, create=True)
        self.save_json(self.get_report_path(report_id), payload)

    def load_report_payload(self, report_id: str) -> Optional[Dict[str, Any]]:
        payload = self.load_json(self.get_report_path(report_id))
        if payload is not None:
            return payload
        return self.load_json(self.get_legacy_report_json_path(report_id))

    def save_outline_payload(self, report_id: str, payload: Dict[str, Any]):
        self.get_report_folder(report_id, create=True)
        self.save_json(self.get_outline_path(report_id), payload)

    def load_progress_payload(self, report_id: str) -> Optional[Dict[str, Any]]:
        return self.load_json(self.get_progress_path(report_id))

    def save_progress_payload(self, report_id: str, payload: Dict[str, Any]):
        self.get_report_folder(report_id, create=True)
        self.save_json(self.get_progress_path(report_id), payload)

    def save_section_markdown(self, report_id: str, section_index: int, content: str) -> str:
        self.get_report_folder(report_id, create=True)
        path = self.get_section_path(report_id, section_index)
        self.save_text(path, content)
        return path

    def save_full_report_markdown(self, report_id: str, content: str):
        self.get_report_folder(report_id, create=True)
        self.save_text(self.get_report_markdown_path(report_id), content)

    def load_report_markdown(self, report_id: str) -> Optional[str]:
        content = self.load_text(self.get_report_markdown_path(report_id))
        if content:
            return content
        return self.load_text(self.get_legacy_report_markdown_path(report_id))

    def list_report_ids(self) -> List[str]:
        self.ensure_reports_dir()
        report_ids = set()
        for item in os.listdir(self.reports_dir):
            item_path = os.path.join(self.reports_dir, item)
            if item.startswith("."):
                continue
            if os.path.isdir(item_path):
                report_ids.add(item)
            elif item.endswith(".json"):
                report_ids.add(item[:-5])
        return sorted(report_ids)

    def delete_report(self, report_id: str) -> bool:
        folder_path = self.get_report_folder(report_id)
        deleted = False

        if os.path.isdir(folder_path):
            shutil.rmtree(folder_path)
            deleted = True

        for path in (
            self.get_legacy_report_json_path(report_id),
            self.get_legacy_report_markdown_path(report_id),
        ):
            if os.path.exists(path):
                os.remove(path)
                deleted = True

        return deleted
