import json
import os
import shutil
import uuid
from typing import Any, Dict, List, Optional


class FileProjectRepository:
    def __init__(self, projects_dir: str):
        self.projects_dir = projects_dir

    def ensure_projects_dir(self):
        os.makedirs(self.projects_dir, exist_ok=True)

    def get_project_dir(self, project_id: str) -> str:
        return os.path.join(self.projects_dir, project_id)

    def get_project_meta_path(self, project_id: str) -> str:
        return os.path.join(self.get_project_dir(project_id), "project.json")

    def get_project_files_dir(self, project_id: str) -> str:
        return os.path.join(self.get_project_dir(project_id), "files")

    def get_project_text_path(self, project_id: str) -> str:
        return os.path.join(self.get_project_dir(project_id), "extracted_text.txt")

    def get_project_artifact_path(self, project_id: str, filename: str) -> str:
        return os.path.join(self.get_project_dir(project_id), filename)

    def create_project_storage(self, project_id: str):
        self.ensure_projects_dir()
        os.makedirs(self.get_project_dir(project_id), exist_ok=True)
        os.makedirs(self.get_project_files_dir(project_id), exist_ok=True)

    def save_project_payload(self, project_id: str, payload: Dict[str, Any]):
        self.create_project_storage(project_id)
        meta_path = self.get_project_meta_path(project_id)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    def load_project_payload(self, project_id: str) -> Optional[Dict[str, Any]]:
        meta_path = self.get_project_meta_path(project_id)
        if not os.path.exists(meta_path):
            return None
        with open(meta_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None

    def list_project_ids(self) -> List[str]:
        self.ensure_projects_dir()
        project_ids: List[str] = []
        for project_id in os.listdir(self.projects_dir):
            project_dir = self.get_project_dir(project_id)
            if project_id.startswith(".") or not os.path.isdir(project_dir):
                continue
            project_ids.append(project_id)
        return project_ids

    def delete_project(self, project_id: str) -> bool:
        project_dir = self.get_project_dir(project_id)
        if not os.path.exists(project_dir):
            return False
        shutil.rmtree(project_dir)
        return True

    def _allocate_file_path(self, project_id: str, original_filename: str) -> str:
        files_dir = self.get_project_files_dir(project_id)
        os.makedirs(files_dir, exist_ok=True)
        ext = os.path.splitext(original_filename)[1].lower()
        safe_filename = f"{uuid.uuid4().hex[:8]}{ext}"
        return os.path.join(files_dir, safe_filename)

    def save_uploaded_file(self, project_id: str, file_storage, original_filename: str) -> Dict[str, str]:
        file_path = self._allocate_file_path(project_id, original_filename)
        file_storage.save(file_path)
        return {
            "original_filename": original_filename,
            "saved_filename": os.path.basename(file_path),
            "path": file_path,
            "size": os.path.getsize(file_path),
        }

    def save_local_file(self, project_id: str, source_path: str) -> Dict[str, str]:
        if not os.path.exists(source_path):
            raise FileNotFoundError(f"文件不存在: {source_path}")
        if not os.path.isfile(source_path):
            raise ValueError(f"不是文件: {source_path}")

        original_filename = os.path.basename(source_path)
        file_path = self._allocate_file_path(project_id, original_filename)
        shutil.copy2(source_path, file_path)
        return {
            "original_filename": original_filename,
            "saved_filename": os.path.basename(file_path),
            "path": file_path,
            "size": os.path.getsize(file_path),
        }

    def save_extracted_text(self, project_id: str, text: str):
        text_path = self.get_project_text_path(project_id)
        os.makedirs(os.path.dirname(text_path), exist_ok=True)
        with open(text_path, "w", encoding="utf-8") as f:
            f.write(text)

    def get_extracted_text(self, project_id: str) -> Optional[str]:
        text_path = self.get_project_text_path(project_id)
        if not os.path.exists(text_path):
            return None
        with open(text_path, "r", encoding="utf-8") as f:
            return f.read()

    def save_json_artifact(self, project_id: str, filename: str, payload: Dict[str, Any]):
        artifact_path = self.get_project_artifact_path(project_id, filename)
        os.makedirs(os.path.dirname(artifact_path), exist_ok=True)
        with open(artifact_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    def get_json_artifact(self, project_id: str, filename: str) -> Optional[Dict[str, Any]]:
        artifact_path = self.get_project_artifact_path(project_id, filename)
        if not os.path.exists(artifact_path):
            return None
        with open(artifact_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None

    def get_project_files(self, project_id: str) -> List[str]:
        files_dir = self.get_project_files_dir(project_id)
        if not os.path.exists(files_dir):
            return []
        return [
            os.path.join(files_dir, filename)
            for filename in os.listdir(files_dir)
            if os.path.isfile(os.path.join(files_dir, filename))
        ]
