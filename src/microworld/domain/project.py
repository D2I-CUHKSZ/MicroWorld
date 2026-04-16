
import os
import uuid
from datetime import datetime
from typing import Dict, Any, List, Optional
from enum import Enum
from dataclasses import dataclass, field
from microworld.config.settings import Config
from microworld.storage.project_repository import FileProjectRepository


class ProjectStatus(str, Enum):
    CREATED = "created"
    ONTOLOGY_GENERATED = "ontology_generated"
    GRAPH_BUILDING = "graph_building"
    GRAPH_COMPLETED = "graph_completed"
    FAILED = "failed"


@dataclass
class Project:
    project_id: str
    name: str
    status: ProjectStatus
    created_at: str
    updated_at: str


    files: List[Dict[str, str]] = field(default_factory=list)
    total_text_length: int = 0


    ontology: Optional[Dict[str, Any]] = None
    analysis_summary: Optional[str] = None
    ingestion_summary: Optional[Dict[str, Any]] = None


    graph_id: Optional[str] = None
    graph_build_task_id: Optional[str] = None


    simulation_requirement: Optional[str] = None
    chunk_size: int = 500
    chunk_overlap: int = 50


    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "project_id": self.project_id,
            "name": self.name,
            "status": self.status.value if isinstance(self.status, ProjectStatus) else self.status,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "files": self.files,
            "total_text_length": self.total_text_length,
            "ontology": self.ontology,
            "analysis_summary": self.analysis_summary,
            "ingestion_summary": self.ingestion_summary,
            "graph_id": self.graph_id,
            "graph_build_task_id": self.graph_build_task_id,
            "simulation_requirement": self.simulation_requirement,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "error": self.error
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Project':
        status = data.get('status', 'created')
        if isinstance(status, str):
            status = ProjectStatus(status)

        return cls(
            project_id=data['project_id'],
            name=data.get('name', 'Unnamed Project'),
            status=status,
            created_at=data.get('created_at', ''),
            updated_at=data.get('updated_at', ''),
            files=data.get('files', []),
            total_text_length=data.get('total_text_length', 0),
            ontology=data.get('ontology'),
            analysis_summary=data.get('analysis_summary'),
            ingestion_summary=data.get('ingestion_summary'),
            graph_id=data.get('graph_id'),
            graph_build_task_id=data.get('graph_build_task_id'),
            simulation_requirement=data.get('simulation_requirement'),
            chunk_size=data.get('chunk_size', 500),
            chunk_overlap=data.get('chunk_overlap', 50),
            error=data.get('error')
        )


class ProjectManager:
    _repository = FileProjectRepository(os.path.join(Config.UPLOAD_FOLDER, 'projects'))
    PROJECTS_DIR = _repository.projects_dir

    @classmethod
    def _ensure_projects_dir(cls):
        cls._repository.ensure_projects_dir()

    @classmethod
    def _get_project_dir(cls, project_id: str) -> str:
        return cls._repository.get_project_dir(project_id)

    @classmethod
    def _get_project_meta_path(cls, project_id: str) -> str:
        return cls._repository.get_project_meta_path(project_id)

    @classmethod
    def _get_project_files_dir(cls, project_id: str) -> str:
        return cls._repository.get_project_files_dir(project_id)

    @classmethod
    def _get_project_text_path(cls, project_id: str) -> str:
        return cls._repository.get_project_text_path(project_id)

    @classmethod
    def _get_project_artifact_path(cls, project_id: str, filename: str) -> str:
        return cls._repository.get_project_artifact_path(project_id, filename)

    @classmethod
    def create_project(cls, name: str = "Unnamed Project") -> Project:
        cls._ensure_projects_dir()

        project_id = f"proj_{uuid.uuid4().hex[:12]}"
        now = datetime.now().isoformat()

        project = Project(
            project_id=project_id,
            name=name,
            status=ProjectStatus.CREATED,
            created_at=now,
            updated_at=now
        )
        cls._repository.create_project_storage(project_id)

        cls.save_project(project)

        return project

    @classmethod
    def save_project(cls, project: Project) -> None:
        project.updated_at = datetime.now().isoformat()
        cls._repository.save_project_payload(project.project_id, project.to_dict())

    @classmethod
    def get_project(cls, project_id: str) -> Optional[Project]:
        data = cls._repository.load_project_payload(project_id)
        if data is None:
            return None
        return Project.from_dict(data)

    @classmethod
    def list_projects(cls, limit: int = 50) -> List[Project]:
        projects = []
        for project_id in cls._repository.list_project_ids():
            project = cls.get_project(project_id)
            if project:
                projects.append(project)


        projects.sort(key=lambda p: p.created_at, reverse=True)

        return projects[:limit]

    @classmethod
    def delete_project(cls, project_id: str) -> bool:
        return cls._repository.delete_project(project_id)

    @classmethod
    def save_file_to_project(cls, project_id: str, file_storage, original_filename: str) -> Dict[str, str]:
        return cls._repository.save_uploaded_file(project_id, file_storage, original_filename)

    @classmethod
    def save_local_file_to_project(cls, project_id: str, source_path: str) -> Dict[str, str]:
        return cls._repository.save_local_file(project_id, source_path)

    @classmethod
    def save_extracted_text(cls, project_id: str, text: str) -> None:
        cls._repository.save_extracted_text(project_id, text)

    @classmethod
    def save_json_artifact(cls, project_id: str, filename: str, payload: Dict[str, Any]) -> None:
        cls._repository.save_json_artifact(project_id, filename, payload)

    @classmethod
    def get_json_artifact(cls, project_id: str, filename: str) -> Optional[Dict[str, Any]]:
        return cls._repository.get_json_artifact(project_id, filename)

    @classmethod
    def get_extracted_text(cls, project_id: str) -> Optional[str]:
        return cls._repository.get_extracted_text(project_id)

    @classmethod
    def get_project_files(cls, project_id: str) -> List[str]:
        return cls._repository.get_project_files(project_id)
