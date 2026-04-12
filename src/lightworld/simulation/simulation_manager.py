
import os
import json
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from lightworld.config.settings import Config
from lightworld.application.simulation_preparation_service import SimulationPreparationFactory
from lightworld.telemetry.logging_config import get_logger
from lightworld.storage.simulation_state_repository import FileSimulationStateRepository

logger = get_logger('lightworld.simulation')


class SimulationStatus(str, Enum):
    CREATED = "created"
    PREPARING = "preparing"
    READY = "ready"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPED = "stopped"
    COMPLETED = "completed"
    FAILED = "failed"


class PlatformType(str, Enum):
    TWITTER = "twitter"
    REDDIT = "reddit"


@dataclass
class SimulationState:
    simulation_id: str
    project_id: str
    graph_id: str


    enable_twitter: bool = True
    enable_reddit: bool = True


    status: SimulationStatus = SimulationStatus.CREATED


    entities_count: int = 0
    profiles_count: int = 0
    entity_types: List[str] = field(default_factory=list)


    config_generated: bool = False
    config_reasoning: str = ""


    current_round: int = 0
    twitter_status: str = "not_started"
    reddit_status: str = "not_started"


    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())


    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "simulation_id": self.simulation_id,
            "project_id": self.project_id,
            "graph_id": self.graph_id,
            "enable_twitter": self.enable_twitter,
            "enable_reddit": self.enable_reddit,
            "status": self.status.value,
            "entities_count": self.entities_count,
            "profiles_count": self.profiles_count,
            "entity_types": self.entity_types,
            "config_generated": self.config_generated,
            "config_reasoning": self.config_reasoning,
            "current_round": self.current_round,
            "twitter_status": self.twitter_status,
            "reddit_status": self.reddit_status,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "error": self.error,
        }

    def to_simple_dict(self) -> Dict[str, Any]:
        return {
            "simulation_id": self.simulation_id,
            "project_id": self.project_id,
            "graph_id": self.graph_id,
            "status": self.status.value,
            "entities_count": self.entities_count,
            "profiles_count": self.profiles_count,
            "entity_types": self.entity_types,
            "config_generated": self.config_generated,
            "error": self.error,
        }


class SimulationManager:
    SIMULATION_DATA_DIR = Config.OASIS_SIMULATION_DATA_DIR

    def __init__(
        self,
        state_repository: Optional[FileSimulationStateRepository] = None,
        preparation_factory: Optional[SimulationPreparationFactory] = None,
    ):
        self._state_repository = state_repository or FileSimulationStateRepository(self.SIMULATION_DATA_DIR)
        self._preparation_factory = preparation_factory or SimulationPreparationFactory()
        self._simulations: Dict[str, SimulationState] = {}

    def _get_simulation_dir(self, simulation_id: str) -> str:
        return self._state_repository.get_simulation_dir(simulation_id)

    def _save_simulation_state(self, state: SimulationState):
        state.updated_at = datetime.now().isoformat()
        self._state_repository.save_state_payload(state.simulation_id, state.to_dict())
        self._simulations[state.simulation_id] = state

    @staticmethod
    def _state_from_payload(simulation_id: str, data: Dict[str, Any]) -> SimulationState:
        return SimulationState(
            simulation_id=simulation_id,
            project_id=data.get("project_id", ""),
            graph_id=data.get("graph_id", ""),
            enable_twitter=data.get("enable_twitter", True),
            enable_reddit=data.get("enable_reddit", True),
            status=SimulationStatus(data.get("status", "created")),
            entities_count=data.get("entities_count", 0),
            profiles_count=data.get("profiles_count", 0),
            entity_types=data.get("entity_types", []),
            config_generated=data.get("config_generated", False),
            config_reasoning=data.get("config_reasoning", ""),
            current_round=data.get("current_round", 0),
            twitter_status=data.get("twitter_status", "not_started"),
            reddit_status=data.get("reddit_status", "not_started"),
            created_at=data.get("created_at", datetime.now().isoformat()),
            updated_at=data.get("updated_at", datetime.now().isoformat()),
            error=data.get("error"),
        )

    def _load_simulation_state(self, simulation_id: str) -> Optional[SimulationState]:
        if simulation_id in self._simulations:
            return self._simulations[simulation_id]

        data = self._state_repository.load_state_payload(simulation_id)
        if data is None:
            return None

        state = self._state_from_payload(simulation_id, data)
        self._simulations[simulation_id] = state
        return state

    def create_simulation(
        self,
        project_id: str,
        graph_id: str,
        enable_twitter: bool = True,
        enable_reddit: bool = True,
    ) -> SimulationState:
        import uuid
        simulation_id = f"sim_{uuid.uuid4().hex[:12]}"

        state = SimulationState(
            simulation_id=simulation_id,
            project_id=project_id,
            graph_id=graph_id,
            enable_twitter=enable_twitter,
            enable_reddit=enable_reddit,
            status=SimulationStatus.CREATED,
        )

        self._save_simulation_state(state)
        logger.info(f"Created simulation: {simulation_id}, project={project_id}, graph={graph_id}")

        return state

    def prepare_simulation(
        self,
        simulation_id: str,
        simulation_requirement: str,
        document_text: str,
        defined_entity_types: Optional[List[str]] = None,
        use_llm_for_profiles: bool = True,
        progress_callback: Optional[callable] = None,
        parallel_profile_count: int = 3
    ) -> SimulationState:
        state = self._load_simulation_state(simulation_id)
        if not state:
            raise ValueError(f"Simulation not found: {simulation_id}")

        try:
            state.status = SimulationStatus.PREPARING
            self._save_simulation_state(state)

            sim_dir = self._get_simulation_dir(simulation_id)


            if progress_callback:
                progress_callback("reading", 0, "Connecting to Zep graph...")

            services = self._preparation_factory.create(graph_id=state.graph_id)
            reader = services.reader

            if progress_callback:
                progress_callback("reading", 30, "Reading node data...")

            filtered = reader.filter_defined_entities(
                graph_id=state.graph_id,
                defined_entity_types=defined_entity_types,
                enrich_with_edges=True
            )

            population_result = services.population_builder.prepare(
                filtered.entities,
                simulation_requirement=simulation_requirement,
            )
            prepared_entities = population_result.entities
            prepared_entity_types = {
                entity.get_entity_type() or "Entity"
                for entity in prepared_entities
                if entity.get_entity_type() or "Entity"
            }
            state.entities_count = len(prepared_entities)
            state.entity_types = sorted(prepared_entity_types)

            if progress_callback:
                progress_callback(
                    "reading", 100,
                    f"Done, {len(prepared_entities)} entities total (including alias merges and ordinary user augmentation)",
                    current=len(prepared_entities),
                    total=len(prepared_entities)
                )

            if len(prepared_entities) == 0:
                state.status = SimulationStatus.FAILED
                state.error = "No matching entities found. Please check if the graph is properly built."
                self._save_simulation_state(state)
                return state

            population_adjustments_file = os.path.join(sim_dir, "population_adjustments.json")
            try:
                with open(population_adjustments_file, "w", encoding="utf-8") as f:
                    json.dump(population_result.to_dict(), f, ensure_ascii=False, indent=2)
            except Exception as e:
                logger.warning(f"Failed to save population adjustments, continuing: {e}")


            entity_graph_edge_count = 0
            entity_graph_file = os.path.join(sim_dir, "entity_graph_snapshot.json")
            try:
                edge_seen = set()
                graph_edges: List[Dict[str, Any]] = []
                uuid_alias_map = population_result.alias_map or {}

                def canonical_uuid(raw_uuid: str) -> str:
                    value = str(raw_uuid or "").strip()
                    return uuid_alias_map.get(value, value)

                for entity in prepared_entities:
                    for edge in (entity.related_edges or []):
                        if not isinstance(edge, dict):
                            continue
                        direction = str(edge.get("direction", "") or "").lower()
                        source_uuid = canonical_uuid(edge.get("source_node_uuid", ""))
                        target_uuid = canonical_uuid(edge.get("target_node_uuid", ""))
                        if direction == "outgoing":
                            source_uuid = source_uuid or canonical_uuid(entity.uuid)
                        elif direction == "incoming":
                            target_uuid = target_uuid or canonical_uuid(entity.uuid)
                        edge_name = str(edge.get("name", "") or "")
                        if not edge_name:
                            edge_name = str(edge.get("edge_name", "") or "")
                        fact = str(edge.get("fact", "") or "")
                        uuid = str(edge.get("uuid", "") or "")
                        if source_uuid == target_uuid and source_uuid:
                            continue
                        key = (
                            uuid
                            or f"{source_uuid}->{target_uuid}|{edge_name}|{fact[:200]}"
                        )
                        if key in edge_seen:
                            continue
                        edge_seen.add(key)
                        graph_edges.append({
                            "uuid": uuid,
                            "name": edge_name,
                            "fact": fact,
                            "source_node_uuid": source_uuid,
                            "target_node_uuid": target_uuid,
                            "attributes": edge.get("attributes", {}) or {},
                        })

                entity_graph_edge_count = len(graph_edges)
                graph_payload = {
                    "simulation_id": simulation_id,
                    "graph_id": state.graph_id,
                    "generated_at": datetime.now().isoformat(),
                    "entity_count": len(prepared_entities),
                    "edge_count": entity_graph_edge_count,
                    "entities": [e.to_dict() for e in prepared_entities],
                    "edges": graph_edges,
                    "population_adjustments_file": os.path.basename(population_adjustments_file),
                }
                with open(entity_graph_file, "w", encoding="utf-8") as f:
                    json.dump(graph_payload, f, ensure_ascii=False, indent=2)
                logger.info(
                    f"Saved entity graph snapshot: {entity_graph_file}, "
                    f"entities={len(prepared_entities)}, edges={entity_graph_edge_count}"
                )
            except Exception as e:
                logger.warning(f"Failed to save entity graph snapshot, continuing: {e}")


            entity_prompts: List[Dict[str, Any]] = []
            entity_prompts_file = os.path.join(sim_dir, "entity_prompts.json")
            try:
                logger.info(f"Starting entity semantic prompt extraction: count={len(prepared_entities)}")
                entity_prompts = services.prompt_extractor.extract_prompts(
                    entities=prepared_entities,
                    simulation_requirement=simulation_requirement
                )
                services.prompt_extractor.save_prompts(entity_prompts, entity_prompts_file)
            except Exception as e:
                logger.warning(f"Failed to extract entity semantic prompts, continuing: {e}")


            total_entities = len(prepared_entities)

            if progress_callback:
                progress_callback(
                    "generating_profiles", 0,
                    "Starting generation...",
                    current=0,
                    total=total_entities
                )


            generator = services.profile_generator

            def profile_progress(current, total, msg):
                if progress_callback:
                    progress_callback(
                        "generating_profiles",
                        int(current / total * 100),
                        msg,
                        current=current,
                        total=total,
                        item_name=msg
                    )


            realtime_output_path = None
            realtime_platform = "reddit"
            if state.enable_reddit:
                realtime_output_path = os.path.join(sim_dir, "reddit_profiles.json")
                realtime_platform = "reddit"
            elif state.enable_twitter:
                realtime_output_path = os.path.join(sim_dir, "twitter_profiles.csv")
                realtime_platform = "twitter"

            profiles = generator.generate_profiles_from_entities(
                entities=prepared_entities,
                use_llm=use_llm_for_profiles,
                progress_callback=profile_progress,
                graph_id=state.graph_id,
                parallel_count=parallel_profile_count,
                realtime_output_path=realtime_output_path,
                output_platform=realtime_platform
            )

            state.profiles_count = len(profiles)


            if progress_callback:
                progress_callback(
                    "generating_profiles", 95,
                    "Saving profile files...",
                    current=total_entities,
                    total=total_entities
                )

            if state.enable_reddit:
                generator.save_profiles(
                    profiles=profiles,
                    file_path=os.path.join(sim_dir, "reddit_profiles.json"),
                    platform="reddit"
                )

            if state.enable_twitter:

                generator.save_profiles(
                    profiles=profiles,
                    file_path=os.path.join(sim_dir, "twitter_profiles.csv"),
                    platform="twitter"
                )

            if progress_callback:
                progress_callback(
                    "generating_profiles", 100,
                    f"Done, {len(profiles)} profiles total",
                    current=len(profiles),
                    total=len(profiles)
                )


            if progress_callback:
                progress_callback(
                    "generating_config", 0,
                    "Analyzing simulation requirements...",
                    current=0,
                    total=3
                )

            if progress_callback:
                progress_callback(
                    "generating_config", 30,
                    "Calling LLM to generate config...",
                    current=1,
                    total=3
                )

            sim_params = services.config_generator.generate_config(
                simulation_id=simulation_id,
                project_id=state.project_id,
                graph_id=state.graph_id,
                simulation_requirement=simulation_requirement,
                document_text=document_text,
                entities=prepared_entities,
                enable_twitter=state.enable_twitter,
                enable_reddit=state.enable_reddit
            )


            social_relation_graph_file = os.path.join(sim_dir, "social_relation_graph.json")
            social_relation_edge_count = 0
            try:
                relation_graph = services.relation_graph_compiler.compile(
                    graph_snapshot_path=entity_graph_file,
                    agent_configs=sim_params.agent_configs,
                    simulation_id=simulation_id,
                    graph_id=state.graph_id,
                )
                social_relation_edge_count = int(relation_graph.get("edge_count", 0))
                services.relation_graph_compiler.save(relation_graph, social_relation_graph_file)
                logger.info(
                    f"Compiled explicit social relation graph: {social_relation_graph_file}, "
                    f"nodes={relation_graph.get('node_count', 0)}, "
                    f"edges={social_relation_edge_count}"
                )
            except Exception as e:
                logger.warning(f"Failed to compile explicit social relation graph, continuing: {e}")

            if progress_callback:
                progress_callback(
                    "generating_config", 70,
                    "Saving config files...",
                    current=2,
                    total=3
                )


            config_path = os.path.join(sim_dir, "simulation_config.json")
            config_data = sim_params.to_dict()
            if entity_prompts:
                config_data["entity_prompts_file"] = "entity_prompts.json"
                config_data["entity_prompts_count"] = len(entity_prompts)
            if os.path.exists(entity_graph_file):
                config_data["entity_graph_file"] = "entity_graph_snapshot.json"
                config_data["entity_graph_entity_count"] = len(prepared_entities)
                config_data["entity_graph_edge_count"] = entity_graph_edge_count
            if os.path.exists(population_adjustments_file):
                config_data["population_adjustments_file"] = "population_adjustments.json"
            if os.path.exists(social_relation_graph_file):
                config_data["social_relation_graph_file"] = "social_relation_graph.json"
                config_data["social_relation_graph_edge_count"] = social_relation_edge_count
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(config_data, f, ensure_ascii=False, indent=2)

            state.config_generated = True
            state.config_reasoning = sim_params.generation_reasoning

            if progress_callback:
                progress_callback(
                    "generating_config", 100,
                    "Config generation complete",
                    current=3,
                    total=3
                )


            state.status = SimulationStatus.READY
            self._save_simulation_state(state)

            logger.info(f"Simulation preparation complete: {simulation_id}, "
                       f"entities={state.entities_count}, profiles={state.profiles_count}")

            return state

        except Exception as e:
            logger.error(f"Simulation preparation failed: {simulation_id}, error={str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            state.status = SimulationStatus.FAILED
            state.error = str(e)
            self._save_simulation_state(state)
            raise

    def get_simulation(self, simulation_id: str) -> Optional[SimulationState]:
        return self._load_simulation_state(simulation_id)

    def list_simulations(self, project_id: Optional[str] = None) -> List[SimulationState]:
        simulations = []
        for sim_id in self._state_repository.list_simulation_ids():
            state = self._load_simulation_state(sim_id)
            if state:
                if project_id is None or state.project_id == project_id:
                    simulations.append(state)

        return simulations

    def get_profiles(self, simulation_id: str, platform: str = "reddit") -> List[Dict[str, Any]]:
        state = self._load_simulation_state(simulation_id)
        if not state:
            raise ValueError(f"Simulation not found: {simulation_id}")

        sim_dir = self._get_simulation_dir(simulation_id)
        profile_path = os.path.join(sim_dir, f"{platform}_profiles.json")

        if not os.path.exists(profile_path):
            return []

        with open(profile_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def get_simulation_config(self, simulation_id: str) -> Optional[Dict[str, Any]]:
        return self._state_repository.load_json_artifact(simulation_id, "simulation_config.json")

    def get_run_instructions(self, simulation_id: str) -> Dict[str, str]:
        sim_dir = self._get_simulation_dir(simulation_id)
        config_path = os.path.join(sim_dir, "simulation_config.json")
        scripts_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../run_scripts'))

        return {
            "simulation_dir": sim_dir,
            "scripts_dir": scripts_dir,
            "config_file": config_path,
            "commands": {
                "twitter": f"uv run python {scripts_dir}/run_twitter_simulation.py --config {config_path}",
                "reddit": f"uv run python {scripts_dir}/run_reddit_simulation.py --config {config_path}",
                "parallel": f"uv run lightworld-parallel-sim --config {config_path}",
            },
            "instructions": (
                f"1. Enter the backend directory and ensure `uv sync` has been run\n"
                f"2. Run simulation (scripts located at {scripts_dir}):\n"
                f"   - Run Twitter only: uv run python {scripts_dir}/run_twitter_simulation.py --config {config_path}\n"
                f"   - Run Reddit only: uv run python {scripts_dir}/run_reddit_simulation.py --config {config_path}\n"
                f"   - Run both platforms in parallel: uv run lightworld-parallel-sim --config {config_path}"
            )
        }
