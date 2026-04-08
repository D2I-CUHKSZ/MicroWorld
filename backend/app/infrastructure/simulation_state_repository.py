import json
import os
from typing import Any, Dict, List, Optional


class FileSimulationStateRepository:
    def __init__(self, root_dir: str):
        self.root_dir = root_dir
        os.makedirs(self.root_dir, exist_ok=True)

    def get_simulation_dir(self, simulation_id: str, create: bool = True) -> str:
        sim_dir = os.path.join(self.root_dir, simulation_id)
        if create:
            os.makedirs(sim_dir, exist_ok=True)
        return sim_dir

    def get_state_path(self, simulation_id: str) -> str:
        return os.path.join(self.get_simulation_dir(simulation_id), "state.json")

    def save_state_payload(self, simulation_id: str, payload: Dict[str, Any]):
        state_path = self.get_state_path(simulation_id)
        with open(state_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    def load_state_payload(self, simulation_id: str) -> Optional[Dict[str, Any]]:
        state_path = self.get_state_path(simulation_id)
        if not os.path.exists(state_path):
            return None
        with open(state_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None

    def list_simulation_ids(self) -> List[str]:
        simulation_ids: List[str] = []
        if not os.path.exists(self.root_dir):
            return simulation_ids
        for simulation_id in os.listdir(self.root_dir):
            sim_dir = self.get_simulation_dir(simulation_id, create=False)
            if simulation_id.startswith(".") or not os.path.isdir(sim_dir):
                continue
            simulation_ids.append(simulation_id)
        return simulation_ids

    def get_artifact_path(self, simulation_id: str, filename: str) -> str:
        return os.path.join(self.get_simulation_dir(simulation_id), filename)

    def load_json_artifact(self, simulation_id: str, filename: str) -> Optional[Dict[str, Any]]:
        path = self.get_artifact_path(simulation_id, filename)
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
