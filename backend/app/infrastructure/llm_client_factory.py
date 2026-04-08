from threading import Lock
from typing import Dict, Tuple

from ..config import Config
from .llm_client import LLMClient


class LLMClientFactory:
    _shared_clients: Dict[Tuple[str, str, str], LLMClient] = {}
    _lock = Lock()

    @classmethod
    def create(
        cls,
        api_key: str = None,
        base_url: str = None,
        model: str = None,
    ) -> LLMClient:
        return LLMClient(api_key=api_key, base_url=base_url, model=model)

    @classmethod
    def get_shared_client(
        cls,
        api_key: str = None,
        base_url: str = None,
        model: str = None,
    ) -> LLMClient:
        resolved_api_key = api_key or Config.LLM_API_KEY or ""
        resolved_base_url = base_url or Config.LLM_BASE_URL or ""
        resolved_model = model or Config.LLM_MODEL_NAME or ""
        cache_key = (resolved_api_key, resolved_base_url, resolved_model)

        with cls._lock:
            client = cls._shared_clients.get(cache_key)
            if client is None:
                client = cls.create(api_key=api_key, base_url=base_url, model=model)
                cls._shared_clients[cache_key] = client
            return client
