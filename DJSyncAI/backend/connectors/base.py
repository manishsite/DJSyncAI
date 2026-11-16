from abc import ABC, abstractmethod
from typing import List, Dict, Any
from ..models import PlaylistDNA

class BaseConnector(ABC):
    """Abstract interface for all music discovery connectors."""

    @property
    @abstractmethod
    def source_name(self) -> str:
        """Name of the discovery source."""
        pass

    @abstractmethod
    async def search_candidates(self, dna: PlaylistDNA, limit: int = 50) -> List[Dict[str, Any]]:
        """Query the source for potential electronic track candidates matching the DNA."""
        pass
