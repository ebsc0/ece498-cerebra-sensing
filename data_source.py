"""Source interfaces for packet producers (simulator, UART, etc.)."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Optional


PacketCallback = Callable[[bytes], None]


@dataclass(frozen=True)
class SourceEvent:
    """Structured lifecycle/health event from a data source."""

    kind: str
    message: str


SourceEventCallback = Callable[[SourceEvent], None]


class DataSource(ABC):
    """Abstract packet source used by the app orchestrator."""

    @abstractmethod
    def start(
        self,
        packet_callback: PacketCallback,
        event_callback: Optional[SourceEventCallback] = None,
    ) -> None:
        """Begin emitting packets to callback."""

    @abstractmethod
    def stop(self) -> None:
        """Stop source and release underlying resources."""

    @abstractmethod
    def is_running(self) -> bool:
        """Whether source is currently active."""

