from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import Protocol, TextIO


class Logger(Protocol):
    def log(self, message: str, verbosity_level: int = 0) -> None: ...


@dataclass
class ConsoleLogger:
    verbosity: int = 0
    stream: TextIO = field(default_factory=lambda: sys.stdout)
    flush: bool = True

    def log(self, message: str, verbosity_level: int = 0) -> None:
        if verbosity_level > self.verbosity:
            return
        print(message, file=self.stream, flush=self.flush)


class NullLogger:
    def log(self, message: str, verbosity_level: int = 0) -> None:
        return

