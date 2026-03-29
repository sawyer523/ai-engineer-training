"""Game logger for tracking and replay."""

import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict

from rich.console import Console
from rich.panel import Panel
from rich.text import Text


@dataclass
class LogEntry:
    """A single log entry."""
    timestamp: str
    level: str
    message: str
    metadata: dict = None

    def to_dict(self) -> dict:
        return asdict(self)


class GameLogger:
    """Logger for werewolf game."""

    def __init__(self, log_file: Optional[str] = None):
        self.console = Console()
        self.entries: List[LogEntry] = []
        self.log_file = log_file

        # Color scheme
        self.colors = {
            "info": "blue",
            "debug": "dim",
            "warning": "yellow",
            "error": "red",
            "success": "green",
            "moderator": "cyan",
            "werewolf": "red",
            "villager": "green"
        }

    def log(self, message: str, level: str = "info", metadata: dict = None):
        """Log a message."""
        entry = LogEntry(
            timestamp=datetime.now().isoformat(),
            level=level,
            message=message,
            metadata=metadata or {}
        )
        self.entries.append(entry)

        # Console output
        color = self.colors.get(level, "white")
        text = Text(message, style=color)

        if level == "moderator" or "主持人" in message:
            self.console.print(Panel(text, border_style="cyan", padding=(0, 1)))
        else:
            prefix = f"[{level.upper()}]" if level != "info" else ""
            if prefix:
                self.console.print(f"{prefix} {message}", style=color)
            else:
                self.console.print(message)

    def save(self, path: Optional[str] = None):
        """Save log to file."""
        save_path = path or self.log_file
        if not save_path:
            save_path = f"game_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        Path(save_path).parent.mkdir(parents=True, exist_ok=True)

        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(
                {
                    "game_time": datetime.now().isoformat(),
                    "entries": [e.to_dict() for e in self.entries]
                },
                f,
                ensure_ascii=False,
                indent=2
            )

        self.console.print(f"\n📝 游戏日志已保存至: {save_path}")

    def get_transcript(self) -> str:
        """Get plain text transcript."""
        lines = []
        for entry in self.entries:
            lines.append(f"[{entry.timestamp}] {entry.level}: {entry.message}")
        return "\n".join(lines)

    def print_summary(self):
        """Print log summary."""
        self.console.print("\n" + "="*50)
        self.console.print("📊 游戏日志统计", style="bold cyan")
        self.console.print("="*50)

        level_counts = {}
        for entry in self.entries:
            level_counts[entry.level] = level_counts.get(entry.level, 0) + 1

        for level, count in sorted(level_counts.items()):
            self.console.print(f"  {level.upper()}: {count}")

        self.console.print(f"\n总计事件数: {len(self.entries)}")
        self.console.print("="*50)


class ThoughtTracker:
    """Track agent thoughts, actions, and observations."""

    def __init__(self):
        self.traces: Dict[int, List[Dict]] = {}

    def add_trace(
        self,
        agent_id: int,
        phase: str,
        thought: str,
        action: str = None,
        observation: str = None
    ):
        """Add a trace entry for an agent."""
        if agent_id not in self.traces:
            self.traces[agent_id] = []

        self.traces[agent_id].append({
            "phase": phase,
            "thought": thought,
            "action": action,
            "observation": observation
        })

    def get_agent_trace(self, agent_id: int) -> List[Dict]:
        """Get trace for specific agent."""
        return self.traces.get(agent_id, [])

    def print_trace(self, agent_id: int, console: Console = None):
        """Print trace for an agent."""
        if agent_id not in self.traces:
            return

        console = console or Console()
        traces = self.traces[agent_id]

        console.print(f"\n🔍 玩家{agent_id} 思维追踪:", style="bold")

        for i, trace in enumerate(traces, 1):
            console.print(f"\n  [{i}] 阶段: {trace['phase']}")
            if trace['thought']:
                console.print(f"      💭 思考: {trace['thought'][:100]}...")
            if trace['action']:
                console.print(f"      ⚡ 行动: {trace['action'][:100]}...")
            if trace['observation']:
                console.print(f"      👀 观察: {trace['observation'][:100]}...")

    def save_trace(self, path: str):
        """Save traces to file."""
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.traces, f, ensure_ascii=False, indent=2)
