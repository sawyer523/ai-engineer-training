"""Game state management for Werewolf game."""

from enum import Enum
from typing import List, Dict, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime


class GamePhase(Enum):
    """Game phases."""
    SETUP = "setup"
    NIGHT = "night"
    NIGHT_DISCUSSION = "night_discussion"  # Werewolves discuss who to kill
    NIGHT_ACTION = "night_action"  # Werewolves vote for victim
    DAY_ANNOUNCE = "day_announce"  # Announce deaths
    DAY_DISCUSSION = "day_discussion"  # Players discuss
    DAY_VOTE = "day_vote"  # Players vote to eliminate
    GAME_OVER = "game_over"


class Role(Enum):
    """Player roles."""
    VILLAGER = "villager"
    WEREWOLF = "werewolf"


class PlayerStatus(Enum):
    """Player status."""
    ALIVE = "alive"
    DEAD = "dead"


@dataclass
class Player:
    """Represents a player in the game."""
    id: int
    name: str
    role: Role
    status: PlayerStatus = PlayerStatus.ALIVE
    personality: str = "neutral"  # aggressive, cautious, neutral, observant
    memory: List[str] = field(default_factory=list)

    def is_alive(self) -> bool:
        return self.status == PlayerStatus.ALIVE

    def kill(self):
        self.status = PlayerStatus.DEAD

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "role": self.role.value,
            "status": self.status.value,
            "personality": self.personality
        }


@dataclass
class GameEvent:
    """Represents a game event for logging and memory."""
    timestamp: datetime
    phase: GamePhase
    event_type: str
    description: str
    involved_players: List[int] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


@dataclass
class Vote:
    """Represents a vote."""
    voter_id: int
    target_id: int
    round: int
    phase: str  # "night" or "day"


@dataclass
class GameState:
    """Central game state manager."""
    players: List[Player] = field(default_factory=list)
    phase: GamePhase = GamePhase.SETUP
    round: int = 0
    events: List[GameEvent] = field(default_factory=list)
    votes: List[Vote] = field(default_factory=list)
    winner: Optional[Role] = None
    current_night_target: Optional[int] = None  # Target of werewolves
    last_deaths: List[int] = field(default_factory=list)  # Players who died last night

    # Discussion tracking
    current_discussion: List[dict] = field(default_factory=list)  # Current round discussion

    def add_player(self, player: Player):
        self.players.append(player)

    def get_player(self, player_id: int) -> Optional[Player]:
        for p in self.players:
            if p.id == player_id:
                return p
        return None

    def get_alive_players(self) -> List[Player]:
        return [p for p in self.players if p.is_alive()]

    def get_alive_werewolves(self) -> List[Player]:
        return [p for p in self.players if p.is_alive() and p.role == Role.WEREWOLF]

    def get_alive_villagers(self) -> List[Player]:
        return [p for p in self.players if p.is_alive() and p.role == Role.VILLAGER]

    def add_event(self, event_type: str, description: str, involved_players: List[int] = None, metadata: dict = None):
        event = GameEvent(
            timestamp=datetime.now(),
            phase=self.phase,
            event_type=event_type,
            description=description,
            involved_players=involved_players or [],
            metadata=metadata or {}
        )
        self.events.append(event)

    def add_vote(self, voter_id: int, target_id: int):
        vote = Vote(
            voter_id=voter_id,
            target_id=target_id,
            round=self.round,
            phase="night" if self.phase == GamePhase.NIGHT_ACTION else "day"
        )
        self.votes.append(vote)

    def get_votes_for_round(self, round_num: int, phase: str = None) -> List[Vote]:
        votes = [v for v in self.votes if v.round == round_num]
        if phase:
            votes = [v for v in votes if v.phase == phase]
        return votes

    def check_win_condition(self) -> Optional[Role]:
        """Check if win condition is met."""
        alive_werewolves = len(self.get_alive_werewolves())
        alive_villagers = len(self.get_alive_villagers())

        if alive_werewolves == 0:
            self.winner = Role.VILLAGER
            return Role.VILLAGER

        if alive_werewolves >= alive_villagers:
            self.winner = Role.WEREWOLF
            return Role.WEREWOLF

        return None

    def transition_to(self, new_phase: GamePhase):
        """Transition to a new phase."""
        old_phase = self.phase
        self.phase = new_phase
        self.add_event(
            "phase_transition",
            f"Phase changed from {old_phase.value} to {new_phase.value}"
        )

        # Increment round when starting new night
        if new_phase == GamePhase.NIGHT and old_phase != GamePhase.NIGHT_DISCUSSION:
            self.round += 1
            self.current_discussion = []

    def get_public_info(self) -> dict:
        """Get information visible to all players."""
        return {
            "phase": self.phase.value,
            "round": self.round,
            "alive_players": [
                {"id": p.id, "name": p.name}
                for p in self.get_alive_players()
            ],
            "dead_players": [
                {"id": p.id, "name": p.name}
                for p in self.players if not p.is_alive()
            ],
            "recent_deaths": self.last_deaths
        }

    def get_werewolf_info(self) -> dict:
        """Get information only visible to werewolves."""
        return {
            "fellow_werewolves": [
                {"id": p.id, "name": p.name}
                for p in self.get_alive_werewolves()
            ]
        }
