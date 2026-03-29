"""Moderator agent for game flow control."""

import asyncio
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
from datetime import datetime

from ..game.state import GameState, GamePhase, Role, Player, PlayerStatus
from ..agents.player import PlayerAgent
from ..utils.logger import GameLogger


@dataclass
class GameResult:
    """Result of a completed game."""
    winner: Role
    total_rounds: int
    total_events: int
    player_stats: List[dict]
    duration_seconds: float


class ModeratorAgent:
    """Moderator agent that controls game flow."""

    def __init__(
        self,
        players: List[PlayerAgent],
        logger: Optional[GameLogger] = None,
        verbose: bool = True
    ):
        self.players = players
        self.game_state = GameState()
        self.logger = logger or GameLogger()
        self.verbose = verbose

        # Initialize game state with players
        for agent in self.players:
            self.game_state.add_player(agent.player)

        # Track game statistics
        self.start_time = None
        self.end_time = None

    def _log(self, message: str, level: str = "info"):
        """Log a message."""
        if self.verbose:
            self.logger.log(message, level)

    def _announce(self, message: str):
        """Make a moderator announcement."""
        self._log(f"🎯 主持人: {message}")
        self.game_state.add_event("announcement", message)

    async def start_game(self):
        """Start the game."""
        self.start_time = datetime.now()
        self._announce("游戏开始！欢迎来到狼人杀。")
        self._announce(f"共有 {len(self.players)} 名玩家参与游戏。")

        # Reveal roles privately
        for agent in self.players:
            role = "狼人" if agent.player.role == Role.WEREWOLF else "村民"
            self._log(f"   玩家{agent.player.id} ({agent.player.name}) 的身份是: {role}", "debug")

            # Let agents know their teammates if werewolf
            if agent.player.role == Role.WEREWOLF:
                werewolves = self.game_state.get_alive_werewolves()
                teammates = [w for w in werewolves if w.id != agent.player.id]
                if teammates:
                    teammate_names = ", ".join([t.name for t in teammates])
                    agent.observe_event(
                        "game_start",
                        f"你的狼人队友是: {teammate_names}",
                        teammates=[t.id for t in teammates]
                    )

        self.game_state.transition_to(GamePhase.NIGHT)

    async def run_night_phase(self):
        """Run the night phase."""
        self._announce(f"\n🌙 第 {self.game_state.round} 夜 - 天黑请闭眼")

        self.game_state.transition_to(GamePhase.NIGHT_DISCUSSION)

        # Werewolf discussion
        werewolves = [p for p in self.players if p.player.role == Role.WEREWOLF and p.player.is_alive()]

        if werewolves:
            self._log("   🐺 狼人正在讨论...", "debug")

            discussion = []
            for wolf in werewolves:
                speech = await wolf.night_discussion(self.game_state, discussion)
                if speech:
                    discussion.append({
                        "speaker": wolf.player.id,
                        "content": speech
                    })
                    self._log(f"      狼人{wolf.player.id}: {speech}", "debug")

            self.game_state.transition_to(GamePhase.NIGHT_ACTION)

            # Werewolf voting for target
            votes = {}
            for wolf in werewolves:
                target = await wolf.night_vote(self.game_state, discussion)
                if target is not None:
                    votes[target] = votes.get(target, 0) + 1
                    self.game_state.add_vote(wolf.player.id, target)

            # Determine victim (most voted)
            if votes:
                victim_id = max(votes, key=votes.get)
                self.game_state.current_night_target = victim_id

                # Record in werewolves' memory
                for wolf in werewolves:
                    wolf.observe_event(
                        "night_action",
                        f"今晚决定杀害玩家{victim_id}",
                        target=victim_id
                    )

    async def run_day_announce(self):
        """Run day announcement phase."""
        self._announce(f"\n☀️ 天亮了 - 第 {self.game_state.round} 天")
        self.game_state.transition_to(GamePhase.DAY_ANNOUNCE)

        # Process night death
        if self.game_state.current_night_target is not None:
            victim = self.game_state.get_player(self.game_state.current_night_target)
            if victim and victim.is_alive():
                victim.kill()
                self.game_state.last_deaths = [victim.id]

                self._announce(f"昨晚，玩家{victim.id} ({victim.name}) 惨遭杀害！")

                # Record in all players' memory
                for agent in self.players:
                    agent.observe_event(
                        "death",
                        f"玩家{victim.id}昨晚死亡",
                        victim=victim.id
                    )
            else:
                self._announce("昨晚是平安夜，没有人死亡。")
                self.game_state.last_deaths = []
        else:
            self._announce("昨晚是平安夜，没有人死亡。")
            self.game_state.last_deaths = []

        self.game_state.current_night_target = None

    async def run_discussion_phase(self):
        """Run day discussion phase."""
        self.game_state.transition_to(GamePhase.DAY_DISCUSSION)
        self._announce("\n🗣️ 现在进入发言环节")

        alive_players = [p for p in self.players if p.player.is_alive()]
        speeches = []

        for position, agent in enumerate(alive_players, 1):
            self._announce(f"\n   请玩家{agent.player.id} ({agent.player.name}) 发言...")

            speech = await agent.day_speech(
                self.game_state,
                position,
                speeches
            )

            speeches.append({
                "speaker": agent.player.id,
                "content": speech
            })

            self._log(f"   玩家{agent.player.id}: {speech}")

            # Store in game state
            self.game_state.current_discussion.append({
                "player_id": agent.player.id,
                "speech": speech
            })

            # Other players observe this speech
            for other_agent in self.players:
                if other_agent.player.id != agent.player.id:
                    other_agent.observe_event(
                        "speech",
                        f"玩家{agent.player.id}说: {speech[:100]}...",
                        speaker=agent.player.id
                    )

    async def run_voting_phase(self):
        """Run day voting phase."""
        self.game_state.transition_to(GamePhase.DAY_VOTE)
        self._announce("\n🗳️ 现在开始投票")

        alive_players = [p for p in self.players if p.player.is_alive()]
        votes = {}

        for agent in alive_players:
            self._log(f"   玩家{agent.player.id} 正在投票...", "debug")

            target = await agent.day_vote(
                self.game_state,
                self.game_state.current_discussion
            )

            if target is not None:
                votes[target] = votes.get(target, 0) + 1
                self.game_state.add_vote(agent.player.id, target)
                self._log(f"      玩家{agent.player.id} -> 玩家{target}", "debug")

        # Announce voting results
        self._announce("\n投票结果:")
        for target, count in sorted(votes.items(), key=lambda x: -x[1]):
            self._announce(f"   玩家{target}: {count} 票")

        # Eliminate player with most votes
        if votes:
            eliminated_id = max(votes, key=votes.get)
            eliminated = self.game_state.get_player(eliminated_id)

            if eliminated and eliminated.is_alive():
                eliminated.kill()
                role_reveal = "狼人" if eliminated.role == Role.WEREWOLF else "村民"

                self._announce(f"\n💀 玩家{eliminated_id} ({eliminated.name}) 被投票处决！")
                self._announce(f"   身份揭晓: {role_reveal}")

                # Record in memory
                for agent in self.players:
                    agent.observe_event(
                        "execution",
                        f"玩家{eliminated_id}被投票处决，身份是{role_reveal}",
                        eliminated=eliminated_id,
                        role=eliminated.role.value
                    )

    async def check_game_end(self) -> Optional[GameResult]:
        """Check if game has ended."""
        winner = self.game_state.check_win_condition()

        if winner:
            self.end_time = datetime.now()
            duration = (self.end_time - self.start_time).total_seconds()

            winner_name = "狼人" if winner == Role.WEREWOLF else "村民"
            self._announce(f"\n\n🏆 游戏结束！{winner_name}获胜！")
            self._announce(f"   总回合数: {self.game_state.round}")

            # Gather statistics
            player_stats = [agent.get_stats() for agent in self.players]

            return GameResult(
                winner=winner,
                total_rounds=self.game_state.round,
                total_events=len(self.game_state.events),
                player_stats=player_stats,
                duration_seconds=duration
            )

        return None

    async def run_game(self) -> GameResult:
        """Run the complete game loop."""
        await self.start_game()

        max_rounds = 20  # Prevent infinite games

        while self.game_state.round < max_rounds:
            # Night phase
            await self.run_night_phase()

            # Check win condition
            result = await self.check_game_end()
            if result:
                return result

            # Day announcement
            await self.run_day_announce()

            # Check win condition
            result = await self.check_game_end()
            if result:
                return result

            # Discussion phase
            await self.run_discussion_phase()

            # Voting phase
            await self.run_voting_phase()

            # Check win condition
            result = await self.check_game_end()
            if result:
                return result

        # Max rounds reached
        self._announce("\n达到最大回合数，游戏结束。")
        return GameResult(
            winner=Role.VILLAGER,  # Default
            total_rounds=self.game_state.round,
            total_events=len(self.game_state.events),
            player_stats=[agent.get_stats() for agent in self.players],
            duration_seconds=0
        )


async def run_game_session(
    players: List[PlayerAgent],
    verbose: bool = True
) -> GameResult:
    """Run a complete game session."""
    moderator = ModeratorAgent(players, verbose=verbose)
    return await moderator.run_game()
