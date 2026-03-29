"""Player agent implementation using LangChain."""

import os
from typing import Optional, Dict, Any, List
from dataclasses import dataclass

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from ..game.state import Player, Role, GameState, GamePhase
from ..memory.manager import MemoryManager
from ..prompts.templates import get_role_prompt, get_phase_prompt, format_player_list, format_speeches


class PlayerAgent:
    """AI player agent for Werewolf game."""

    def __init__(
        self,
        player: Player,
        api_key: str,
        base_url: str = "https://api.deepseek.com/v1",
        model: str = "deepseek-chat",
        temperature: float = 0.8
    ):
        self.player = player
        self.model_name = model

        # Initialize LLM
        self.llm = ChatOpenAI(
            model=model,
            api_key=api_key,
            base_url=base_url,
            temperature=temperature
        )

        # Initialize memory
        self.memory = MemoryManager(
            agent_id=player.id,
            agent_name=player.name
        )

        # Initial memory about own role
        role_reveal = "狼人" if player.role == Role.WEREWOLF else "村民"
        self.memory.add_memory(
            f"我的身份是{role_reveal}，我的性格是{player.personality}",
            memory_type="semantic",
            source="observation",
            importance=1.0
        )

        # Token tracking for cost analysis
        self.total_tokens = 0
        self.call_count = 0

    def _build_system_prompt(self, game_state: GameState) -> str:
        """Build system prompt for current state."""
        # Get teammates if werewolf
        teammates = []
        if self.player.role == Role.WEREWOLF:
            werewolves = game_state.get_alive_werewolves()
            teammates = [w.name for w in werewolves if w.id != self.player.id]

        # Build context
        memory_context = self.memory.get_context_string(include_recent=5)

        game_context = self._build_game_context(game_state)

        return get_role_prompt(
            role="狼人" if self.player.role == Role.WEREWOLF else "村民",
            personality=self.player.personality,
            teammates=teammates,
            memory_context=memory_context,
            game_context=game_context
        )

    def _build_game_context(self, game_state: GameState) -> str:
        """Build game context string."""
        alive = game_state.get_alive_players()
        dead = [p for p in game_state.players if not p.is_alive()]

        context = f"""
当前回合: {game_state.round}
当前阶段: {game_state.phase.value}

存活玩家 ({len(alive)}人):
{format_player_list([p.to_dict() for p in alive])}

已死亡玩家:
{format_player_list([p.to_dict() for p in dead])}
"""
        return context

    async def think(
        self,
        prompt: str,
        game_state: GameState
    ) -> str:
        """Generate a response using the LLM."""
        system_prompt = self._build_system_prompt(game_state)

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=prompt)
        ]

        try:
            response = await self.llm.ainvoke(messages)

            # Track token usage
            if hasattr(response, 'response_metadata'):
                usage = response.response_metadata.get('token_usage', {})
                self.total_tokens += usage.get('total_tokens', 0)
            self.call_count += 1

            return response.content
        except Exception as e:
            print(f"Error in LLM call: {e}")
            return f"[Error generating response: {str(e)}]"

    async def night_discussion(
        self,
        game_state: GameState,
        discussion_history: List[dict]
    ) -> str:
        """Werewolf night discussion phase."""
        if self.player.role != Role.WEREWOLF:
            return ""

        alive = game_state.get_alive_players()
        alive_players = [p.to_dict() for p in alive if p.role == Role.VILLAGER]

        prompt = get_phase_prompt(
            "night",
            alive_players=format_player_list(alive_players),
            discussion=format_speeches(discussion_history)
        )

        response = await self.think(prompt, game_state)

        # Store in memory
        self.memory.add_memory(
            f"夜晚讨论: {response}",
            memory_type="episodic",
            source="conversation",
            importance=0.6
        )

        return response

    async def night_vote(
        self,
        game_state: GameState,
        discussion: List[dict]
    ) -> Optional[int]:
        """Werewolf night vote for target."""
        if self.player.role != Role.WEREWOLF:
            return None

        alive_villagers = game_state.get_alive_villagers()
        if not alive_villagers:
            return None

        # Build voting prompt
        prompt = f"""基于夜晚讨论，你需要选择要杀害的目标。

可选目标（村民）:
{format_player_list([p.to_dict() for p in alive_villagers])}

请直接输出目标玩家ID数字，不要其他内容。
"""

        response = await self.think(prompt, game_state)

        # Parse target ID
        try:
            target_id = int(response.strip().split()[0])
            # Validate target
            if any(p.id == target_id and p.role == Role.VILLAGER for p in alive_villagers):
                self.memory.add_memory(
                    f"夜晚投票杀害玩家{target_id}",
                    memory_type="episodic",
                    source="action",
                    importance=0.8,
                    metadata={"target": target_id}
                )
                return target_id
        except (ValueError, IndexError):
            pass

        # Fallback to random villager
        import random
        target = random.choice(alive_villagers)
        return target.id

    async def day_speech(
        self,
        game_state: GameState,
        position: int,
        previous_speeches: List[dict]
    ) -> str:
        """Generate day speech."""
        # Build context
        deaths = game_state.last_deaths
        deaths_str = f"玩家{deaths[0]}" if deaths else "无人"

        prompt = get_phase_prompt(
            "day_discussion",
            deaths=deaths_str,
            speaker_position=position,
            previous_speeches=format_speeches(previous_speeches),
            observations=self._get_observations(game_state)
        )

        response = await self.think(prompt, game_state)

        # Store speech in memory
        self.memory.add_memory(
            f"第{game_state.round}轮发言: {response}",
            memory_type="episodic",
            source="conversation",
            importance=0.7
        )

        return response

    async def day_vote(
        self,
        game_state: GameState,
        speeches: List[dict]
    ) -> Optional[int]:
        """Cast day vote."""
        alive = game_state.get_alive_players()
        other_players = [p for p in alive if p.id != self.player.id]

        prompt = get_phase_prompt(
            "day_vote",
            alive_players=format_player_list([p.to_dict() for p in other_players]),
            speech_summary=format_speeches(speeches),
            suspicions=self._get_suspicions(game_state)
        )

        response = await self.think(prompt, game_state)

        # Parse vote
        try:
            # Try to extract number from response
            import re
            match = re.search(r'\d+', response)
            if match:
                target_id = int(match.group())
                if any(p.id == target_id for p in other_players):
                    self.memory.add_memory(
                        f"白天投票给玩家{target_id}",
                        memory_type="episodic",
                        source="action",
                        importance=0.8,
                        metadata={"vote_target": target_id}
                    )
                    return target_id
        except (ValueError, AttributeError):
            pass

        # Fallback: vote for random other player
        import random
        target = random.choice(other_players)
        return target.id

    def _get_observations(self, game_state: GameState) -> str:
        """Get current observations."""
        observations = []

        # Deaths
        if game_state.last_deaths:
            observations.append(f"昨晚玩家{game_state.last_deaths[0]}死亡")

        # Recent votes
        recent_votes = game_state.get_votes_for_round(game_state.round - 1, "day")
        if recent_votes:
            vote_summary = "上轮投票: "
            vote_summary += ", ".join([f"玩家{v.voter_id}->玩家{v.target_id}" for v in recent_votes])
            observations.append(vote_summary)

        return "\n".join(observations) if observations else "暂无特别观察"

    def _get_suspicions(self, game_state: GameState) -> str:
        """Get current suspicions based on memory."""
        suspicious_memories = self.memory.get_memories_about_player(-1)  # Get all

        # Simple suspicion extraction
        suspicions = []
        for mem in self.memory.get_recent_memories(10):
            if "怀疑" in mem.content or "可疑" in mem.content:
                suspicions.append(mem.content)

        return "\n".join(suspicions) if suspicions else "暂无明确怀疑对象"

    def observe_event(self, event_type: str, description: str, **metadata):
        """Record an event in memory."""
        importance = 0.5

        # Higher importance for certain events
        if "死亡" in event_type or "kill" in event_type.lower():
            importance = 0.9
        elif "投票" in event_type or "vote" in event_type.lower():
            importance = 0.7

        self.memory.add_memory(
            description,
            memory_type="episodic",
            source="observation",
            importance=importance,
            metadata=metadata
        )

    def get_stats(self) -> dict:
        """Get agent statistics."""
        return {
            "player_id": self.player.id,
            "player_name": self.player.name,
            "role": self.player.role.value,
            "total_tokens": self.total_tokens,
            "call_count": self.call_count,
            "memory_count": len(self.memory.episodic_memories) + len(self.memory.semantic_memories)
        }


def create_player_agent(
    player_id: int,
    name: str,
    role: Role,
    personality: str = "neutral",
    **kwargs
) -> PlayerAgent:
    """Factory function to create a player agent."""
    player = Player(
        id=player_id,
        name=name,
        role=role,
        personality=personality
    )

    api_key = kwargs.get("api_key", os.getenv("DEEPSEEK_API_KEY"))
    base_url = kwargs.get("base_url", os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"))
    model = kwargs.get("model", os.getenv("DEEPSEEK_MODEL", "deepseek-chat"))

    return PlayerAgent(
        player=player,
        api_key=api_key,
        base_url=base_url,
        model=model,
        temperature=kwargs.get("temperature", 0.8)
    )
