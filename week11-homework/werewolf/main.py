"""Main entry point for Werewolf game."""

import asyncio
import os
import sys
from pathlib import Path
from typing import List
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

from werewolf.game.state import Role
from werewolf.agents.player import create_player_agent
from werewolf.agents.moderator import run_game_session
from werewolf.utils.logger import GameLogger


# Load environment variables
load_dotenv()


def setup_game() -> List:
    """Setup game with player agents."""
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        print("Error: DEEPSEEK_API_KEY not found in environment variables.")
        print("Please create a .env file with your API key.")
        sys.exit(1)

    base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
    model = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

    # Define players: 3 villagers + 2 werewolves
    player_configs = [
        # Villagers
        {"id": 1, "name": "张三", "role": Role.VILLAGER, "personality": "observant"},
        {"id": 2, "name": "李四", "role": Role.VILLAGER, "personality": "cautious"},
        {"id": 3, "name": "王五", "role": Role.VILLAGER, "personality": "aggressive"},
        # Werewolves
        {"id": 4, "name": "赵六", "role": Role.WEREWOLF, "personality": "charismatic"},
        {"id": 5, "name": "钱七", "role": Role.WEREWOLF, "personality": "suspicious"},
    ]

    players = []
    for config in player_configs:
        agent = create_player_agent(
            player_id=config["id"],
            name=config["name"],
            role=config["role"],
            personality=config["personality"],
            api_key=api_key,
            base_url=base_url,
            model=model,
            temperature=0.8
        )
        players.append(agent)

    return players


def print_cost_analysis(result, logger: GameLogger):
    """Print cost and complexity analysis."""
    print("\n" + "="*60)
    print("📊 成本与复杂度分析报告".center(60))
    print("="*60)

    total_tokens = sum(stat["total_tokens"] for stat in result.player_stats)
    total_calls = sum(stat["call_count"] for stat in result.player_stats)

    print(f"\n游戏时长: {result.duration_seconds:.2f} 秒")
    print(f"总回合数: {result.total_rounds}")
    print(f"总事件数: {result.total_events}")

    print(f"\nLLM 调用统计:")
    print(f"  总调用次数: {total_calls}")
    print(f"  总 Token 数: {total_tokens:,}")

    if total_calls > 0:
        avg_tokens_per_call = total_tokens / total_calls
        print(f"  平均每次调用 Token: {avg_tokens_per_call:.0f}")

    print(f"\n玩家统计:")
    for stat in result.player_stats:
        role_icon = "🐺" if stat["role"] == "werewolf" else "👤"
        print(f"  {role_icon} {stat['player_name']} (ID:{stat['player_id']})")
        print(f"      角色: {stat['role']}")
        print(f"      调用: {stat['call_count']} 次, {stat['total_tokens']:,} tokens")
        print(f"      记忆: {stat['memory_count']} 条")

    # Cost estimation (DeepSeek pricing)
    # DeepSeek-Chat: ~0.14 CNY/1M input tokens, ~0.28 CNY/1M output tokens
    # Assuming 50/50 input/output split
    estimated_cost_cny = (total_tokens / 2) * 0.14 / 1_000_000 + (total_tokens / 2) * 0.28 / 1_000_000
    print(f"\n💰 预估成本 (DeepSeek): ¥{estimated_cost_cny:.4f}")

    print("\n" + "="*60)


async def main():
    """Main game execution."""
    print("\n" + "="*60)
    print("🐺 狼人杀游戏系统 - LangChain + LangGraph".center(60))
    print("="*60 + "\n")

    # Setup game
    players = setup_game()

    # Create logger
    logger = GameLogger()

    try:
        # Run game
        result = await run_game_session(players, verbose=True)

        # Print cost analysis
        print_cost_analysis(result, logger)

        # Print log summary
        logger.print_summary()

        # Save game log
        log_dir = Path(__file__).parent.parent / "game_logs"
        log_dir.mkdir(exist_ok=True)
        log_path = log_dir / f"game_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        logger.save(log_path)

        print(f"\n✅ 游戏完成！日志已保存至: {log_path}")

    except KeyboardInterrupt:
        print("\n\n⚠️  游戏被用户中断")
    except Exception as e:
        print(f"\n❌ 游戏执行出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
