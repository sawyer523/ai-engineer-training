"""Memory management system with vector database support."""

import os
import pickle
import json
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np

# Try to import FAISS
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("Warning: FAISS not available. Install with: pip install faiss-cpu")


@dataclass
class Memory:
    """A single memory entry."""
    content: str
    timestamp: datetime
    memory_type: str  # "episodic", "semantic", "working"
    source: str  # "observation", "conversation", "action", "reflection"
    importance: float = 0.5  # 0-1 score
    embedding: Optional[np.ndarray] = None
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "memory_type": self.memory_type,
            "source": self.source,
            "importance": self.importance,
            "metadata": self.metadata
        }


class MemoryManager:
    """Manages agent memories with vector database support."""

    def __init__(
        self,
        agent_id: int,
        agent_name: str,
        embedding_dim: int = 1536,
        use_faiss: bool = True
    ):
        self.agent_id = agent_id
        self.agent_name = agent_name
        self.embedding_dim = embedding_dim
        self.use_faiss = use_faiss and FAISS_AVAILABLE

        # Memory storage
        self.episodic_memories: List[Memory] = []  # Specific events
        self.semantic_memories: List[Memory] = []  # General knowledge
        self.working_memory: List[Memory] = []  # Current context

        # Vector index for semantic search
        self.index: Optional[Any] = None
        self.memory_texts: List[str] = []

        if self.use_faiss:
            self._init_faiss_index()

    def _init_faiss_index(self):
        """Initialize FAISS index."""
        self.index = faiss.IndexFlatL2(self.embedding_dim)

    def add_memory(
        self,
        content: str,
        memory_type: str = "episodic",
        source: str = "observation",
        importance: float = 0.5,
        embedding: Optional[np.ndarray] = None,
        metadata: dict = None
    ):
        """Add a new memory."""
        memory = Memory(
            content=content,
            timestamp=datetime.now(),
            memory_type=memory_type,
            source=source,
            importance=importance,
            embedding=embedding,
            metadata=metadata or {}
        )

        # Add to appropriate memory list
        if memory_type == "episodic":
            self.episodic_memories.append(memory)
        elif memory_type == "semantic":
            self.semantic_memories.append(memory)
        elif memory_type == "working":
            self.working_memory.append(memory)

        # Add to vector index if embedding provided
        if embedding is not None and self.use_faiss:
            self.memory_texts.append(content)
            self.index.add(embedding.reshape(1, -1))

    def search_memories(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        memory_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Search memories by semantic similarity."""
        if not self.use_faiss or self.index.ntotal == 0:
            # Return recent memories if no vector search available
            memories = self.episodic_memories + self.semantic_memories
            if memory_type:
                memories = [m for m in memories if m.memory_type == memory_type]
            return [{"memory": m, "score": 0.0} for m in memories[-top_k:]]

        # Search by similarity
        query_embedding = query_embedding.reshape(1, -1)
        distances, indices = self.index.search(query_embedding, min(top_k, self.index.ntotal))

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < len(self.memory_texts):
                results.append({
                    "content": self.memory_texts[idx],
                    "score": float(dist),
                    "memory_type": "semantic"
                })

        return results

    def get_recent_memories(
        self,
        n: int = 10,
        memory_type: Optional[str] = None
    ) -> List[Memory]:
        """Get recent memories."""
        memories = self.episodic_memories + self.semantic_memories
        if memory_type:
            memories = [m for m in memories if m.memory_type == memory_type]
        return memories[-n:]

    def get_important_memories(
        self,
        threshold: float = 0.7,
        memory_type: Optional[str] = None
    ) -> List[Memory]:
        """Get memories above importance threshold."""
        memories = self.episodic_memories + self.semantic_memories
        if memory_type:
            memories = [m for m in memories if m.memory_type == memory_type]
        return [m for m in memories if m.importance >= threshold]

    def get_memories_about_player(self, player_id: int) -> List[Memory]:
        """Get memories specifically about a player."""
        relevant = []
        for memory in self.episodic_memories + self.semantic_memories:
            # Check if memory involves this player
            if str(player_id) in memory.content or \
               any(memory.metadata.get(k) == player_id for k in ["target", "voter", "suspect"]):
                relevant.append(memory)
        return relevant

    def summarize_conversation(self, conversation: List[dict]) -> str:
        """Summarize a conversation for memory storage."""
        summary_parts = []
        for msg in conversation:
            speaker = msg.get("speaker", "Unknown")
            content = msg.get("content", "")
            summary_parts.append(f"{speaker}: {content[:100]}...")
        return " | ".join(summary_parts)

    def clear_working_memory(self):
        """Clear working memory for new round."""
        self.working_memory = []

    def get_context_string(
        self,
        include_recent: int = 5,
        include_important: bool = True
    ) -> str:
        """Get formatted context string for LLM."""
        context_parts = []

        # Recent important memories
        recent = self.get_recent_memories(include_recent)
        if recent:
            context_parts.append("=== Recent Memories ===")
            for mem in recent:
                context_parts.append(f"- [{mem.timestamp.strftime('%H:%M')}] {mem.content}")

        # Important memories
        if include_important:
            important = self.get_important_memories(0.7)
            if important:
                context_parts.append("\n=== Important Memories ===")
                for mem in important:
                    context_parts.append(f"- {mem.content}")

        # Working memory
        if self.working_memory:
            context_parts.append("\n=== Current Context ===")
            for mem in self.working_memory:
                context_parts.append(f"- {mem.content}")

        return "\n".join(context_parts)

    def save(self, path: str):
        """Save memories to disk."""
        data = {
            "episodic": [m.to_dict() for m in self.episodic_memories],
            "semantic": [m.to_dict() for m in self.semantic_memories],
            "working": [m.to_dict() for m in self.working_memory],
            "agent_id": self.agent_id,
            "agent_name": self.agent_name
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

    def load(self, path: str):
        """Load memories from disk."""
        if not os.path.exists(path):
            return

        with open(path, 'r') as f:
            data = json.load(f)

        self.agent_id = data["agent_id"]
        self.agent_name = data["agent_name"]

        for mem_dict in data["episodic"]:
            memory = Memory(
                content=mem_dict["content"],
                timestamp=datetime.fromisoformat(mem_dict["timestamp"]),
                memory_type=mem_dict["memory_type"],
                source=mem_dict["source"],
                importance=mem_dict["importance"],
                metadata=mem_dict["metadata"]
            )
            self.episodic_memories.append(memory)

        for mem_dict in data["semantic"]:
            memory = Memory(
                content=mem_dict["content"],
                timestamp=datetime.fromisoformat(mem_dict["timestamp"]),
                memory_type=mem_dict["memory_type"],
                source=mem_dict["source"],
                importance=mem_dict["importance"],
                metadata=mem_dict["metadata"]
            )
            self.semantic_memories.append(memory)


class SimpleMemoryEmbedder:
    """Simple embedding generator for memories when no API available."""

    def __init__(self, dim: int = 1536):
        self.dim = dim

    def embed(self, text: str) -> np.ndarray:
        """Generate a simple hash-based embedding."""
        # This is a placeholder - in production use real embeddings
        # Generate deterministic pseudo-random vector from text
        np.random.seed(hash(text) % (2**32))
        return np.random.randn(self.dim).astype(np.float32)
