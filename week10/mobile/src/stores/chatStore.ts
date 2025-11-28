import { create } from 'zustand/index.js';
import { persist } from 'zustand/middleware.js';
import type { ChatState, ChatSession, Message } from '../types';
import { v4 as uuidv4 } from 'uuid';

interface ChatStore extends ChatState {
  // Actions
  setCurrentSession: (session: ChatSession | null) => void;
  addMessage: (message: Message) => void;
  updateMessage: (messageId: string, updates: Partial<Message>) => void;
  createSession: (title?: string) => ChatSession;
  deleteSession: (sessionId: string) => void;
  setLoading: (loading: boolean) => void;
  setConnected: (connected: boolean) => void;
  setError: (error: string | null) => void;
  setThreadId: (threadId: string | null) => void;
  clearCurrentSession: () => void;
}

export const useChatStore = create<ChatStore>()(
  persist(
    (set, get) => ({
      // Initial state
      currentSession: null,
      sessions: [],
      isLoading: false,
      isConnected: false,
      error: null,
      threadId: null,

      // Actions
      setCurrentSession: (session) => set({ currentSession: session }),

      addMessage: (message) => {
        const { currentSession, sessions } = get();
        if (!currentSession) return;

        const updatedSession = {
          ...currentSession,
          messages: [...currentSession.messages, message],
          updatedAt: new Date(),
        };

        const updatedSessions = sessions.map(session =>
          session.id === currentSession.id ? updatedSession : session
        );

        set({
          currentSession: updatedSession,
          sessions: updatedSessions,
        });
      },

      updateMessage: (messageId, updates) => {
        const { currentSession, sessions } = get();
        if (!currentSession) return;

        const updatedMessages = currentSession.messages.map(message =>
          message.id === messageId ? { ...message, ...updates } : message
        );

        const updatedSession = {
          ...currentSession,
          messages: updatedMessages,
          updatedAt: new Date(),
        };

        const updatedSessions = sessions.map(session =>
          session.id === currentSession.id ? updatedSession : session
        );

        set({
          currentSession: updatedSession,
          sessions: updatedSessions,
        });
      },

      createSession: (title = '新对话') => {
        const newSession: ChatSession = {
          id: uuidv4(),
          title,
          messages: [],
          createdAt: new Date(),
          updatedAt: new Date(),
        };

        set((state) => ({
          currentSession: newSession,
          sessions: [newSession, ...state.sessions],
          threadId: uuidv4(),
        }));

        return newSession;
      },

      deleteSession: (sessionId) => {
        set((state) => {
          const filteredSessions = state.sessions.filter(session => session.id !== sessionId);
          const newCurrentSession = state.currentSession?.id === sessionId
            ? (filteredSessions[0] || null)
            : state.currentSession;

          return {
            sessions: filteredSessions,
            currentSession: newCurrentSession,
          };
        });
      },

      setLoading: (loading) => set({ isLoading: loading }),
      setConnected: (connected) => set({ isConnected: connected }),
      setError: (error) => set({ error }),
      setThreadId: (threadId) => set({ threadId }),

      clearCurrentSession: () => {
        set({ currentSession: null, threadId: null });
      },
    }),
    {
      name: 'chat-storage',
      partialize: (state) => ({
        sessions: state.sessions,
        currentSession: state.currentSession,
        threadId: state.threadId,
      }),
    }
  )
);