import { create } from 'zustand';
import { persist } from 'zustand/middleware';
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
      currentSession: null,
      sessions: [],
      isLoading: false,
      isConnected: false,
      error: null,
      threadId: null,

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

        const updatedMessages = currentSession.messages.map(msg =>
          msg.id === messageId ? { ...msg, ...updates } : msg
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
          tenantId: localStorage.getItem('tenantId') || 'default',
        };

        set(state => ({
          currentSession: newSession,
          sessions: [newSession, ...state.sessions],
          threadId: uuidv4(),
        }));

        return newSession;
      },

      deleteSession: (sessionId) => {
        const { sessions, currentSession } = get();
        const updatedSessions = sessions.filter(session => session.id !== sessionId);
        
        if (currentSession?.id === sessionId) {
          set({
            currentSession: updatedSessions[0] || null,
            sessions: updatedSessions,
            threadId: updatedSessions[0] ? get().threadId : uuidv4(),
          });
        } else {
          set({ sessions: updatedSessions });
        }
      },

      setLoading: (loading) => set({ isLoading: loading }),
      setConnected: (connected) => set({ isConnected: connected }),
      setError: (error) => set({ error }),
      setThreadId: (threadId) => set({ threadId }),

      clearCurrentSession: () => {
        const { createSession } = get();
        createSession('新对话');
      },
    }),
    {
      name: 'ai-agent-chat-store',
      partialize: (state) => ({
        sessions: state.sessions,
        currentSession: state.currentSession,
        threadId: state.threadId,
      }),
    }
  )
);