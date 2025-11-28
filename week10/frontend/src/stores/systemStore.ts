import { create } from 'zustand';
import type { SystemConfig } from '../types';

interface SystemStore {
  config: SystemConfig | null;
  isLoading: boolean;
  error: string | null;
  
  // Actions
  setConfig: (config: SystemConfig) => void;
  setLoading: (loading: boolean) => void;
  setError: (error: string | null) => void;
  updateCurrentModel: (model: string) => void;
}

export const useSystemStore = create<SystemStore>()((set) => ({
  config: null,
  isLoading: false,
  error: null,

  setConfig: (config) => set({ config }),
  setLoading: (loading) => set({ isLoading: loading }),
  setError: (error) => set({ error }),
  updateCurrentModel: (model) => 
    set((state) => ({
      config: state.config ? { ...state.config, currentModel: model } : null,
    })),
}));