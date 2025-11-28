import { create } from 'zustand/index.js';
import type { SystemConfig } from '../types';

interface SystemStore {
  config: SystemConfig | null;
  isLoading: boolean;
  setConfig: (config: SystemConfig) => void;
  setLoading: (loading: boolean) => void;
}

export const useSystemStore = create<SystemStore>((set) => ({
  config: null,
  isLoading: false,
  setConfig: (config) => set({ config }),
  setLoading: (loading) => set({ isLoading: loading }),
}));