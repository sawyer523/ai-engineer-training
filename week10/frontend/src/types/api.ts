import type { Message } from './chat';

export interface ApiResponse<T = any> {
  code: number;
  message: string;
  data: T;
}

export interface HealthStatus {
  model: string;
  kb_index: boolean;
  orders_db: boolean;
  metrics: Record<string, MetricSnapshot>;
}

export interface MetricSnapshot {
  count: number;
  min_ms: number;
  max_ms: number;
  avg_ms: number;
  p95_ms: number;
}

export interface ModelInfo {
  current: string;
  models: string[];
}

export interface GreetingResponse {
  message: string;
  options: GreetingOption[];
}

export interface GreetingOption {
  key: string;
  title: string;
  desc: string;
}

export interface CommandResponse {
  commands?: CommandInfo[];
  history?: Message[];
  reset?: boolean;
}

export interface CommandInfo {
  cmd: string;
  desc: string;
}

export interface SuggestionEvent {
  route?: string;
  suggestions: string[];
  event: 'react_start' | 'react' | 'error' | 'suggest';
  final?: boolean;
  error?: {
    code: string;
    message: string;
  };
}