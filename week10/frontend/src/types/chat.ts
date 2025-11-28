export interface Message {
  id: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  timestamp: Date;
  images?: string[];
  audio?: string;
  sources?: Source[];
  route?: string;
  suggestions?: string[];
  error?: string;
}

export interface Source {
  title: string;
  content: string;
  url?: string;
  metadata?: Record<string, any>;
}

export interface ChatSession {
  id: string;
  title: string;
  messages: Message[];
  createdAt: Date;
  updatedAt: Date;
  tenantId?: string;
}

export interface ChatState {
  currentSession: ChatSession | null;
  sessions: ChatSession[];
  isLoading: boolean;
  isConnected: boolean;
  error: string | null;
  threadId: string | null;
}

export interface SendMessageParams {
  content: string;
  images?: string[];
  audio?: string;
  asrLanguage?: string;
  asrItn?: boolean;
}

export interface ChatResponse {
  route: string;
  answer: string;
  sources?: Source[];
  suggestions?: string[];
  error?: string;
  commands?: { cmd: string; desc: string }[];
  history?: Message[];
  reset?: boolean;
}

export interface OrderInfo {
  orderId: string;
  status: string;
  amount: number;
  updatedAt: string;
  startTime?: string;
  enrollTime?: string;
}

export interface SystemConfig {
  currentModel: string;
  supportedModels: string[];
  tenantId: string;
  kbIndexAvailable: boolean;
  ordersDbAvailable: boolean;
}