import { apiService } from './api';
import type { ApiResponse, ChatResponse, SuggestionEvent } from '../types';

export class ChatService {
  private eventSource: EventSource | null = null;
  private suggestionCallbacks: Map<string, (event: SuggestionEvent) => void> = new Map();

  async sendMessage(
    content: string,
    threadId?: string,
    images?: string[],
    audio?: string
  ): Promise<ChatResponse> {
    try {
      const response = await apiService.sendMessage(content, threadId, images, audio) as ApiResponse<ChatResponse>;
      if (response.code !== 0) {
        throw new Error(response.message || '发送消息失败');
      }
      return response.data;
    } catch (error) {
      console.error('Failed to send message:', error);
      throw error;
    }
  }

  async executeCommand(command: string, threadId: string): Promise<ChatResponse> {
    try {
      const response = await apiService.sendMessage(command, threadId) as ApiResponse<ChatResponse>;
      if (response.code !== 0) {
        throw new Error(response.message || '执行命令失败');
      }
      return response.data;
    } catch (error) {
      console.error('Failed to execute command:', error);
      throw error;
    }
  }

  startSuggestionStream(threadId: string, callback: (event: SuggestionEvent) => void): void {
    this.suggestionCallbacks.set(threadId, callback);
    if (this.eventSource) {
      this.eventSource.close();
    }
    try {
      const es = apiService.getSuggestions(threadId);
      if (typeof (es as any).then === 'function') {
        (es as Promise<EventSource>).then((eventSource) => {
          this.eventSource = eventSource;
          this.eventSource.onmessage = (event) => {
            try {
              const data = JSON.parse(event.data) as SuggestionEvent;
              const cb = this.suggestionCallbacks.get(threadId);
              if (cb) cb(data);
              if (data.final) {
                this.stopSuggestionStream();
              }
            } catch {}
          };
          this.eventSource.onerror = () => {
            this.stopSuggestionStream();
          };
          setTimeout(() => {
            this.stopSuggestionStream();
          }, 15000);
        }).catch(() => {
          this.stopSuggestionStream();
        });
      } else {
        this.stopSuggestionStream();
      }
    } catch {
      this.stopSuggestionStream();
    }
  }

  stopSuggestionStream(): void {
    if (this.eventSource) {
      this.eventSource.close();
      this.eventSource = null;
    }
    this.suggestionCallbacks.clear();
  }

  async getHealth() {
    return apiService.getHealth();
  }

  async getModels() {
    return apiService.getModels();
  }

  async getGreeting() {
    return apiService.getGreeting();
  }
}

export const chatService = new ChatService();