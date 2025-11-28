import { apiService } from './api';
import type { ChatResponse, SuggestionEvent } from '../types';

export class ChatService {
  private eventSource: EventSource | null = null;
  private suggestionCallbacks: Map<string, (event: SuggestionEvent) => void> = new Map();

  async sendMessage(query: string, threadId?: string, images?: string[], audio?: string): Promise<ChatResponse> {
    try {
      const response = await apiService.sendMessage(query, threadId, images, audio);
      if (response.code !== 0) {
        throw new Error(response.message || '发送消息失败');
      }
      return response.data;
    } catch (error) {
      console.error('Send message error:', error);
      throw error;
    }
  }

  async startSuggestionStream(threadId: string, callback: (event: SuggestionEvent) => void) {
    this.suggestionCallbacks.set(threadId, callback);
    
    if (this.eventSource) {
      this.eventSource.close();
    }

    try {
      this.eventSource = await apiService.getSuggestions(threadId);
      
      this.eventSource.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data) as SuggestionEvent;
          const callback = this.suggestionCallbacks.get(threadId);
          if (callback) {
            callback(data);
          }
          
          if (data.final) {
            this.stopSuggestionStream();
          }
        } catch (error) {
          console.error('Parse suggestion event error:', error);
        }
      };

      this.eventSource.onerror = (error) => {
        console.error('Suggestion stream error:', error);
        this.stopSuggestionStream();
      };

      // Set timeout for suggestion stream
      setTimeout(() => {
        this.stopSuggestionStream();
      }, 15000); // 15 seconds timeout
      
    } catch (error) {
      console.error('Start suggestion stream error:', error);
      this.stopSuggestionStream();
    }
  }

  stopSuggestionStream() {
    if (this.eventSource) {
      this.eventSource.close();
      this.eventSource = null;
    }
    this.suggestionCallbacks.clear();
  }

  async executeCommand(command: string, threadId: string) {
    try {
      const response = await apiService.sendMessage(command, threadId);
      if (response.code !== 0) {
        throw new Error(response.message || '执行命令失败');
      }
      return response.data;
    } catch (error) {
      console.error('Execute command error:', error);
      throw error;
    }
  }
}

export const chatService = new ChatService();
export default chatService;