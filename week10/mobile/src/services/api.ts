import axios, { type AxiosInstance } from 'axios';
import AsyncStorage from '@react-native-async-storage/async-storage';
import type { ApiResponse, ChatResponse, HealthStatus, ModelInfo, GreetingResponse } from '../types';

class ApiService {
  private client: AxiosInstance;
  private baseURL: string;

  constructor(baseURL: string = process.env.EXPO_PUBLIC_API_URL || 'http://localhost:8000') {
    this.baseURL = baseURL;
    this.client = axios.create({
      baseURL,
      timeout: 30000,
      headers: {},
    });
    this.setupInterceptors();
  }

  private setupInterceptors() {
    this.client.interceptors.request.use(
      async (config) => {
        const method = (config.method || 'get').toLowerCase();
        const tenantFromStorage = (await AsyncStorage.getItem('tenantId')) || undefined;
        const keyFromStorage = await AsyncStorage.getItem('apiKey');
        const tenantFromEnv = (process.env.EXPO_PUBLIC_TENANT_ID as string | undefined) || undefined;
        const keyFromEnv = (process.env.EXPO_PUBLIC_API_KEY as string | undefined) || undefined;
        const tenantId = tenantFromStorage || tenantFromEnv || 'default';
        const apiKey = keyFromStorage || keyFromEnv;
        if (!config.headers) config.headers = {} as any;
        if (tenantId) (config.headers as any)['X-Tenant-ID'] = tenantId;
        if (apiKey) (config.headers as any)['X-API-Key'] = apiKey;
        if (method === 'get' && config.headers) {
          delete (config.headers as any)['Content-Type'];
        }
        Reflect.set(config, '_startTime', Date.now());
        return config;
      },
      (error) => Promise.reject(error)
    );
    this.client.interceptors.response.use(
      (response) => response,
      (error) => Promise.reject(error)
    );
  }

  async sendMessage(query: string, threadId?: string, images?: string[], audio?: string) {
    const response = await this.client.post('/chat', {
      query,
      thread_id: threadId,
      images,
      audio,
    });
    const dataUnknown: unknown = response.data;
    if (typeof dataUnknown === 'object' && dataUnknown !== null && !('code' in dataUnknown)) {
      const chat = dataUnknown as ChatResponse;
      return { code: 0, message: 'OK', data: chat } as ApiResponse<ChatResponse>;
    }
    return dataUnknown as ApiResponse<ChatResponse>;
  }

  async getHealth(): Promise<HealthStatus> {
    const response = await this.client.get('/health');
    return response.data;
  }

  async getModels(): Promise<ModelInfo> {
    const response = await this.client.get('/models/list');
    const dataUnknown: unknown = response.data;
    if (typeof dataUnknown === 'object' && dataUnknown !== null && 'current' in dataUnknown) {
      return dataUnknown as ModelInfo;
    }
    if (typeof dataUnknown === 'object' && dataUnknown !== null && 'data' in dataUnknown) {
      return (dataUnknown as ApiResponse<ModelInfo>).data;
    }
    return response.data as ModelInfo;
  }

  async getGreeting(): Promise<GreetingResponse> {
    const response = await this.client.get('/greet');
    const dataUnknown: unknown = response.data;
    if (typeof dataUnknown === 'object' && dataUnknown !== null && 'message' in dataUnknown && 'options' in dataUnknown) {
      return dataUnknown as GreetingResponse;
    }
    if (typeof dataUnknown === 'object' && dataUnknown !== null && 'data' in dataUnknown) {
      return (dataUnknown as ApiResponse<GreetingResponse>).data;
    }
    return response.data as GreetingResponse;
  }

  async getSuggestions(threadId: string): Promise<EventSource> {
    if (typeof EventSource === 'undefined') {
      throw new Error('EventSource not supported');
    }
    return new EventSource(`${this.baseURL}/suggest/${threadId}`);
  }

  async getOrder(orderId: string): Promise<ApiResponse<any>> {
    const response = await this.client.get(`/api/orders/${orderId}`);
    const dataUnknown: unknown = response.data;
    if (typeof dataUnknown === 'object' && dataUnknown !== null && !('code' in dataUnknown)) {
      return { code: 0, message: 'OK', data: dataUnknown } as ApiResponse;
    }
    return dataUnknown as ApiResponse;
  }
}

export const apiService = new ApiService();
export default apiService;