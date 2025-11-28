import axios, { type AxiosInstance } from 'axios';
import type { ApiResponse, ChatResponse, HealthStatus, ModelInfo, GreetingResponse } from '../types';

class ApiService {
  private client: AxiosInstance;
  private baseURL: string;

  constructor(baseURL: string = 'http://localhost:8000') {
    this.baseURL = baseURL;
    this.client = axios.create({
      baseURL,
      timeout: 30000,
      headers: {},
    });

    console.log('[API] 初始化', { baseURL });
    this.setupInterceptors();
  }

  private setupInterceptors() {
    // Request interceptor
    this.client.interceptors.request.use(
      (config) => {
        const method = (config.method || 'get').toLowerCase();
        const urlStr = config.url ? (config.url.startsWith('http') ? config.url : `${this.baseURL}${config.url}`) : this.baseURL;
        let pathname = '/';
        try {
          pathname = new URL(urlStr).pathname;
        } catch {
          void 0;
        }

        const simpleGetPaths = ['/health', '/models/list', '/greet'];
        const isSimpleGet = method === 'get' && simpleGetPaths.includes(pathname);

        if (!isSimpleGet) {
          const tenantId = localStorage.getItem('tenantId') || 'default';
          if (tenantId) config.headers['X-Tenant-ID'] = tenantId;

          const apiKey = localStorage.getItem('apiKey');
          if (apiKey) config.headers['X-API-Key'] = apiKey;
        }

        if (method === 'get' && config.headers) {
          delete (config.headers as any)['Content-Type'];
        }
        Reflect.set(config, '_startTime', Date.now());
        const tenantIdLog = localStorage.getItem('tenantId') || 'default';
        console.log('[API] 请求', {
          method,
          url: urlStr,
          tenantId: tenantIdLog,
          params: config.params,
          data: config.data,
        });
        return config;
      },
      (error) => {
        console.log('[API] 请求拦截错误', error);
        return Promise.reject(error);
      }
    );

    // Response interceptor
    this.client.interceptors.response.use(
      (response) => {
        const start = Reflect.get(response.config, '_startTime') as number | undefined;
        const duration = typeof start === 'number' ? Date.now() - start : undefined;
        console.log('[API] 响应', {
          url: response.config.url,
          status: response.status,
          duration,
        });
        return response;
      },
      (error) => {
        try {
          const cfg = error.config || {};
          const start = Reflect.get(cfg, '_startTime') as number | undefined;
          const duration = typeof start === 'number' ? Date.now() - start : undefined;
          const url = cfg.url || '';
          const status = error.response?.status;
          console.log('[API] 响应错误', { url, status, duration, message: error.message });
        } catch {
          void 0;
        }
        if (error.response?.status === 401) {
          // Handle unauthorized access
          localStorage.removeItem('apiKey');
          window.location.href = '/login';
        }
        return Promise.reject(error);
      }
    );
  }

  // Chat API
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

  // Health check
  async getHealth() {
    const response = await this.client.get<HealthStatus>('/health');
    return response.data;
  }

  // Models management
  async getModels() {
    const response = await this.client.get<ApiResponse<ModelInfo>>('/models/list');
    return response.data;
  }

  async switchModel(modelName: string) {
    const response = await this.client.post<ApiResponse>('/models/switch', {
      name: modelName,
    });
    return response.data;
  }

  // Greeting
  async getGreeting() {
    const response = await this.client.get<ApiResponse<GreetingResponse>>('/greet');
    return response.data;
  }

  // Suggestions stream
  async getSuggestions(threadId: string): Promise<EventSource> {
    return new EventSource(`${this.baseURL}/suggest/${threadId}`);
  }

  // Order queries
  async getOrder(orderId: string) {
    const response = await this.client.get(`/api/orders/${orderId}`);
    const dataUnknown: unknown = response.data;
    if (typeof dataUnknown === 'object' && dataUnknown !== null && !('code' in dataUnknown)) {
      return { code: 0, message: 'OK', data: dataUnknown } as ApiResponse;
    }
    return dataUnknown as ApiResponse;
  }

  // Vector operations
  async addVectors(items: Array<{ text: string; metadata?: Record<string, any>; id?: string }>) {
    const response = await this.client.post<ApiResponse>('/api/v1/vectors/items', {
      items,
    });
    return response.data;
  }

  async deleteVectors(ids: string[]) {
    const response = await this.client.delete<ApiResponse>('/api/v1/vectors/items', {
      data: { ids },
    });
    return response.data;
  }
}

// Create singleton instance
export const apiService = new ApiService();

export default apiService;