import React, { useState, useRef, useEffect } from 'react';
import { Input, Button, App as AntdApp } from 'antd';
import { SendOutlined, AudioOutlined, PictureOutlined, LoadingOutlined } from '@ant-design/icons';
import type { Message } from '../../types';
import { useChatStore } from '../../stores';
import { chatService } from '../../services';
import MessageList from './MessageList';
import AudioRecorder from './AudioRecorder';
import ImageUploader from './ImageUploader';
import './ChatInterface.css';

const { TextArea } = Input;

interface ChatInterfaceProps {
  className?: string;
}

const ChatInterface: React.FC<ChatInterfaceProps> = ({ className = '' }) => {
  const { message } = AntdApp.useApp();
  const [inputValue, setInputValue] = useState('');
  const [uploadedImages, setUploadedImages] = useState<string[]>([]);
  const [showAudioRecorder, setShowAudioRecorder] = useState(false);
  const [showImageUploader, setShowImageUploader] = useState(false);
  
  const {
    currentSession,
    isLoading,
    threadId,
    addMessage,
    updateMessage,
    setLoading,
    createSession,
  } = useChatStore();

  const messagesEndRef = useRef<HTMLDivElement>(null);
  const suggestionTimeoutRef = useRef<number | null>(null);

  useEffect(() => {
    // Create initial session if none exists
    if (!currentSession) {
      createSession('AI助手对话');
    }
  }, [currentSession, createSession]);

  useEffect(() => {
    // Scroll to bottom when messages change
    scrollToBottom();
  }, [currentSession?.messages]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  const handleSendMessage = async () => {
    if (!inputValue.trim() && uploadedImages.length === 0) {
      message.warning('请输入消息或上传图片');
      return;
    }

    if (!threadId) {
      message.error('会话初始化失败，请刷新页面重试');
      return;
    }

    const userMessage: Message = {
      id: Date.now().toString(),
      role: 'user',
      content: inputValue.trim(),
      timestamp: new Date(),
      images: uploadedImages.length > 0 ? uploadedImages : undefined,
    };

    addMessage(userMessage);
    setInputValue('');
    setUploadedImages([]);
    setLoading(true);

    try {
      // Handle special commands
      if (inputValue.trim().startsWith('/')) {
        await handleCommand(inputValue.trim(), threadId);
        return;
      }

      // Send message to backend
      const response = await chatService.sendMessage(
        inputValue.trim(),
        threadId,
        uploadedImages.length > 0 ? uploadedImages : undefined,
        undefined // audio data
      );

      // Add assistant response
      const assistantMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: response.answer,
        timestamp: new Date(),
        sources: response.sources,
        route: response.route,
      };

      addMessage(assistantMessage);

      // Start suggestion stream
      if (response.suggestions && response.suggestions.length > 0) {
        // Direct suggestions available
        updateMessage(assistantMessage.id, { suggestions: response.suggestions });
      } else {
        // Start streaming suggestions
        startSuggestionStream(threadId, assistantMessage.id);
      }

    } catch (error) {
      console.error('Failed to send message:', error);
      
      let errorMessage = '消息发送失败，请重试';
      if (error instanceof Error) {
        errorMessage = error.message;
      }
      
      const errorMsg: Message = {
        id: (Date.now() + 2).toString(),
        role: 'system',
        content: errorMessage,
        timestamp: new Date(),
        error: errorMessage,
      };
      
      addMessage(errorMsg);
    } finally {
      setLoading(false);
    }
  };

  const handleCommand = async (command: string, threadId: string) => {
    try {
      const response = await chatService.executeCommand(command, threadId);
      
      let commandResult = '';
      if (response.commands) {
        commandResult = '可用命令：\n' + response.commands.map((cmd: any) => 
          `${cmd.cmd}: ${cmd.desc}`
        ).join('\n');
      } else if (response.history) {
        commandResult = '历史消息：\n' + response.history.map((msg: Message) => 
          `${msg.role}: ${msg.content}`
        ).join('\n');
      } else if (response.reset) {
        commandResult = '会话已重置';
      }

      const systemMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: 'system',
        content: commandResult,
        timestamp: new Date(),
      };

      addMessage(systemMessage);
    } catch (error) {
      console.error('Command execution failed:', error);
      throw error;
    }
  };

  const startSuggestionStream = (threadId: string, messageId: string) => {
    // Clear existing timeout
    if (suggestionTimeoutRef.current) {
      clearTimeout(suggestionTimeoutRef.current);
    }

    // Start suggestion stream after a short delay
    suggestionTimeoutRef.current = setTimeout(() => {
      chatService.startSuggestionStream(threadId, (event) => {
        if (event.event === 'react' && event.suggestions) {
          updateMessage(messageId, { suggestions: event.suggestions });
        }
      });
    }, 500);
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  const handleAudioRecorded = (audioData: string, duration: number) => {
    setShowAudioRecorder(false);
    
    if (!threadId) {
      message.error('会话初始化失败，请刷新页面重试');
      return;
    }

    const audioMessage: Message = {
      id: Date.now().toString(),
      role: 'user',
      content: `[语音消息 - 时长: ${Math.round(duration)}秒]`,
      timestamp: new Date(),
      audio: audioData,
    };

    addMessage(audioMessage);
    setLoading(true);

    try {
      // Send audio message to backend
      chatService.sendMessage(
        `[语音消息]`,
        threadId,
        undefined,
        audioData
      ).then(response => {
        const assistantMessage: Message = {
          id: (Date.now() + 1).toString(),
          role: 'assistant',
          content: response.answer,
          timestamp: new Date(),
          sources: response.sources,
          route: response.route,
        };

        addMessage(assistantMessage);

        // Start suggestion stream
        if (response.suggestions && response.suggestions.length > 0) {
          updateMessage(assistantMessage.id, { suggestions: response.suggestions });
        } else {
          startSuggestionStream(threadId, assistantMessage.id);
        }
      }).catch(error => {
        console.error('Failed to send audio message:', error);
        
        const errorMsg: Message = {
          id: (Date.now() + 2).toString(),
          role: 'system',
          content: '语音消息发送失败，请重试',
          timestamp: new Date(),
          error: 'Audio message failed',
        };
        
        addMessage(errorMsg);
      }).finally(() => {
        setLoading(false);
      });
    } catch (error) {
      console.error('Audio message error:', error);
      setLoading(false);
      
      const errorMsg: Message = {
        id: (Date.now() + 2).toString(),
        role: 'system',
        content: '语音消息处理失败，请重试',
        timestamp: new Date(),
        error: 'Audio processing failed',
      };
      
      addMessage(errorMsg);
    }
  };

  const handleImagesUploaded = (images: string[]) => {
    setUploadedImages(images);
    setShowImageUploader(false);
    message.success(`已上传 ${images.length} 张图片`);
  };

  const handleSuggestionClick = (suggestion: string) => {
    setInputValue(suggestion);
    // Focus input after setting value
    setTimeout(() => {
      const textarea = document.querySelector('.chat-textarea textarea') as HTMLTextAreaElement;
      if (textarea) {
        textarea.focus();
      }
    }, 0);
  };

  return (
    <div className={`chat-interface ${className}`}>
      <div className="chat-messages">
        {!currentSession && (
          <div style={{ padding: 12 }}>
            请输入消息开始对话，支持图片与语音
          </div>
        )}
        {currentSession && (
          <MessageList 
            messages={currentSession.messages} 
            onSuggestionClick={handleSuggestionClick}
          />
        )}
        <div ref={messagesEndRef} />
      </div>

      <div className="chat-input-container">
        {uploadedImages.length > 0 && (
          <div className="uploaded-images">
            {uploadedImages.map((image, index) => (
              <div key={index} className="uploaded-image">
                <img src={image} alt={`上传图片 ${index + 1}`} />
                <button
                  className="remove-image"
                  onClick={() => setUploadedImages(uploadedImages.filter((_, i) => i !== index))}
                >
                  ×
                </button>
              </div>
            ))}
          </div>
        )}

        <div className="chat-input-wrapper">
          <div className="input-actions">
            <Button
              type="text"
              icon={<AudioOutlined />}
              onClick={() => setShowAudioRecorder(!showAudioRecorder)}
              className={showAudioRecorder ? 'active' : ''}
              disabled={isLoading}
            />
            <Button
              type="text"
              icon={<PictureOutlined />}
              onClick={() => setShowImageUploader(!showImageUploader)}
              className={showImageUploader ? 'active' : ''}
              disabled={isLoading}
            />
          </div>

          <TextArea
            value={inputValue}
            onChange={(e) => setInputValue(e.target.value)}
            onKeyPress={handleKeyPress}
            placeholder="输入消息，按Enter发送，Shift+Enter换行..."
            autoSize={{ minRows: 1, maxRows: 6 }}
            disabled={isLoading}
            className="chat-textarea"
          />

          <Button
            type="primary"
            icon={isLoading ? <LoadingOutlined /> : <SendOutlined />}
            onClick={handleSendMessage}
            disabled={isLoading || (!inputValue.trim() && uploadedImages.length === 0)}
            className="send-button"
          />
        </div>

        {showAudioRecorder && (
          <AudioRecorder
            onRecord={handleAudioRecorded}
            onCancel={() => setShowAudioRecorder(false)}
          />
        )}

        {showImageUploader && (
          <ImageUploader
            onUpload={handleImagesUploaded}
            onCancel={() => setShowImageUploader(false)}
          />
        )}
      </div>
    </div>
  );
};

export default ChatInterface;