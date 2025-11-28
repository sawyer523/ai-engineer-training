import React, { useState, useRef, useEffect } from 'react';
import {
  View,
  Text,
  TextInput,
  TouchableOpacity,
  FlatList,
  StyleSheet,
  Alert,
  KeyboardAvoidingView,
  Platform,
  ActivityIndicator,
} from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { useChatStore } from '../stores';
import { chatService } from '../services';
import type { Message } from '../types';

interface ChatInterfaceProps {
  className?: string;
}

const ChatInterface: React.FC<ChatInterfaceProps> = () => {
  const [inputValue, setInputValue] = useState('');
  const [uploadedImages, setUploadedImages] = useState<string[]>([]);
  
  const {
    currentSession,
    isLoading,
    threadId,
    addMessage,
    updateMessage,
    setLoading,
    createSession,
  } = useChatStore();

  const flatListRef = useRef<FlatList>(null);

  useEffect(() => {
    // Create initial session if none exists
    if (!currentSession) {
      createSession('AI助手对话');
    }
  }, [currentSession, createSession]);

  useEffect(() => {
    // Scroll to bottom when messages change
    if (currentSession?.messages.length) {
      setTimeout(() => {
        flatListRef.current?.scrollToEnd({ animated: true });
      }, 100);
    }
  }, [currentSession?.messages]);

  const handleSendMessage = async () => {
    if (!inputValue.trim() && uploadedImages.length === 0) {
      Alert.alert('提示', '请输入消息或上传图片');
      return;
    }

    if (!threadId) {
      Alert.alert('错误', '会话初始化失败，请重试');
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
        content:
          typeof response.answer === 'string' && response.answer.trim()
            ? response.answer
            : (response.error && `错误：${response.error}`) || '暂未获取到答案，请稍后重试',
        timestamp: new Date(),
        sources: response.sources,
        route: response.route,
      };

      addMessage(assistantMessage);

      // Start suggestion stream
      if (response.suggestions && response.suggestions.length > 0) {
        updateMessage(assistantMessage.id, { suggestions: response.suggestions });
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

  const handleSuggestionClick = (suggestion: string) => {
    setInputValue(suggestion);
  };

  const renderMessage = ({ item: message }: { item: Message }) => {
    const isUser = message.role === 'user';
    const isSystem = message.role === 'system';
    const displayContent = typeof message.content === 'string' ? message.content : '';

    return (
      <View style={[
        styles.messageContainer,
        isUser ? styles.userMessage : isSystem ? styles.systemMessage : styles.assistantMessage
      ]}>
        <View style={[
          styles.messageBubble,
          isUser ? styles.userBubble : isSystem ? styles.systemBubble : styles.assistantBubble
        ]}>
          <Text style={[
            styles.messageText,
            isUser ? styles.userText : isSystem ? styles.systemText : styles.assistantText
          ]}>
            {displayContent}
          </Text>
          
          {message.images && message.images.length > 0 && (
            <View style={styles.messageImages}>
              {message.images.map((image, index) => (
                <Text key={index} style={styles.imageText}>
                  📷 图片 {index + 1}
                </Text>
              ))}
            </View>
          )}

          {message.suggestions && message.suggestions.length > 0 && (
            <View style={styles.suggestionsContainer}>
              {message.suggestions.map((suggestion, index) => (
                <TouchableOpacity
                  key={index}
                  style={styles.suggestionButton}
                  onPress={() => handleSuggestionClick(suggestion)}
                >
                  <Text style={styles.suggestionText}>{suggestion}</Text>
                </TouchableOpacity>
              ))}
            </View>
          )}

          {message.error && (
            <Text style={styles.errorText}>{message.error}</Text>
          )}

          <Text style={styles.timestamp}>
            {new Date(message.timestamp).toLocaleTimeString('zh-CN', {
              hour: '2-digit',
              minute: '2-digit',
            })}
          </Text>
        </View>
      </View>
    );
  };

  return (
    <KeyboardAvoidingView
      style={styles.container}
      behavior={Platform.OS === 'ios' ? 'padding' : 'height'}
      keyboardVerticalOffset={Platform.OS === 'ios' ? 90 : 0}
    >
      <View style={styles.container}>
        <FlatList
          ref={flatListRef}
          data={currentSession?.messages || []}
          renderItem={renderMessage}
          keyExtractor={(item) => item.id}
          style={styles.messagesList}
          contentContainerStyle={styles.messagesContent}
          onContentSizeChange={() => {
            flatListRef.current?.scrollToEnd({ animated: true });
          }}
        />

        <View style={styles.inputContainer}>
          <View style={styles.inputWrapper}>
            <TextInput
              style={styles.textInput}
              value={inputValue}
              onChangeText={setInputValue}
              placeholder="输入消息..."
              multiline
              maxLength={1000}
              editable={!isLoading}
              onSubmitEditing={handleSendMessage}
              returnKeyType="send"
            />
            
            <View style={styles.inputActions}>
              <TouchableOpacity
                style={styles.actionButton}
                onPress={() => {}}
                disabled={isLoading}
              >
                <Ionicons name="mic" size={24} color={isLoading ? '#ccc' : '#007AFF'} />
              </TouchableOpacity>
              
              <TouchableOpacity
                style={styles.actionButton}
                onPress={() => {}}
                disabled={isLoading}
              >
                <Ionicons name="image" size={24} color={isLoading ? '#ccc' : '#007AFF'} />
              </TouchableOpacity>
            </View>
          </View>

          <TouchableOpacity
            style={[styles.sendButton, isLoading && styles.sendButtonDisabled]}
            onPress={handleSendMessage}
            disabled={isLoading || (!inputValue.trim() && uploadedImages.length === 0)}
          >
            {isLoading ? (
              <ActivityIndicator size="small" color="#fff" />
            ) : (
              <Ionicons name="send" size={20} color="#fff" />
            )}
          </TouchableOpacity>
        </View>
      </View>
    </KeyboardAvoidingView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f5f5f5',
  },
  messagesList: {
    flex: 1,
  },
  messagesContent: {
    paddingVertical: 10,
    paddingHorizontal: 15,
  },
  messageContainer: {
    marginBottom: 10,
  },
  userMessage: {
    alignItems: 'flex-end',
  },
  assistantMessage: {
    alignItems: 'flex-start',
  },
  systemMessage: {
    alignItems: 'center',
  },
  messageBubble: {
    maxWidth: '80%',
    padding: 12,
    borderRadius: 16,
  },
  userBubble: {
    backgroundColor: '#007AFF',
    borderTopRightRadius: 4,
  },
  assistantBubble: {
    backgroundColor: '#fff',
    borderTopLeftRadius: 4,
    borderWidth: 1,
    borderColor: '#e0e0e0',
  },
  systemBubble: {
    backgroundColor: '#fffbe6',
    borderWidth: 1,
    borderColor: '#ffd591',
    borderRadius: 12,
  },
  messageText: {
    fontSize: 16,
    lineHeight: 22,
  },
  userText: {
    color: '#fff',
  },
  assistantText: {
    color: '#333',
  },
  systemText: {
    color: '#ad6800',
    fontSize: 14,
  },
  messageImages: {
    marginTop: 8,
  },
  imageText: {
    color: '#666',
    fontSize: 12,
    marginTop: 2,
  },
  suggestionsContainer: {
    marginTop: 8,
    flexDirection: 'row',
    flexWrap: 'wrap',
  },
  suggestionButton: {
    backgroundColor: '#f0f9ff',
    borderWidth: 1,
    borderColor: '#91d5ff',
    borderRadius: 16,
    paddingHorizontal: 12,
    paddingVertical: 6,
    marginRight: 8,
    marginBottom: 8,
  },
  suggestionText: {
    color: '#1890ff',
    fontSize: 14,
  },
  errorText: {
    color: '#ff4d4f',
    fontSize: 12,
    marginTop: 4,
  },
  timestamp: {
    fontSize: 11,
    color: '#999',
    marginTop: 4,
    textAlign: 'right',
  },
  inputContainer: {
    flexDirection: 'row',
    alignItems: 'flex-end',
    padding: 10,
    backgroundColor: '#fff',
    borderTopWidth: 1,
    borderTopColor: '#e0e0e0',
  },
  inputWrapper: {
    flex: 1,
    flexDirection: 'row',
    alignItems: 'flex-end',
    backgroundColor: '#f5f5f5',
    borderRadius: 20,
    paddingHorizontal: 12,
    paddingVertical: 8,
    marginRight: 8,
  },
  textInput: {
    flex: 1,
    fontSize: 16,
    maxHeight: 100,
    paddingTop: 8,
    paddingBottom: 8,
  },
  inputActions: {
    flexDirection: 'row',
    marginLeft: 8,
  },
  actionButton: {
    padding: 4,
    marginLeft: 8,
  },
  sendButton: {
    width: 40,
    height: 40,
    borderRadius: 20,
    backgroundColor: '#007AFF',
    justifyContent: 'center',
    alignItems: 'center',
  },
  sendButtonDisabled: {
    backgroundColor: '#ccc',
  },
});

export default ChatInterface;