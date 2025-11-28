import React from 'react';
import { Avatar, Space, Typography, Tag } from 'antd';
import { UserOutlined, RobotOutlined } from '@ant-design/icons';
import ReactMarkdown from 'react-markdown';
import type { Message } from '../../types';
import './MessageList.css';

const { Text } = Typography;

interface MessageListProps {
  messages: Message[];
  onSuggestionClick?: (suggestion: string) => void;
}

const MessageList: React.FC<MessageListProps> = ({ messages, onSuggestionClick }) => {
  const formatTime = (date: Date) => {
    return new Intl.DateTimeFormat('zh-CN', {
      hour: '2-digit',
      minute: '2-digit',
    }).format(date);
  };

  const renderContent = (content: string) => {
    return (
      <ReactMarkdown
        components={{
          p: ({ children }) => <p style={{ margin: '0 0 8px 0' }}>{children}</p>,
          ul: ({ children }) => <ul style={{ margin: '0 0 8px 0', paddingLeft: '20px' }}>{children}</ul>,
          ol: ({ children }) => <ol style={{ margin: '0 0 8px 0', paddingLeft: '20px' }}>{children}</ol>,
          li: ({ children }) => <li style={{ margin: '2px 0' }}>{children}</li>,
          code: ({ children }) => (
            <code style={{ 
              backgroundColor: '#f0f0f0', 
              padding: '2px 4px', 
              borderRadius: '3px',
              fontSize: '0.9em'
            }}>
              {children}
            </code>
          ),
          pre: ({ children }) => (
            <pre style={{ 
              backgroundColor: '#f5f5f5', 
              padding: '12px', 
              borderRadius: '6px',
              overflowX: 'auto',
              margin: '8px 0'
            }}>
              {children}
            </pre>
          ),
        }}
      >
        {content}
      </ReactMarkdown>
    );
  };

  const renderAudio = (audio?: string) => {
    if (!audio) return null;

    return (
      <div className="message-audio">
        <audio controls className="audio-player">
          <source src={audio} type="audio/webm" />
          您的浏览器不支持音频播放。
        </audio>
      </div>
    );
  };

  const renderImages = (images?: string[]) => {
    if (!images || images.length === 0) return null;

    return (
      <div className="message-images">
        {images.map((image, index) => (
          <img
            key={index}
            src={image}
            alt={`用户图片 ${index + 1}`}
            className="message-image"
            onClick={() => window.open(image, '_blank')}
          />
        ))}
      </div>
    );
  };

  const renderSources = (sources?: Message['sources']) => {
    if (!sources || sources.length === 0) return null;

    return (
      <div className="message-sources">
        <Text type="secondary" style={{ fontSize: '12px', marginBottom: '4px' }}>
          参考来源：
        </Text>
        {sources.map((source, index) => (
          <div key={index} className="message-source">
            <Tag color="blue" style={{ marginBottom: '4px' }}>
              {source.title}
            </Tag>
            <Text type="secondary" style={{ fontSize: '12px' }}>
              {source.content}
            </Text>
          </div>
        ))}
      </div>
    );
  };

  const renderSuggestions = (suggestions?: string[]) => {
    if (!suggestions || suggestions.length === 0) return null;

    return (
      <div className="message-suggestions">
        <Text type="secondary" style={{ fontSize: '12px', marginBottom: '4px' }}>
          建议追问：
        </Text>
        <Space size="small" wrap>
          {suggestions.map((suggestion, index) => (
            <Tag
              key={index}
              color="green"
              style={{ cursor: 'pointer' }}
              onClick={() => onSuggestionClick?.(suggestion)}
            >
              {suggestion}
            </Tag>
          ))}
        </Space>
      </div>
    );
  };

  return (
    <div className="message-list">
      {messages.map((message) => (
        <div
          key={message.id}
          className={`message-item ${message.role === 'user' ? 'user-message' : message.role === 'system' ? 'system-message' : 'assistant-message'}`}
        >
          <div className="message-avatar">
            <Avatar
              icon={message.role === 'user' ? <UserOutlined /> : message.role === 'system' ? null : <RobotOutlined />}
              style={{
                backgroundColor: message.role === 'user' ? '#1890ff' : message.role === 'system' ? '#faad14' : '#52c41a',
              }}
            >
              {message.role === 'system' && '!'}
            </Avatar>
          </div>

          <div className="message-content-wrapper">
            <div className="message-header">
              <Text type="secondary" style={{ fontSize: '12px' }}>
                {formatTime(message.timestamp)}
              </Text>
              {message.route && (
                <Tag color="blue" style={{ marginLeft: '8px' }}>
                  {message.route}
                </Tag>
              )}
            </div>

            <div className="message-content">
              {renderContent(message.content)}
              {renderAudio(message.audio)}
              {renderImages(message.images)}
              {renderSources(message.sources)}
              {renderSuggestions(message.suggestions)}
            </div>

            {message.error && (
              <div className="message-error">
                <Text type="danger" style={{ fontSize: '12px' }}>
                  {message.error}
                </Text>
              </div>
            )}
          </div>
        </div>
      ))}
    </div>
  );
};

export default MessageList;