import React, { useState } from 'react';
import { Card, Input, Button, Typography, Space, Divider, List, Tag, App as AntdApp } from 'antd';
import { SendOutlined, PlusOutlined, DeleteOutlined } from '@ant-design/icons';
import { v4 as uuidv4 } from 'uuid';
import { apiService } from '../../services';
import type { ChatResponse } from '../../types';

const { TextArea } = Input;
const { Title, Text } = Typography;

const KnowledgeSearch: React.FC = () => {
  const { message } = AntdApp.useApp();
  const [query, setQuery] = useState('');
  const [answer, setAnswer] = useState<ChatResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [importText, setImportText] = useState('');
  const [importMetadata, setImportMetadata] = useState('');
  const [deleteIds, setDeleteIds] = useState('');

  const handleSearch = async () => {
    console.log('[KB] 点击查询', { query });
    if (!query.trim()) {
      message.warning('请输入查询内容');
      return;
    }
    setLoading(true);
    try {
      const threadId = uuidv4();
      console.log('[KB] 发送查询请求', { threadId, query: query.trim() });
      const res = await apiService.sendMessage(query.trim(), threadId);
      console.log('[KB] 查询响应', { code: res.code, route: res.data?.route });
      if (res.code !== 0) {
        throw new Error(res.message || '查询失败');
      }
      setAnswer(res.data);
    } catch (e: any) {
      console.log('[KB] 查询错误', e);
      message.error(e?.message || '查询失败');
    } finally {
      console.log('[KB] 查询完成');
      setLoading(false);
    }
  };

  const handleImport = async () => {
    console.log('[KB] 点击导入');
    const lines = importText
      .split('\n')
      .map(s => s.trim())
      .filter(Boolean);
    console.log('[KB] 待导入文本行数', { count: lines.length });
    if (lines.length === 0) {
      message.warning('请输入待导入文本，每行一条');
      return;
    }
    let metadata: Record<string, any> | undefined = undefined;
    if (importMetadata.trim()) {
      try {
        metadata = JSON.parse(importMetadata.trim());
      } catch (e) {
        message.error('Metadata 需为合法 JSON');
        return;
      }
    }
    try {
      const items = lines.map(text => ({ text, metadata }));
      console.log('[KB] 发送导入请求', { itemsCount: items.length, hasMetadata: !!metadata });
      const res = await apiService.addVectors(items);
      console.log('[KB] 导入响应', { code: res.code });
      if (res.code !== 0) {
        throw new Error(res.message || '导入失败');
      }
      message.success('导入成功');
      setImportText('');
      setImportMetadata('');
    } catch (e: any) {
      console.log('[KB] 导入错误', e);
      message.error(e?.message || '导入失败');
    }
  };

  const handleDelete = async () => {
    console.log('[KB] 点击删除');
    const ids = deleteIds
      .split(',')
      .map(s => s.trim())
      .filter(Boolean);
    console.log('[KB] 待删除 ID', { ids });
    if (ids.length === 0) {
      message.warning('请输入待删除的 ID，逗号分隔');
      return;
    }
    try {
      console.log('[KB] 发送删除请求', { count: ids.length });
      const res = await apiService.deleteVectors(ids);
      console.log('[KB] 删除响应', { code: res.code });
      if (res.code !== 0) {
        throw new Error(res.message || '删除失败');
      }
      message.success('删除成功');
      setDeleteIds('');
    } catch (e: any) {
      console.log('[KB] 删除错误', e);
      message.error(e?.message || '删除失败');
    }
  };

  return (
    <Space direction="vertical" style={{ width: '100%' }} size="large">
      <Card>
        <Title level={4}>知识库查询</Title>
        <Space.Compact style={{ width: '100%' }}>
          <Input
            value={query}
            onChange={e => setQuery(e.target.value)}
            placeholder="输入查询内容"
            disabled={loading}
          />
          <Button type="primary" icon={<SendOutlined />} onClick={handleSearch} loading={loading}>
            查询
          </Button>
        </Space.Compact>
        {answer && (
          <div style={{ marginTop: 16 }}>
            <Space align="center" style={{ marginBottom: 8 }}>
              {answer.route && <Tag color="blue">{answer.route}</Tag>}
              <Text type="secondary">答案与来源</Text>
            </Space>
            <Card size="small" style={{ marginBottom: 12 }}>
              <Text>{answer.answer}</Text>
            </Card>
            {answer.sources && answer.sources.length > 0 && (
              <List
                header={<Text type="secondary">参考来源</Text>}
                bordered
                dataSource={answer.sources}
                renderItem={(s) => (
                  <List.Item>
                    <Space direction="vertical" size={4} style={{ width: '100%' }}>
                      <Space align="center">
                        <Tag color="geekblue">{s.title}</Tag>
                        {s.url && (
                          <a href={s.url} target="_blank" rel="noreferrer">链接</a>
                        )}
                      </Space>
                      <Text type="secondary">{s.content}</Text>
                    </Space>
                  </List.Item>
                )}
              />
            )}
          </div>
        )}
      </Card>

      <Card>
        <Title level={4}>知识库管理</Title>
        <Text type="secondary">批量导入：每行一条文本，可选 Metadata JSON</Text>
        <Divider />
        <Space direction="vertical" style={{ width: '100%' }}>
          <TextArea
            rows={6}
            value={importText}
            onChange={e => setImportText(e.target.value)}
            placeholder="每行一条文本"
          />
          <TextArea
            rows={4}
            value={importMetadata}
            onChange={e => setImportMetadata(e.target.value)}
            placeholder="可选 Metadata（JSON）"
          />
          <Button type="primary" icon={<PlusOutlined />} onClick={handleImport}>导入</Button>
        </Space>
        <Divider />
        <Text type="secondary">按 ID 删除：多个 ID 使用逗号分隔</Text>
        <Space.Compact style={{ width: '100%', marginTop: 8 }}>
          <Input
            value={deleteIds}
            onChange={e => setDeleteIds(e.target.value)}
            placeholder="id1,id2,id3"
          />
          <Button danger icon={<DeleteOutlined />} onClick={handleDelete}>删除</Button>
        </Space.Compact>
      </Card>
    </Space>
  );
};

export default KnowledgeSearch;