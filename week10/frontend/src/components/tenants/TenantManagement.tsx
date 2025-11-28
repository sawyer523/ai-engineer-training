import React, { useEffect, useState } from 'react';
import { Card, Input, Button, Typography, Space, Divider, List, Tag, App as AntdApp } from 'antd';
import { PlusOutlined, CheckOutlined, DeleteOutlined, ReloadOutlined } from '@ant-design/icons';
import { apiService } from '../../services';

const { Title, Text } = Typography;

interface Tenant {
  tenantId: string;
  apiKey?: string;
  note?: string;
}

const LS_KEY = 'tenants';

const TenantManagement: React.FC = () => {
  const { message } = AntdApp.useApp();
  const [tenants, setTenants] = useState<Tenant[]>([]);
  const [tenantId, setTenantId] = useState('');
  const [apiKey, setApiKey] = useState('');
  const [note, setNote] = useState('');
  const [health, setHealth] = useState<any>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    const raw = localStorage.getItem(LS_KEY);
    if (raw) {
      try {
        const data = JSON.parse(raw);
        if (Array.isArray(data)) setTenants(data);
      } catch {}
    }
  }, []);

  const saveTenants = (list: Tenant[]) => {
    setTenants(list);
    localStorage.setItem(LS_KEY, JSON.stringify(list));
  };

  const handleAdd = () => {
    console.log('[Tenants] 点击保存/新增', { tenantId, hasApiKey: !!apiKey, hasNote: !!note });
    if (!tenantId.trim()) {
      message.warning('请输入租户 ID');
      return;
    }
    const list = [...tenants];
    const idx = list.findIndex(t => t.tenantId === tenantId.trim());
    const item = { tenantId: tenantId.trim(), apiKey: apiKey.trim() || undefined, note: note.trim() || undefined };
    if (idx >= 0) list[idx] = item; else list.unshift(item);
    saveTenants(list);
    message.success('已保存租户');
    setTenantId('');
    setApiKey('');
    setNote('');
  };

  const handleSetCurrent = (t: Tenant) => {
    console.log('[Tenants] 设为当前租户', { tenantId: t.tenantId, hasApiKey: !!t.apiKey });
    localStorage.setItem('tenantId', t.tenantId);
    if (t.apiKey) localStorage.setItem('apiKey', t.apiKey); else localStorage.removeItem('apiKey');
    message.success(`已切换到租户 ${t.tenantId}`);
  };

  const handleDelete = (t: Tenant) => {
    console.log('[Tenants] 删除租户', { tenantId: t.tenantId });
    const list = tenants.filter(x => x.tenantId !== t.tenantId);
    saveTenants(list);
    message.success('已删除租户');
  };

  const handleHealth = async () => {
    console.log('[Tenants] 点击健康检查');
    setLoading(true);
    try {
      console.log('[Tenants] 发送健康检查请求');
      const res = await apiService.getHealth();
      console.log('[Tenants] 健康检查响应', { model: res.model, kb_index: res.kb_index, orders_db: res.orders_db });
      setHealth(res);
      message.success('健康检查成功');
    } catch (e: any) {
      console.log('[Tenants] 健康检查错误', e);
      message.error(e?.message || '健康检查失败');
    } finally {
      console.log('[Tenants] 健康检查完成');
      setLoading(false);
    }
  };

  const currentTenant = localStorage.getItem('tenantId') || 'default';

  return (
    <Space direction="vertical" style={{ width: '100%' }} size="large">
      <Card>
        <Title level={4}>当前租户</Title>
        <Space align="center">
          <Tag color="blue">{currentTenant}</Tag>
          <Button icon={<ReloadOutlined />} onClick={handleHealth} loading={loading}>健康检查</Button>
        </Space>
        {health && (
          <div style={{ marginTop: 12 }}>
            <Space direction="vertical" size={4}>
              <Text type="secondary">模型：{health.model}</Text>
              <Text type="secondary">知识库：{health.kb_index ? '可用' : '不可用'}</Text>
              <Text type="secondary">订单库：{health.orders_db ? '可用' : '不可用'}</Text>
            </Space>
          </div>
        )}
      </Card>

      <Card>
        <Title level={4}>维护租户</Title>
        <Space direction="vertical" style={{ width: '100%' }}>
          <Input value={tenantId} onChange={e => setTenantId(e.target.value)} placeholder="租户 ID" />
          <Input value={apiKey} onChange={e => setApiKey(e.target.value)} placeholder="API Key（可选）" />
          <Input value={note} onChange={e => setNote(e.target.value)} placeholder="备注（可选）" />
          <Button type="primary" icon={<PlusOutlined />} onClick={handleAdd}>保存/新增</Button>
        </Space>
        <Divider />
        <List
          header={<Text type="secondary">租户列表</Text>}
          bordered
          dataSource={tenants}
          renderItem={(t) => (
            <List.Item
              actions={[
                <Button key="set" type="link" icon={<CheckOutlined />} onClick={() => handleSetCurrent(t)}>设为当前</Button>,
                <Button key="del" type="link" danger icon={<DeleteOutlined />} onClick={() => handleDelete(t)}>删除</Button>,
              ]}
            >
              <Space direction="vertical" size={2} style={{ width: '100%' }}>
                <Space>
                  <Tag color="blue">{t.tenantId}</Tag>
                  {t.apiKey ? <Tag color="green">有 API Key</Tag> : <Tag>无 API Key</Tag>}
                </Space>
                {t.note && <Text type="secondary">{t.note}</Text>}
              </Space>
            </List.Item>
          )}
        />
      </Card>
    </Space>
  );
};

export default TenantManagement;