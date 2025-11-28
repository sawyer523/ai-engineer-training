import React, { useState } from 'react';
import { Card, Input, Button, Typography, Space, Tag, App as AntdApp } from 'antd';
import { SearchOutlined } from '@ant-design/icons';
import { apiService } from '../../services';

const { Title, Text } = Typography;

const OrdersSearch: React.FC = () => {
  const { message } = AntdApp.useApp();
  const [orderId, setOrderId] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<any>(null);

  const handleSearch = async () => {
    console.log('[Orders] 点击查询', { orderId });
    const id = orderId.trim();
    if (!id) {
      message.warning('请输入订单号');
      return;
    }
    setLoading(true);
    try {
      console.log('[Orders] 发送查询请求', { orderId: id });
      const res = await apiService.getOrder(id);
      console.log('[Orders] 查询响应', { code: res.code });
      if (res.code !== 0) {
        throw new Error(res.message || '查询失败');
      }
      setResult(res.data);
    } catch (e: any) {
      console.log('[Orders] 查询错误', e);
      message.error(e?.message || '查询失败');
      setResult(null);
    } finally {
      console.log('[Orders] 查询完成');
      setLoading(false);
    }
  };

  return (
    <Space direction="vertical" style={{ width: '100%' }} size="large">
      <Card>
        <Title level={4}>订单查询</Title>
        <Space.Compact style={{ width: '100%', marginTop: 8 }}>
          <Input
            placeholder="输入订单号"
            value={orderId}
            onChange={e => setOrderId(e.target.value)}
            disabled={loading}
          />
          <Button type="primary" icon={<SearchOutlined />} onClick={handleSearch} loading={loading}>
            查询
          </Button>
        </Space.Compact>
        {!result && (
          <Text type="secondary" style={{ display: 'block', marginTop: 12 }}>
            支持按订单号查询；若数据库未配置或无此订单，将返回错误提示。
          </Text>
        )}
      </Card>

      {result && (
        <Card>
          <Space direction="vertical" size={8} style={{ width: '100%' }}>
            <Space align="center">
              <Tag color="blue">订单号</Tag>
              <Text strong>{result.order_id}</Text>
            </Space>
            <Space align="center">
              <Tag color="green">状态</Tag>
              <Text>{result.status}</Text>
            </Space>
            <Space align="center">
              <Tag color="purple">金额</Tag>
              <Text>{result.amount ?? '-'}</Text>
            </Space>
            <Space align="center">
              <Tag color="geekblue">更新时间</Tag>
              <Text>{result.updated_at ?? '-'}</Text>
            </Space>
            <Space align="center">
              <Tag>开始时间</Tag>
              <Text>{result.start_time ?? '-'}</Text>
            </Space>
            <Space align="center">
              <Tag>报名时间</Tag>
              <Text>{result.enroll_time ?? '-'}</Text>
            </Space>
          </Space>
        </Card>
      )}
    </Space>
  );
};

export default OrdersSearch;