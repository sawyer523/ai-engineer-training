import React, { useEffect, useState } from 'react';
import { Layout, Menu, Button, Avatar, Drawer, Dropdown, Typography, Tag, App as AntdApp } from 'antd';
import {
  MessageOutlined,
  DatabaseOutlined,
  SettingOutlined,
  MenuOutlined,
  UserOutlined,
  LogoutOutlined,
} from '@ant-design/icons';
import { useChatStore, useSystemStore } from '../../stores';
import { apiService } from '../../services';
import ChatInterface from '../chat/ChatInterface';
import KnowledgeSearch from '../kb/KnowledgeSearch';
import TenantManagement from '../tenants/TenantManagement';
import OrdersSearch from '../orders/OrdersSearch';
import './MainLayout.css';

const { Header, Sider, Content } = Layout;
const { Title, Text } = Typography;

interface MainLayoutProps {
  children?: React.ReactNode;
}

const MainLayout: React.FC<MainLayoutProps> = () => {
  const { message } = AntdApp.useApp();
  const [collapsed, setCollapsed] = useState(false);
  const [mobileDrawerVisible, setMobileDrawerVisible] = useState(false);
  const [selectedMenu, setSelectedMenu] = useState('chat');
  
  const { config, setConfig, setLoading } = useSystemStore();
  const { createSession } = useChatStore();

  useEffect(() => {
    loadSystemConfig();
  }, []);

  const loadSystemConfig = async () => {
    try {
      setLoading(true);
      const [healthResponse, modelsResponse] = await Promise.all([
        apiService.getHealth(),
        apiService.getModels(),
      ]);

      const systemConfig = {
        currentModel: healthResponse.model,
        supportedModels: modelsResponse.data.models,
        tenantId: localStorage.getItem('tenantId') || 'default',
        kbIndexAvailable: healthResponse.kb_index,
        ordersDbAvailable: healthResponse.orders_db,
      };

      setConfig(systemConfig);
    } catch (error) {
      console.error('Failed to load system config:', error);
      message.error('系统配置加载失败');
    } finally {
      setLoading(false);
    }
  };

  const handleNewChat = () => {
    console.log('[Layout] 点击新对话');
    createSession('新对话');
    message.success('已创建新对话');
  };

  const handleLogout = () => {
    console.log('[Layout] 点击退出登录');
    localStorage.removeItem('apiKey');
    localStorage.removeItem('tenantId');
    window.location.href = '/login';
  };

  const userMenuItems = [
    {
      key: 'profile',
      icon: <UserOutlined />,
      label: '个人资料',
    },
    {
      key: 'logout',
      icon: <LogoutOutlined />,
      label: '退出登录',
      onClick: handleLogout,
    },
  ];

  const menuItems = [
    {
      key: 'chat',
      icon: <MessageOutlined />,
      label: '智能对话',
    },
    {
      key: 'kb',
      icon: <DatabaseOutlined />,
      label: '知识库查询',
    },
    {
      key: 'orders',
      icon: <DatabaseOutlined />,
      label: '订单查询',
    },
    {
      key: 'settings',
      icon: <SettingOutlined />,
      label: '系统设置',
    },
    {
      key: 'tenants',
      icon: <UserOutlined />,
      label: '租户管理',
    },
  ];

  const renderContent = () => {
    switch (selectedMenu) {
      case 'chat':
        return <ChatInterface />;
      case 'kb':
        return <KnowledgeSearch />;
      case 'orders':
        return <OrdersSearch />;
      case 'settings':
        return <div>系统设置功能开发中...</div>;
      case 'tenants':
        return <TenantManagement />;
      default:
        return <ChatInterface />;
    }
  };

  const siderContent = (
    <div className="sider-content">
      <div className="logo-section">
        <div className="logo">
          <Avatar size={32} style={{ backgroundColor: '#1890ff' }}>
            AI
          </Avatar>
          {!collapsed && <Title level={4} style={{ margin: 0, color: '#fff' }}>AI助手</Title>}
        </div>
        <Button
          type="primary"
          onClick={handleNewChat}
          style={{ width: '100%', marginBottom: '16px' }}
        >
          新对话
        </Button>
      </div>
      
      <Menu
        theme="dark"
        mode="inline"
        selectedKeys={[selectedMenu]}
        items={menuItems}
        onClick={({ key }) => {
          console.log('[Layout] 菜单切换', { key });
          setSelectedMenu(key);
          setMobileDrawerVisible(false);
        }}
      />

      <div className="system-info">
        {!collapsed && config && (
          <div className="info-section">
            <Text type="secondary" style={{ fontSize: '12px' }}>
              当前模型: {config.currentModel}
            </Text>
            <Text type="secondary" style={{ fontSize: '12px' }}>
              知识库: {config.kbIndexAvailable ? '可用' : '不可用'}
            </Text>
            <Text type="secondary" style={{ fontSize: '12px' }}>
              订单库: {config.ordersDbAvailable ? '可用' : '不可用'}
            </Text>
          </div>
        )}
      </div>
    </div>
  );

  return (
    <Layout className="main-layout">
      {/* Desktop Sider */}
      <Sider
        trigger={null}
        collapsible
        collapsed={collapsed}
        breakpoint="lg"
        onBreakpoint={(broken) => {
          if (broken) {
            setCollapsed(true);
          }
        }}
        className="desktop-sider"
      >
        {siderContent}
      </Sider>

      {/* Mobile Drawer */}
      <Drawer
        title="菜单"
        placement="left"
        onClose={() => setMobileDrawerVisible(false)}
        open={mobileDrawerVisible}
        className="mobile-drawer"
      >
        {siderContent}
      </Drawer>

      <Layout className="site-layout">
        <Header className="site-layout-header">
          <div className="header-left">
            <Button
              type="text"
              icon={<MenuOutlined />}
              onClick={() => setMobileDrawerVisible(true)}
              className="mobile-menu-button"
              onMouseDown={() => console.log('[Layout] 打开移动端菜单')}
            />
            <Button
              type="text"
              icon={<MenuOutlined />}
              onClick={() => {
                console.log('[Layout] 切换侧边栏折叠', { collapsed: !collapsed });
                setCollapsed(!collapsed);
              }}
              className="desktop-menu-button"
            />
          </div>
          
          <div className="header-title">
            <Text strong style={{ fontSize: '16px' }}>
              {menuItems.find(item => item.key === selectedMenu)?.label || 'AI助手'}
            </Text>
          </div>

          <div className="header-right">
            {config && (
              <Tag color="blue" style={{ marginRight: 8 }}>
                当前租户: {config.tenantId}
              </Tag>
            )}
            <Dropdown
              menu={{ items: userMenuItems }}
              placement="bottomRight"
              arrow
            >
              <Button type="text" icon={<UserOutlined />}>
                用户
              </Button>
            </Dropdown>
          </div>
        </Header>

        <Content className="site-layout-content">
          <div className="content-wrapper">
            {renderContent()}
          </div>
        </Content>
      </Layout>
    </Layout>
  );
};

export default MainLayout;