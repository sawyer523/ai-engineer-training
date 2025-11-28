import { ConfigProvider, App as AntdApp } from 'antd';
import zhCN from 'antd/locale/zh_CN';
import MainLayout from './components/layout/MainLayout';
import './App.css';

function App() {
  return (
    <ConfigProvider
      locale={zhCN}
      theme={{
        token: {
          colorPrimary: '#1890ff',
          borderRadius: 6,
          wireframe: false,
        },
        components: {
          Layout: {
            headerBg: '#fff',
            headerHeight: 64,
            siderBg: '#001529',
          },
          Menu: {
            darkItemBg: '#001529',
            darkSubMenuItemBg: '#000c17',
            darkItemSelectedBg: '#1890ff',
          },
        },
      }}
    >
      <AntdApp>
        <div className="App">
          <MainLayout />
        </div>
      </AntdApp>
    </ConfigProvider>
  );
}

export default App;
