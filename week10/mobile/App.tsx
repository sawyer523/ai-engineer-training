import React, { useEffect } from 'react';
import { NavigationContainer } from '@react-navigation/native';
import { createStackNavigator } from '@react-navigation/stack';
import { View, Text, TouchableOpacity, StyleSheet } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import ChatInterface from './src/components/ChatInterface';
import Settings from './src/components/Settings';
import { useSystemStore } from './src/stores';
import { chatService } from './src/services';
import AsyncStorage from '@react-native-async-storage/async-storage';

const Stack = createStackNavigator();

const HomeScreen = ({ navigation }: any) => {
  const { config, setConfig, setLoading } = useSystemStore();

  useEffect(() => {
    loadSystemConfig();
  }, []);

  const loadSystemConfig = async () => {
    try {
      setLoading(true);
      const envTenant = (process.env.EXPO_PUBLIC_TENANT_ID as string | undefined) || 'default';
      const envKey = (process.env.EXPO_PUBLIC_API_KEY as string | undefined) || '';
      let storedTenant = (await AsyncStorage.getItem('tenantId')) || '';
      let storedKey = (await AsyncStorage.getItem('apiKey')) || '';
      if (!storedTenant) {
        storedTenant = envTenant;
        await AsyncStorage.setItem('tenantId', storedTenant);
      }
      if (!storedKey) {
        storedKey = envKey;
        await AsyncStorage.setItem('apiKey', storedKey);
      }
      const [healthResponse, modelsResponse] = await Promise.all([
        chatService.getHealth(),
        chatService.getModels(),
      ]);

      const systemConfig = {
        currentModel: healthResponse.model,
        supportedModels: modelsResponse.models,
        tenantId: storedTenant,
        kbIndexAvailable: healthResponse.kb_index,
        ordersDbAvailable: healthResponse.orders_db,
      };

      setConfig(systemConfig);
    } catch (error) {
      console.error('Failed to load system config:', error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <View style={styles.container}>
      <View style={styles.header}>
        <Text style={styles.headerTitle}>AI助手</Text>
        <View style={styles.headerActions}>
          <TouchableOpacity style={styles.headerButton} onPress={() => navigation.navigate('Settings')}>
            <Ionicons name="settings-outline" size={24} color="#007AFF" />
          </TouchableOpacity>
        </View>
      </View>
      
      <View style={styles.statusBar}>
        <View style={styles.statusItem}>
          <Ionicons name="hardware-chip-outline" size={16} color="#666" />
          <Text style={styles.statusText}>
            {config?.currentModel || '加载中...'}
          </Text>
        </View>
        
        <View style={styles.statusItem}>
          <Ionicons 
            name={config?.kbIndexAvailable ? "checkmark-circle" : "close-circle"} 
            size={16} 
            color={config?.kbIndexAvailable ? "#52c41a" : "#ff4d4f"} 
          />
          <Text style={styles.statusText}>
            知识库
          </Text>
        </View>

        <View style={styles.statusItem}>
          <Ionicons name="id-card-outline" size={16} color="#666" />
          <Text style={styles.statusText}>
            {config?.tenantId || 'default'}
          </Text>
        </View>
      </View>

      <ChatInterface />
    </View>
  );
};

export default function App() {
  return (
    <NavigationContainer>
      <Stack.Navigator
        screenOptions={{
          headerStyle: {
            backgroundColor: '#fff',
            elevation: 0,
            shadowOpacity: 0,
          },
          headerTintColor: '#007AFF',
          headerTitleStyle: {
            fontWeight: '600',
            fontSize: 18,
          },
        }}
      >
        <Stack.Screen
          name="Home"
          component={HomeScreen}
          options={{
            headerShown: false,
          }}
        />
        <Stack.Screen
          name="Settings"
          component={Settings}
          options={{
            title: '系统设置',
          }}
        />
      </Stack.Navigator>
    </NavigationContainer>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f5f5f5',
  },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingHorizontal: 20,
    paddingTop: 50,
    paddingBottom: 15,
    backgroundColor: '#fff',
    borderBottomWidth: 1,
    borderBottomColor: '#e0e0e0',
  },
  headerTitle: {
    fontSize: 24,
    fontWeight: '600',
    color: '#333',
  },
  headerActions: {
    flexDirection: 'row',
  },
  headerButton: {
    padding: 8,
  },
  statusBar: {
    flexDirection: 'row',
    justifyContent: 'space-around',
    alignItems: 'center',
    paddingVertical: 10,
    backgroundColor: '#fff',
    borderBottomWidth: 1,
    borderBottomColor: '#e0e0e0',
  },
  statusItem: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  statusText: {
    fontSize: 12,
    color: '#666',
    marginLeft: 4,
  },
});
