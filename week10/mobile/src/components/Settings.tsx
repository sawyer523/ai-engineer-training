import React, { useEffect, useState } from 'react';
import { View, Text, TextInput, TouchableOpacity, StyleSheet } from 'react-native';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { useSystemStore } from '../stores';

export default function Settings({ navigation }: any) {
  const { config, setConfig } = useSystemStore();
  const [tenantId, setTenantId] = useState('');
  const [apiKey, setApiKey] = useState('');

  useEffect(() => {
    loadValues();
  }, []);

  const loadValues = async () => {
    const t = (await AsyncStorage.getItem('tenantId')) || config?.tenantId || 'default';
    const k = (await AsyncStorage.getItem('apiKey')) || '';
    setTenantId(t);
    setApiKey(k);
  };

  const save = async () => {
    await AsyncStorage.setItem('tenantId', tenantId || 'default');
    await AsyncStorage.setItem('apiKey', apiKey || '');
    const next = {
      currentModel: config?.currentModel || '',
      supportedModels: config?.supportedModels || [],
      tenantId: tenantId || 'default',
      kbIndexAvailable: !!config?.kbIndexAvailable,
      ordersDbAvailable: !!config?.ordersDbAvailable,
    };
    setConfig(next);
    navigation.goBack();
  };

  return (
    <View style={styles.container}>
      <Text style={styles.title}>系统设置</Text>
      <View style={styles.field}>
        <Text style={styles.label}>租户ID</Text>
        <TextInput
          style={styles.input}
          value={tenantId}
          onChangeText={setTenantId}
          placeholder="例如 t1"
          autoCapitalize="none"
        />
      </View>
      <View style={styles.field}>
        <Text style={styles.label}>API Key</Text>
        <TextInput
          style={styles.input}
          value={apiKey}
          onChangeText={setApiKey}
          placeholder="例如 test"
          autoCapitalize="none"
        />
      </View>
      <TouchableOpacity style={styles.saveButton} onPress={save}>
        <Text style={styles.saveText}>保存</Text>
      </TouchableOpacity>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#fff', paddingHorizontal: 20, paddingTop: 60 },
  title: { fontSize: 22, fontWeight: '600', marginBottom: 20 },
  field: { marginBottom: 16 },
  label: { fontSize: 14, color: '#666', marginBottom: 6 },
  input: { borderWidth: 1, borderColor: '#e0e0e0', borderRadius: 8, paddingHorizontal: 12, paddingVertical: 10 },
  saveButton: { backgroundColor: '#007AFF', borderRadius: 8, paddingVertical: 12, alignItems: 'center', marginTop: 8 },
  saveText: { color: '#fff', fontSize: 16, fontWeight: '600' },
});