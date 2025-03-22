import React, { useState, useEffect } from 'react';
import { View, Text, StyleSheet, TouchableOpacity, Alert } from 'react-native';
import * as LocalAuthentication from 'expo-local-authentication';
import { biometricAuthService } from '../services/biometricAuth';

const BiometricTest = () => {
  const [biometricInfo, setBiometricInfo] = useState({
    available: false,
    enrolled: false,
    types: [],
    enabled: false
  });
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    checkBiometricStatus();
  }, []);

  const checkBiometricStatus = async () => {
    try {
      setLoading(true);
      console.log('Checking hardware...');
      const compatible = await LocalAuthentication.hasHardwareAsync();
      console.log('Hardware compatible:', compatible);

      let enrolled = false;
      let types = [];
      if (compatible) {
        enrolled = await LocalAuthentication.isEnrolledAsync();
        console.log('Biometrics enrolled:', enrolled);
        
        const supportedTypes = await LocalAuthentication.supportedAuthenticationTypesAsync();
        types = supportedTypes.map(type => {
          switch (type) {
            case LocalAuthentication.AuthenticationType.FINGERPRINT:
              return 'Fingerprint';
            case LocalAuthentication.AuthenticationType.FACIAL_RECOGNITION:
              return 'Face ID';
            default:
              return 'Unknown';
          }
        });
        console.log('Supported types:', types);
      }

      const enabled = await biometricAuthService.isBiometricEnabled();
      console.log('Biometric enabled in app:', enabled);

      setBiometricInfo({
        available: compatible,
        enrolled,
        types,
        enabled
      });
    } catch (error) {
      console.error('Error checking biometric status:', error);
      Alert.alert('Error', 'Failed to check biometric status: ' + error.message);
    } finally {
      setLoading(false);
    }
  };

  const testAuthentication = async () => {
    try {
      setLoading(true);
      console.log('Starting authentication test...');
      
      const result = await LocalAuthentication.authenticateAsync({
        promptMessage: 'Authenticate for testing',
        cancelLabel: 'Cancel',
        disableDeviceFallback: false,
      });
      
      console.log('Authentication result:', result);
      
      if (result.success) {
        Alert.alert('Success', 'Authentication successful!');
      } else {
        Alert.alert('Failed', result.error ? `Authentication failed: ${result.error}` : 'Authentication was cancelled or failed');
      }
    } catch (error) {
      console.error('Authentication error:', error);
      Alert.alert('Error', 'Authentication error: ' + error.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <View style={styles.container}>
      <Text style={styles.title}>Biometric Authentication Test</Text>
      
      <View style={styles.infoContainer}>
        <Text style={styles.infoText}>Hardware available: <Text style={styles.highlight}>{biometricInfo.available ? 'Yes' : 'No'}</Text></Text>
        <Text style={styles.infoText}>Biometrics enrolled: <Text style={styles.highlight}>{biometricInfo.enrolled ? 'Yes' : 'No'}</Text></Text>
        <Text style={styles.infoText}>Enabled in app: <Text style={styles.highlight}>{biometricInfo.enabled ? 'Yes' : 'No'}</Text></Text>
        <Text style={styles.infoText}>Supported types: <Text style={styles.highlight}>{biometricInfo.types.length > 0 ? biometricInfo.types.join(', ') : 'None'}</Text></Text>
      </View>
      
      <TouchableOpacity 
        style={[styles.button, !biometricInfo.available || !biometricInfo.enrolled ? styles.buttonDisabled : null]} 
        onPress={testAuthentication}
        disabled={!biometricInfo.available || !biometricInfo.enrolled || loading}
      >
        <Text style={styles.buttonText}>Test Authentication</Text>
      </TouchableOpacity>
      
      <TouchableOpacity 
        style={styles.button} 
        onPress={checkBiometricStatus}
        disabled={loading}
      >
        <Text style={styles.buttonText}>Refresh Status</Text>
      </TouchableOpacity>
      
      {loading && <Text style={styles.loadingText}>Loading...</Text>}
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    padding: 20,
    backgroundColor: '#f5f5f5',
    borderRadius: 10,
    margin: 10,
  },
  title: {
    fontSize: 18,
    fontWeight: 'bold',
    marginBottom: 15,
  },
  infoContainer: {
    backgroundColor: '#fff',
    borderRadius: 8,
    padding: 15,
    marginBottom: 20,
  },
  infoText: {
    fontSize: 16,
    marginBottom: 8,
  },
  highlight: {
    fontWeight: 'bold',
  },
  button: {
    backgroundColor: '#3970BE',
    padding: 15,
    borderRadius: 8,
    alignItems: 'center',
    marginBottom: 10,
  },
  buttonDisabled: {
    backgroundColor: '#aaa',
  },
  buttonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '600',
  },
  loadingText: {
    textAlign: 'center',
    marginTop: 10,
    color: '#666',
  }
});

export default BiometricTest; 