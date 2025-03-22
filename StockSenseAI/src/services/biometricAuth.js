import * as LocalAuthentication from 'expo-local-authentication';
import AsyncStorage from '@react-native-async-storage/async-storage';

const BIOMETRIC_ENABLED_KEY = 'biometric_auth_enabled';

export const biometricAuthService = {
  // Check if biometric authentication is available
  isBiometricAvailable: async () => {
    try {
      const compatible = await LocalAuthentication.hasHardwareAsync();
      if (!compatible) {
        return false;
      }
      
      const enrolled = await LocalAuthentication.isEnrolledAsync();
      return enrolled;
    } catch (error) {
      console.error('Error checking biometric availability:', error);
      return false;
    }
  },

  // Get available biometric types
  getBiometricTypes: async () => {
    try {
      const types = await LocalAuthentication.supportedAuthenticationTypesAsync();
      return types.map(type => {
        switch (type) {
          case LocalAuthentication.AuthenticationType.FINGERPRINT:
            return 'fingerprint';
          case LocalAuthentication.AuthenticationType.FACIAL_RECOGNITION:
            return 'facial';
          case LocalAuthentication.AuthenticationType.IRIS:
            return 'iris';
          default:
            return 'unknown';
        }
      });
    } catch (error) {
      console.error('Error getting biometric types:', error);
      return [];
    }
  },

  // Check if biometric authentication is enabled
  isBiometricEnabled: async () => {
    try {
      const enabled = await AsyncStorage.getItem(BIOMETRIC_ENABLED_KEY);
      return enabled === 'true';
    } catch (error) {
      console.error('Error checking biometric status:', error);
      return false;
    }
  },

  // Enable biometric authentication
  enableBiometric: async () => {
    try {
      const available = await biometricAuthService.isBiometricAvailable();
      if (!available) {
        return { success: false, error: 'Biometric authentication is not available on this device' };
      }

      // First, authenticate to ensure the user can use biometrics
      const result = await LocalAuthentication.authenticateAsync({
        promptMessage: 'Authenticate to enable biometric login',
        cancelLabel: 'Cancel',
        disableDeviceFallback: true,
      });

      if (result.success) {
        await AsyncStorage.setItem(BIOMETRIC_ENABLED_KEY, 'true');
        return { success: true };
      } else {
        return { 
          success: false, 
          error: result.error === 'user_cancel' ? 'Authentication canceled' : 'Authentication failed' 
        };
      }
    } catch (error) {
      console.error('Error enabling biometric:', error);
      return { success: false, error: error.message };
    }
  },

  // Disable biometric authentication
  disableBiometric: async () => {
    try {
      await AsyncStorage.setItem(BIOMETRIC_ENABLED_KEY, 'false');
      return { success: true };
    } catch (error) {
      console.error('Error disabling biometric:', error);
      return { success: false, error: error.message };
    }
  },

  // Authenticate with biometric
  authenticate: async () => {
    try {
      const enabled = await biometricAuthService.isBiometricEnabled();
      if (!enabled) {
        return { success: false, error: 'Biometric authentication is not enabled' };
      }

      const result = await LocalAuthentication.authenticateAsync({
        promptMessage: 'Authenticate to continue',
        cancelLabel: 'Cancel',
        disableDeviceFallback: true,
      });

      return { 
        success: result.success, 
        error: result.success ? null : 
               result.error === 'user_cancel' ? 'Authentication canceled' : 'Authentication failed'
      };
    } catch (error) {
      console.error('Error during biometric authentication:', error);
      return { success: false, error: error.message };
    }
  }
}; 