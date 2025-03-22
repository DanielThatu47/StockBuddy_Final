import React, { useState, useEffect } from 'react';
import { View, Text, StyleSheet, Switch, TouchableOpacity, ScrollView, Alert } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import Header from '../../components/Header';
import Colors from '../../constants/colors';
import { useAuth } from '../../context/AuthContext';
import { biometricAuthService } from '../../services/biometricAuth';

const PrivacySecurityScreen = ({ navigation }) => {
  const { deleteAccount, enableBiometric, disableBiometric, checkBiometricEnabled } = useAuth();
  const [securitySettings, setSecuritySettings] = useState({
    biometricAuth: false,
    twoFactorAuth: false,
    emailVerification: true,
    deviceManagement: true,
    activityLog: true,
    dataSharing: false,
    marketingEmails: false,
  });
  const [isLoading, setIsLoading] = useState(false);

  useEffect(() => {
    checkBiometricStatus();
  }, []);

  const checkBiometricStatus = async () => {
    try {
      console.log('Checking biometric status...');
      // First check hardware availability
      const isAvailable = await biometricAuthService.isBiometricAvailable();
      console.log('Biometric hardware available:', isAvailable);
      
      if (!isAvailable) {
        setSecuritySettings(prev => ({
          ...prev,
          biometricAuth: false
        }));
        return;
      }
      
      // Then check if it's enabled in the app
      const isEnabled = await checkBiometricEnabled();
      console.log('Biometric enabled in app:', isEnabled);
      
      setSecuritySettings(prev => ({
        ...prev,
        biometricAuth: isEnabled
      }));
      
      // Get and log supported biometric types for debugging
      const types = await biometricAuthService.getBiometricTypes();
      console.log('Available biometric types:', types);
    } catch (error) {
      console.error('Error checking biometric status:', error);
      setSecuritySettings(prev => ({
        ...prev,
        biometricAuth: false
      }));
    }
  };

  const toggleSetting = (key) => {
    setSecuritySettings(prev => ({
      ...prev,
      [key]: !prev[key]
    }));
  };

  const handleBiometricAuth = async () => {
    try {
      setIsLoading(true);
      console.log('Toggling biometric authentication...');
      
      // First check if biometric hardware is available
      const isAvailable = await biometricAuthService.isBiometricAvailable();
      console.log('Biometric hardware available:', isAvailable);
      
      if (!isAvailable) {
        Alert.alert('Not Available', 'Biometric authentication is not available on this device. Please check your device settings and ensure you have enrolled biometrics.');
        return;
      }
      
      if (!securitySettings.biometricAuth) {
        // Enable biometric auth
        console.log('Attempting to enable biometric authentication...');
        const result = await enableBiometric();
        console.log('Enable biometric result:', result);
        
        if (result.success) {
          setSecuritySettings(prev => ({
            ...prev,
            biometricAuth: true
          }));
          Alert.alert('Success', 'Biometric authentication enabled successfully');
        } else {
          Alert.alert('Error', result.error || 'Failed to enable biometric authentication');
        }
      } else {
        // Disable biometric auth
        console.log('Attempting to disable biometric authentication...');
        const result = await disableBiometric();
        console.log('Disable biometric result:', result);
        
        if (result.success) {
          setSecuritySettings(prev => ({
            ...prev,
            biometricAuth: false
          }));
          Alert.alert('Success', 'Biometric authentication disabled successfully');
        } else {
          Alert.alert('Error', result.error || 'Failed to disable biometric authentication');
        }
      }
    } catch (error) {
      console.error('Error toggling biometric authentication:', error);
      Alert.alert('Error', error.message || 'An error occurred while toggling biometric authentication');
    } finally {
      setIsLoading(false);
    }
  };

  const handleTwoFactorAuth = () => {
    Alert.alert(
      'Two-Factor Authentication',
      'Would you like to enable two-factor authentication?',
      [
        {
          text: 'Cancel',
          style: 'cancel',
        },
        {
          text: 'Enable',
          onPress: () => toggleSetting('twoFactorAuth'),
        },
      ]
    );
  };

  const handleDeleteAccount = () => {
    Alert.alert(
      'Delete Account',
      'Are you sure you want to delete your account? This action cannot be undone.',
      [
        {
          text: 'Cancel',
          style: 'cancel',
        },
        {
          text: 'Delete',
          style: 'destructive',
          onPress: async () => {
            try {
              const result = await deleteAccount();
              if (result.success) {
                // Navigation will be handled by the auth state change
                Alert.alert('Success', 'Your account has been deleted successfully');
              } else {
                Alert.alert('Error', result.error || 'Failed to delete account');
              }
            } catch (error) {
              Alert.alert('Error', error.message || 'Failed to delete account');
            }
          },
        },
      ]
    );
  };

  const renderSecurityItem = (title, description, value, onToggle, icon, isDangerous = false) => (
    <View style={styles.securityItem}>
      <View style={styles.securityLeft}>
        <View style={[styles.iconContainer, isDangerous && styles.dangerIconContainer]}>
          <Ionicons name={icon} size={24} color={isDangerous ? Colors.error : Colors.primary} />
        </View>
        <View style={styles.securityContent}>
          <Text style={styles.securityTitle}>{title}</Text>
          <Text style={styles.securityDescription}>{description}</Text>
        </View>
      </View>
      <Switch
        value={value}
        onValueChange={onToggle}
        trackColor={{ false: Colors.gray, true: isDangerous ? Colors.error : Colors.primary }}
        thumbColor={Colors.white}
        disabled={isLoading}
      />
    </View>
  );

  return (
    <View style={styles.container}>
      <Header 
        title="Privacy & Security" 
        showBackButton={true}
        onBackPress={() => navigation.goBack()}
      />

      <ScrollView style={styles.content} showsVerticalScrollIndicator={false}>
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Security Settings</Text>
          {renderSecurityItem(
            'Biometric Authentication',
            'Use fingerprint or face recognition to sign in',
            securitySettings.biometricAuth,
            handleBiometricAuth,
            'finger-print'
          )}
          {renderSecurityItem(
            'Two-Factor Authentication',
            'Add an extra layer of security to your account',
            securitySettings.twoFactorAuth,
            handleTwoFactorAuth,
            'shield-checkmark'
          )}
          {renderSecurityItem(
            'Email Verification',
            'Verify your email for important updates',
            securitySettings.emailVerification,
            () => toggleSetting('emailVerification'),
            'mail'
          )}
        </View>

        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Device Management</Text>
          {renderSecurityItem(
            'Device Management',
            'Manage devices connected to your account',
            securitySettings.deviceManagement,
            () => toggleSetting('deviceManagement'),
            'phone-portrait'
          )}
          {renderSecurityItem(
            'Activity Log',
            'Track your account activity and sign-ins',
            securitySettings.activityLog,
            () => toggleSetting('activityLog'),
            'time'
          )}
        </View>

        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Privacy Settings</Text>
          {renderSecurityItem(
            'Data Sharing',
            'Share anonymous usage data to improve the app',
            securitySettings.dataSharing,
            () => toggleSetting('dataSharing'),
            'share-social'
          )}
          {renderSecurityItem(
            'Marketing Emails',
            'Receive emails about new features and promotions',
            securitySettings.marketingEmails,
            () => toggleSetting('marketingEmails'),
            'mail-unread'
          )}
        </View>

        <TouchableOpacity 
          style={styles.menuItem}
          onPress={() => navigation.navigate('ChangePassword')}
        >
          <View style={styles.menuItemLeft}>
            <Ionicons name="key" size={24} color={Colors.primary} />
            <Text style={styles.menuItemText}>Change Password</Text>
          </View>
          <Ionicons name="chevron-forward" size={20} color={Colors.darkGray} />
        </TouchableOpacity>

        <TouchableOpacity 
          style={[styles.menuItem, styles.dangerMenuItem]}
          onPress={handleDeleteAccount}
        >
          <View style={styles.menuItemLeft}>
            <Ionicons name="trash" size={24} color={Colors.error} />
            <Text style={[styles.menuItemText, styles.dangerText]}>Delete Account</Text>
          </View>
          <Ionicons name="chevron-forward" size={20} color={Colors.error} />
        </TouchableOpacity>
      </ScrollView>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: Colors.background,
  },
  content: {
    flex: 1,
    paddingHorizontal: 20,
  },
  section: {
    backgroundColor: Colors.white,
    borderRadius: 12,
    overflow: 'hidden',
    marginBottom: 20,
    shadowColor: Colors.black,
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  sectionTitle: {
    fontSize: 18,
    fontWeight: '600',
    color: Colors.secondary,
    padding: 20,
    paddingBottom: 10,
  },
  securityItem: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    padding: 20,
    borderTopWidth: 1,
    borderTopColor: Colors.gray,
  },
  securityLeft: {
    flexDirection: 'row',
    alignItems: 'center',
    flex: 1,
  },
  iconContainer: {
    width: 40,
    height: 40,
    borderRadius: 20,
    backgroundColor: Colors.primary + '20',
    justifyContent: 'center',
    alignItems: 'center',
    marginRight: 12,
  },
  dangerIconContainer: {
    backgroundColor: Colors.error + '20',
  },
  securityContent: {
    flex: 1,
  },
  securityTitle: {
    fontSize: 16,
    fontWeight: '500',
    color: Colors.secondary,
    marginBottom: 4,
  },
  securityDescription: {
    fontSize: 14,
    color: Colors.darkGray,
  },
  menuItem: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    backgroundColor: Colors.white,
    borderRadius: 12,
    padding: 20,
    marginBottom: 20,
    shadowColor: Colors.black,
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  menuItemLeft: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  menuItemText: {
    fontSize: 16,
    color: Colors.secondary,
    marginLeft: 12,
  },
  dangerMenuItem: {
    marginBottom: 30,
  },
  dangerText: {
    color: Colors.error,
  },
});

export default PrivacySecurityScreen; 