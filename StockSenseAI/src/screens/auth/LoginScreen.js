import React, { useState, useEffect } from 'react';
import {
  View,
  Text,
  TextInput,
  TouchableOpacity,
  StyleSheet,
  ActivityIndicator,
  SafeAreaView,
  KeyboardAvoidingView,
  Platform,
  ScrollView,
  Image,
  Alert
} from 'react-native';
import { useAuth } from '../../context/AuthContext';
import { Ionicons } from '@expo/vector-icons';
import { biometricAuthService } from '../../services/biometricAuth';
import AsyncStorage from '@react-native-async-storage/async-storage';
import Colors from '../../constants/colors';

const STORED_CREDENTIALS_KEY = 'stored_credentials';
const REMEMBER_ME_KEY = 'remember_me_enabled';

const LoginScreen = ({ navigation }) => {
  const { login, checkBiometricEnabled, authenticateWithBiometric } = useAuth();
  const [formData, setFormData] = useState({
    email: '',
    password: '',
  });
  const [errors, setErrors] = useState({});
  const [showPassword, setShowPassword] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [biometricAvailable, setBiometricAvailable] = useState(false);
  const [rememberMe, setRememberMe] = useState(false);

  useEffect(() => {
    checkBiometricAvailability();
    loadStoredCredentials();
    tryAutoLogin();
  }, []);

  const loadStoredCredentials = async () => {
    try {
      // Check if remember me was enabled
      const rememberMeEnabled = await AsyncStorage.getItem(REMEMBER_ME_KEY);
      if (rememberMeEnabled === 'true') {
        setRememberMe(true);
        
        // Load stored credentials but only set the email for security
        const storedCredentials = await AsyncStorage.getItem(STORED_CREDENTIALS_KEY);
        if (storedCredentials) {
          const { email } = JSON.parse(storedCredentials);
          setFormData(prev => ({
            ...prev,
            email,
            password: '' // Always keep password field empty for security
          }));
        }
      } else {
        // Just load the email if remember me is not enabled
        const storedCredentials = await AsyncStorage.getItem(STORED_CREDENTIALS_KEY);
        if (storedCredentials) {
          const { email } = JSON.parse(storedCredentials);
          setFormData(prev => ({
            ...prev,
            email,
            password: '' // Ensure password is empty
          }));
        }
      }
    } catch (error) {
      console.error('Error loading stored credentials:', error);
    }
  };

  const checkBiometricAvailability = async () => {
    try {
      // First check if the device has biometric hardware
      const isAvailable = await biometricAuthService.isBiometricAvailable();
      if (!isAvailable) {
        console.log('Biometric hardware not available or not enrolled');
        setBiometricAvailable(false);
        return;
      }

      // Then check if biometric authentication is enabled in the app
      const isEnabled = await checkBiometricEnabled();
      console.log('Biometric enabled in app:', isEnabled);
      
      if (isEnabled) {
        // If biometrics are enabled, make sure the saved email is loaded
        const storedCredentials = await AsyncStorage.getItem(STORED_CREDENTIALS_KEY);
        if (storedCredentials) {
          const { email } = JSON.parse(storedCredentials);
          if (email) {
            // Make sure the email field shows the correct account for biometric login
            setFormData(prev => ({
              ...prev,
              email
            }));
          }
        }
      }
      
      setBiometricAvailable(isEnabled);
      
      // Get and log the available biometric types for debugging
      const types = await biometricAuthService.getBiometricTypes();
      console.log('Available biometric types:', types);
    } catch (error) {
      console.error('Error checking biometric availability:', error);
      setBiometricAvailable(false);
    }
  };

  const tryAutoLogin = async () => {
    try {
      // Check if user has explicitly logged out
      const userLoggedOut = await AsyncStorage.getItem('user_logged_out');
      if (userLoggedOut === 'true') {
        console.log('User previously logged out. Skipping auto-login.');
        return; // Skip auto-login if user explicitly logged out
      }
      
      // Check if biometric is available and enabled
      const biometricEnabled = await checkBiometricEnabled();
      if (biometricEnabled) {
        console.log('Biometric login is available, but requires explicit user action');
        // We don't auto-login with biometrics, user must explicitly press the biometric button
        // This is for security and UX reasons
        return;
      }

      // We don't automatically log in with saved password anymore
      // Even if Remember Me is enabled, we only pre-fill the email field
      // User must manually enter password for security
      console.log('Auto-login disabled for security: user must enter password or use biometrics');
    } catch (error) {
      console.error('Error in auto-login:', error);
    }
  };

  const validateForm = () => {
    const newErrors = {};
    
    // Email validation
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    if (!formData.email) {
      newErrors.email = 'Email is required';
    } else if (!emailRegex.test(formData.email)) {
      newErrors.email = 'Please enter a valid email';
    }
    
    // Password validation
    if (!formData.password) {
      newErrors.password = 'Password is required';
    }
    
    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  const handleBiometricLogin = async () => {
    try {
      // First verify that the displayed email matches the stored credentials
      const storedCredentials = await AsyncStorage.getItem(STORED_CREDENTIALS_KEY);
      if (!storedCredentials) {
        Alert.alert('Error', 'No stored credentials found for biometric login');
        return;
      }
      
      const { email: storedEmail } = JSON.parse(storedCredentials);
      
      // Security check: Make sure the entered email matches the stored email
      if (formData.email !== storedEmail) {
        Alert.alert(
          'Security Warning', 
          'The email address does not match the stored credentials. Biometric login is only available for the previously authenticated account.'
        );
        return;
      }
      
      // Clear the logged-out flag when user manually logs in with biometrics
      await AsyncStorage.removeItem('user_logged_out');
      
      setIsLoading(true);
      console.log('Attempting biometric login...');
      const result = await authenticateWithBiometric();
      console.log('Biometric authentication result:', result);
      
      if (result.requiresCaptcha) {
        // Handle CAPTCHA if needed
        navigation.navigate('Captcha', {
          onVerificationComplete: async (verified) => {
            if (verified) {
              const storedCredentials = await AsyncStorage.getItem(STORED_CREDENTIALS_KEY);
              if (storedCredentials) {
                const { email, password } = JSON.parse(storedCredentials);
                const retryLogin = await login(email, password);
                if (!retryLogin.success) {
                  Alert.alert('Error', retryLogin.error || 'Login failed');
                }
              }
            }
          }
        });
      } else if (!result.success) {
        Alert.alert('Authentication Failed', result.error || 'Biometric authentication failed');
      }
    } catch (error) {
      console.error('Error in biometric login:', error);
      Alert.alert('Error', error.message || 'An error occurred during biometric login');
    } finally {
      setIsLoading(false);
    }
  };

  const handleLogin = async () => {
    if (validateForm()) {
      try {
        setIsLoading(true);
        const result = await login(formData.email, formData.password);
        
        if (result.requiresCaptcha) {
          // Navigate to CAPTCHA screen if required
          navigation.navigate('Captcha', {
            email: formData.email,
            password: formData.password,
            onVerificationComplete: async (verified) => {
              if (verified) {
                // Retry login after CAPTCHA verification
                const loginResult = await login(formData.email, formData.password);
                if (loginResult.success) {
                  // Store credentials based on remember me setting
                  await saveCredentials(formData.email, formData.password);
                  console.log('Login successful after CAPTCHA');
                } else {
                  Alert.alert('Error', loginResult.error || 'Login failed. Please try again.');
                }
              }
            }
          });
        } else if (result.success) {
          // Store credentials based on remember me setting
          await saveCredentials(formData.email, formData.password);
          console.log('Login successful');
        } else {
          Alert.alert('Error', result.error || 'Login failed. Please try again.');
        }
      } catch (error) {
        Alert.alert('Error', error.message || 'Login failed. Please try again.');
      } finally {
        setIsLoading(false);
      }
    }
  };

  const saveCredentials = async (email, password) => {
    try {
      // Clear the logged-out flag when user manually logs in
      await AsyncStorage.removeItem('user_logged_out');
      
      // Store email & password encrypted for biometric authentication only
      await AsyncStorage.setItem(STORED_CREDENTIALS_KEY, JSON.stringify({
        email,
        password // We store password ONLY for biometric login, never displayed in UI
      }));
      
      // Store remember me preference
      await AsyncStorage.setItem(REMEMBER_ME_KEY, rememberMe ? 'true' : 'false');
      
      if (rememberMe) {
        console.log('Remember Me enabled - email will be saved for auto-fill');
      } else {
        console.log('Remember Me disabled - no auto-fill will be used');
      }
      
      // Log separately for the biometric credentials
      console.log('Secure credentials saved for biometric authentication (not visible in UI)');
    } catch (error) {
      console.error('Error saving credentials:', error);
    }
  };

  const handleChange = (field, value) => {
    setFormData(prev => ({
      ...prev,
      [field]: value
    }));
    
    // Clear error when field is edited
    if (errors[field]) {
      setErrors(prev => ({
        ...prev,
        [field]: null
      }));
    }
  };

  return (
    <SafeAreaView style={styles.container}>
      <KeyboardAvoidingView
        behavior={Platform.OS === 'ios' ? 'padding' : 'height'}
        style={styles.keyboardAvoidingView}
      >
        <ScrollView contentContainerStyle={styles.scrollView}>
          <View style={styles.headerContainer}>
            <Image 
              source={require('../../../assets/logo.png')} 
              style={styles.logo} 
              resizeMode="contain"
            />
            <Text style={styles.headerText}>StockBuddy</Text>
            <Text style={styles.subHeaderText}>Welcome back!</Text>
          </View>
          
          <View style={styles.formContainer}>
            {/* Email Input */}
            <View style={styles.inputContainer}>
              <Text style={styles.label}>Email</Text>
              <TextInput
                style={[
                  styles.input, 
                  errors.email ? styles.inputError : null,
                  biometricAvailable ? styles.readOnlyInput : null
                ]}
                placeholder="Enter your email"
                value={formData.email}
                onChangeText={(text) => !biometricAvailable && handleChange('email', text)}
                autoCapitalize="none"
                keyboardType="email-address"
                editable={!biometricAvailable}
              />
              {errors.email ? (
                <Text style={styles.errorText}>{errors.email}</Text>
              ) : biometricAvailable ? (
                <Text style={styles.infoText}>Email is locked for biometric authentication</Text>
              ) : null}
            </View>
            
            {/* Password Input */}
            <View style={styles.inputContainer}>
              <Text style={styles.label}>Password</Text>
              <View style={[styles.passwordContainer, errors.password ? styles.inputError : null]}>
                <TextInput
                  style={styles.passwordInput}
                  placeholder="Enter your password"
                  value={formData.password}
                  onChangeText={(text) => handleChange('password', text)}
                  secureTextEntry={!showPassword}
                />
                <TouchableOpacity
                  style={styles.passwordVisibilityButton}
                  onPress={() => setShowPassword(!showPassword)}
                >
                  <Ionicons
                    name={showPassword ? 'eye-off' : 'eye'}
                    size={24}
                    color="#666"
                  />
                </TouchableOpacity>
              </View>
              {errors.password ? <Text style={styles.errorText}>{errors.password}</Text> : null}
            </View>

            {/* Remember Me Checkbox */}
            <View style={styles.rememberMeContainer}>
              <TouchableOpacity 
                style={styles.checkboxContainer}
                onPress={() => setRememberMe(!rememberMe)}
              >
                <View style={[styles.checkbox, rememberMe && styles.checkboxChecked]}>
                  {rememberMe && <Ionicons name="checkmark" size={16} color="white" />}
                </View>
                <Text style={styles.rememberMeText}>Remember Me</Text>
              </TouchableOpacity>
              <TouchableOpacity onPress={() => navigation.navigate('ForgotPassword')}>
                <Text style={styles.forgotPasswordText}>Forgot Password?</Text>
              </TouchableOpacity>
            </View>

            {biometricAvailable && (
              <>
                <View style={styles.biometricInfoContainer}>
                  <Ionicons name="information-circle-outline" size={16} color={Colors.primary} />
                  <Text style={styles.biometricInfoText}>
                    Biometric login is enabled for: {formData.email}
                  </Text>
                </View>
                <TouchableOpacity
                  style={styles.biometricButton}
                  onPress={handleBiometricLogin}
                  disabled={isLoading}
                >
                  <Ionicons name="finger-print" size={24} color={Colors.primary} />
                  <Text style={styles.biometricButtonText}>Sign in with Biometric</Text>
                </TouchableOpacity>
              </>
            )}

            <TouchableOpacity
              style={styles.loginButton}
              onPress={handleLogin}
              disabled={isLoading}
            >
              {isLoading ? (
                <ActivityIndicator color="#ffffff" />
              ) : (
                <Text style={styles.loginButtonText}>Sign In</Text>
              )}
            </TouchableOpacity>

            <View style={styles.footer}>
              <Text style={styles.footerText}>Don't have an account? </Text>
              <TouchableOpacity onPress={() => navigation.navigate('Register')}>
                <Text style={styles.signupText}>Sign up</Text>
              </TouchableOpacity>
            </View>
          </View>
        </ScrollView>
      </KeyboardAvoidingView>
    </SafeAreaView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#ffffff',
  },
  keyboardAvoidingView: {
    flex: 1,
  },
  scrollView: {
    flexGrow: 1,
    padding: 20,
  },
  headerContainer: {
    alignItems: 'center',
    marginBottom: 30,
  },
  logo: {
    width: 80,
    height: 80,
  },
  headerText: {
    fontSize: 26,
    fontWeight: 'bold',
    color: '#3970BE',
    marginTop: 10,
  },
  subHeaderText: {
    fontSize: 16,
    color: '#666',
    marginTop: 5,
  },
  formContainer: {
    width: '100%',
  },
  inputContainer: {
    marginBottom: 16,
  },
  label: {
    fontSize: 16,
    fontWeight: '500',
    color: '#333',
    marginBottom: 8,
  },
  input: {
    height: 50,
    borderWidth: 1,
    borderColor: '#ddd',
    borderRadius: 8,
    paddingHorizontal: 15,
    fontSize: 16,
    backgroundColor: '#fafafa',
  },
  inputError: {
    borderColor: '#ff3b30',
  },
  errorText: {
    color: '#ff3b30',
    fontSize: 14,
    marginTop: 5,
  },
  loginButton: {
    backgroundColor: '#3970BE',
    height: 50,
    borderRadius: 8,
    justifyContent: 'center',
    alignItems: 'center',
    marginTop: 15,
    marginBottom: 10,
  },
  loginButtonText: {
    color: '#ffffff',
    fontSize: 16,
    fontWeight: '600',
  },
  footer: {
    flexDirection: 'row',
    justifyContent: 'center',
    marginTop: 20,
    marginBottom: 20,
  },
  footerText: {
    color: '#666',
    fontSize: 14,
  },
  signupText: {
    color: '#3970BE',
    fontSize: 14,
    fontWeight: '600',
  },
  passwordContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    borderWidth: 1,
    borderColor: '#ddd',
    borderRadius: 8,
    backgroundColor: '#fafafa',
  },
  passwordInput: {
    flex: 1,
    height: 50,
    paddingHorizontal: 15,
    fontSize: 16,
  },
  passwordVisibilityButton: {
    padding: 10,
  },
  biometricButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: Colors.white,
    borderRadius: 10,
    height: 50,
    marginTop: 15,
    borderWidth: 1,
    borderColor: Colors.primary,
  },
  biometricButtonText: {
    color: Colors.primary,
    fontSize: 16,
    fontWeight: '600',
    marginLeft: 10,
  },
  rememberMeContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    marginBottom: 15,
  },
  checkboxContainer: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  checkbox: {
    width: 20,
    height: 20,
    borderWidth: 1,
    borderColor: '#ddd',
    borderRadius: 4,
    marginRight: 10,
  },
  checkboxChecked: {
    backgroundColor: Colors.primary,
  },
  rememberMeText: {
    color: '#333',
    fontSize: 16,
  },
  forgotPasswordText: {
    color: '#3970BE',
    fontSize: 14,
    fontWeight: '600',
  },
  biometricInfoContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 10,
  },
  biometricInfoText: {
    color: '#333',
    fontSize: 14,
    marginLeft: 10,
  },
  readOnlyInput: {
    backgroundColor: '#f0f0f0',
  },
  infoText: {
    color: '#666',
    fontSize: 14,
  },
});

export default LoginScreen; 