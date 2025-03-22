import React, { useState, useEffect } from 'react';
import {
  View,
  Text,
  TextInput,
  TouchableOpacity,
  StyleSheet,
  ActivityIndicator,
  SafeAreaView,
  Image,
  Alert,
  Platform
} from 'react-native';
import { useAuth } from '../../context/AuthContext';
import { apiRequest } from '../../services/api';
import { Ionicons } from '@expo/vector-icons';

const CaptchaScreen = ({ route, navigation }) => {
  const { verifyCaptcha } = useAuth();
  const [captchaText, setCaptchaText] = useState('');
  const [userInput, setUserInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [attempts, setAttempts] = useState(0);
  const { onVerificationComplete } = route.params;

  useEffect(() => {
    loadCaptcha();
  }, []);

  const loadCaptcha = async () => {
    try {
      setIsLoading(true);
      setError(null);
      const response = await apiRequest('/captcha', 'GET');
      setCaptchaText(response.captchaText);
    } catch (error) {
      setError('Failed to load CAPTCHA. Please try again.');
    } finally {
      setIsLoading(false);
    }
  };

  const handleVerify = async () => {
    // Clear any previous errors
    setError(null);
    
    // Validate input
    if (!userInput.trim()) {
      setError('Please enter the verification code');
      return;
    }

    try {
      setIsLoading(true);
      
      // Verify the captcha
      const response = await apiRequest('/verify-captcha', 'POST', {
        userInput,
      });
      
      // Handle successful verification
      if (response.success) {
        // Only in this case do we call onVerificationComplete and go back
        onVerificationComplete(true);
        navigation.goBack();
        return; // Early return to completely separate success from failure
      }
      
      // If we get here, verification has failed
      setAttempts(prev => prev + 1);
      
      // Generate appropriate error message
      const errorMsg = response.message || 'Invalid CAPTCHA. Please try again.';
      const attemptsLeft = 5 - attempts;
      
      if (attempts >= 3 && attemptsLeft > 0) {
        setError(`${errorMsg} Remember: characters are case-sensitive (${attemptsLeft} attempts left)`);
      } else if (attemptsLeft <= 0) {
        setError('Maximum attempts reached. Please try a new CAPTCHA.');
      } else {
        setError(errorMsg);
      }
      
      // Always get a new captcha and reset input on failure
      loadCaptcha();
      setUserInput('');
      
    } catch (error) {
      // Handle any API errors
      setError('Connection error. Please try again.');
      loadCaptcha();
      setUserInput('');
    } finally {
      setIsLoading(false);
    }
  };

  const resetAttempts = () => {
    setAttempts(0);
    setError(null);
    loadCaptcha();
    setUserInput('');
  };

  return (
    <SafeAreaView style={styles.container}>
      <View style={styles.content}>
        <Text style={styles.title}>CAPTCHA Verification</Text>
        <Text style={styles.subtitle}>Please enter the verification code exactly as shown below</Text>
        
        {/* Add back button */}
        <TouchableOpacity 
          style={styles.backButton}
          onPress={() => navigation.goBack()}
        >
          <Ionicons name="arrow-back" size={24} color="#3970BE" />
          <Text style={styles.backButtonText}>Go Back</Text>
        </TouchableOpacity>
        
        {isLoading ? (
          <ActivityIndicator size="large" color="#3970BE" />
        ) : (
          <>
            <View style={styles.captchaContainer}>
              <Text style={styles.captchaText}>{captchaText}</Text>
            </View>
            
            <View style={styles.caseSensitiveNote}>
              <Ionicons name="information-circle" size={16} color="#3970BE" />
              <Text style={styles.caseSensitiveText}>Case-sensitive: Enter exactly as shown</Text>
            </View>
            
            {/* Display error message above the input */}
            {error ? <Text style={styles.errorText}>{error}</Text> : null}
            
            <View style={styles.inputContainer}>
              <TextInput
                style={styles.input}
                placeholder="Enter the code"
                placeholderTextColor="#aaa"
                value={userInput}
                onChangeText={setUserInput}
                autoCapitalize="none"
                maxLength={captchaText.length}
                autoCorrect={false}
                spellCheck={false}
                textAlign="center"
                textContentType="none"
              />
            </View>
            
            <TouchableOpacity
              style={styles.verifyButton}
              onPress={handleVerify}
              disabled={isLoading || !userInput}
            >
              <Text style={styles.verifyButtonText}>Verify</Text>
            </TouchableOpacity>
            
            {/* Add Reset button if too many attempts */}
            {attempts >= 5 && (
              <TouchableOpacity
                style={styles.resetButton}
                onPress={resetAttempts}
              >
                <Ionicons name="refresh-circle" size={20} color="#fff" style={styles.resetIcon} />
                <Text style={styles.resetButtonText}>Reset Attempts</Text>
              </TouchableOpacity>
            )}
            
            <TouchableOpacity
              style={styles.refreshButton}
              onPress={loadCaptcha}
              disabled={isLoading}
            >
              <Ionicons name="refresh" size={16} color="#3970BE" style={styles.refreshIcon} />
              <Text style={styles.refreshButtonText}>Refresh CAPTCHA</Text>
            </TouchableOpacity>
          </>
        )}
      </View>
    </SafeAreaView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#ffffff',
  },
  content: {
    flex: 1,
    padding: 20,
    justifyContent: 'center',
  },
  title: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#3970BE',
    marginBottom: 10,
    textAlign: 'center',
  },
  subtitle: {
    fontSize: 16,
    color: '#666',
    marginBottom: 30,
    textAlign: 'center',
  },
  captchaContainer: {
    backgroundColor: '#f0f5ff',
    padding: 20,
    borderRadius: 8,
    marginBottom: 10,
    alignItems: 'center',
    borderWidth: 1,
    borderColor: '#d0e0ff',
  },
  captchaText: {
    fontSize: 36,
    fontWeight: 'bold',
    letterSpacing: 5,
    color: '#333',
    fontFamily: Platform.OS === 'ios' ? 'Courier' : 'monospace',
    textAlign: 'center',
  },
  caseSensitiveNote: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: 20,
  },
  caseSensitiveText: {
    fontSize: 14,
    color: '#3970BE',
    marginLeft: 5,
  },
  inputContainer: {
    marginBottom: 20,
  },
  input: {
    height: 50,
    borderWidth: 1,
    borderColor: '#ddd',
    borderRadius: 8,
    paddingHorizontal: 15,
    fontSize: 24,
    textAlign: 'center',
    letterSpacing: 5,
    fontFamily: Platform.OS === 'ios' ? 'Courier' : 'monospace',
    color: '#333',
    includeFontPadding: false,
    textAlignVertical: 'center',
  },
  errorText: {
    color: '#ff3b30',
    fontSize: 14,
    marginBottom: 15,
    marginTop: 5,
    textAlign: 'center',
    backgroundColor: '#ffeeee',
    padding: 10,
    borderRadius: 5,
    borderWidth: 1,
    borderColor: '#ffcccc',
    fontWeight: '500',
  },
  verifyButton: {
    backgroundColor: '#3970BE',
    height: 50,
    borderRadius: 8,
    justifyContent: 'center',
    alignItems: 'center',
    marginBottom: 15,
  },
  verifyButtonText: {
    color: '#ffffff',
    fontSize: 16,
    fontWeight: '600',
  },
  refreshButton: {
    padding: 10,
    alignItems: 'center',
    flexDirection: 'row',
    justifyContent: 'center',
  },
  refreshIcon: {
    marginRight: 5,
  },
  refreshButtonText: {
    color: '#3970BE',
    fontSize: 14,
  },
  resetButton: {
    backgroundColor: '#3970BE',
    height: 50,
    borderRadius: 8,
    justifyContent: 'center',
    alignItems: 'center',
    marginBottom: 15,
    flexDirection: 'row',
  },
  resetIcon: {
    marginRight: 5,
  },
  resetButtonText: {
    color: '#ffffff',
    fontSize: 16,
    fontWeight: '600',
  },
  backButton: {
    position: 'absolute',
    top: 20,
    left: 20,
    flexDirection: 'row',
    alignItems: 'center',
  },
  backButtonText: {
    color: '#3970BE',
    fontSize: 16,
    fontWeight: '600',
    marginLeft: 5,
  },
});

export default CaptchaScreen; 