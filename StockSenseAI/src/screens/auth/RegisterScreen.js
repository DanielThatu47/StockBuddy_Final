import React, { useState } from 'react';
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
import DateTimePicker from '@react-native-community/datetimepicker';
import { useAuth } from '../../context/AuthContext';
import { Ionicons } from '@expo/vector-icons';
import CountryCodePicker from '../../components/CountryCodePicker';

const RegisterScreen = ({ navigation }) => {
  const { register, verifyCaptcha } = useAuth();
  const [formData, setFormData] = useState({
    name: '',
    email: '',
    password: '',
    confirmPassword: '',
    countryCode: '+91',
    phoneNumber: '',
    address: '',
    dateOfBirth: new Date(
      new Date().getFullYear() - 18,
      new Date().getMonth(),
      new Date().getDate()
    ),
  });
  
  const [errors, setErrors] = useState({});
  const [showDatePicker, setShowDatePicker] = useState(false);
  const [showPassword, setShowPassword] = useState(false);
  const [showConfirmPassword, setShowConfirmPassword] = useState(false);
  
  const [isLoading, setIsLoading] = useState(false);

  const validateForm = () => {
    const newErrors = {};
    
    // Name validation
    if (!formData.name.trim()) {
      newErrors.name = 'Name is required';
    }
    
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
    } else if (formData.password.length < 6) {
      newErrors.password = 'Password must be at least 6 characters';
    }
    
    // Confirm password validation
    if (formData.password !== formData.confirmPassword) {
      newErrors.confirmPassword = 'Passwords do not match';
    }
    
    // Address validation
    if (!formData.address.trim()) {
      newErrors.address = 'Address is required';
    }
    
    // Date of birth validation
    if (!formData.dateOfBirth) {
      newErrors.dateOfBirth = 'Date of birth is required';
    } else {
      const today = new Date();
      if (formData.dateOfBirth > today) {
        newErrors.dateOfBirth = 'Date of birth cannot be in the future';
      } else {
        // Check if user is at least 18 years old
        const birthDate = new Date(formData.dateOfBirth);
        const age = today.getFullYear() - birthDate.getFullYear();
        const monthDiff = today.getMonth() - birthDate.getMonth();
        
        if (monthDiff < 0 || (monthDiff === 0 && today.getDate() < birthDate.getDate())) {
          // If birthday hasn't occurred yet this year, subtract one year
          const calculatedAge = age - 1;
          if (calculatedAge < 18) {
            newErrors.dateOfBirth = 'You must be at least 18 years old to register';
          }
        } else {
          if (age < 18) {
            newErrors.dateOfBirth = 'You must be at least 18 years old to register';
          }
        }
      }
    }
    
    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  const handleRegister = async () => {
    if (validateForm()) {
      try {
        setIsLoading(true);
        
        // Navigate to CAPTCHA screen first
        navigation.navigate('Captcha', {
          onVerificationComplete: async (verified) => {
            if (verified) {
              // Format date for API
              const formattedDate = formData.dateOfBirth.toISOString().split('T')[0];
              
              // Call register with CAPTCHA verification status
              const result = await register({
                ...formData,
                dateOfBirth: formattedDate,
                captchaVerified: true
              });

              if (!result.success) {
                Alert.alert('Registration Error', result.error || 'Registration failed. Please try again.');
              } else {
                // Show success message
                Alert.alert(
                  'Registration Successful', 
                  'Your account has been created successfully. Please log in.',
                  [{ text: 'OK', onPress: () => navigation.navigate('Login') }]
                );
              }
            } else {
              Alert.alert('Verification Error', 'CAPTCHA verification failed. Please try again.');
            }
          }
        });
      } catch (error) {
        Alert.alert('Error', error.message || 'Registration failed. Please try again.');
      } finally {
        setIsLoading(false);
      }
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

  const handleDateChange = (event, selectedDate) => {
    setShowDatePicker(false);
    if (selectedDate) {
      // Check if the selected date makes the user at least 18 years old
      const today = new Date();
      const birthDate = new Date(selectedDate);
      let age = today.getFullYear() - birthDate.getFullYear();
      const monthDiff = today.getMonth() - birthDate.getMonth();
      
      if (monthDiff < 0 || (monthDiff === 0 && today.getDate() < birthDate.getDate())) {
        age--;
      }
      
      if (age >= 18) {
        handleChange('dateOfBirth', selectedDate);
      } else {
        setErrors(prev => ({
          ...prev,
          dateOfBirth: 'You must be at least 18 years old to register'
        }));
      }
    }
  };

  const toggleShowPassword = () => {
    setShowPassword(!showPassword);
  };

  const toggleShowConfirmPassword = () => {
    setShowConfirmPassword(!showConfirmPassword);
  };

  const formatDate = (date) => {
    return date.toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'long',
      day: 'numeric',
    });
  };

  // Calculate maximum date (18 years ago from today)
  const getMaximumDate = () => {
    const today = new Date();
    return new Date(
      today.getFullYear() - 18,
      today.getMonth(),
      today.getDate()
    );
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
            <Text style={styles.subHeaderText}>Create your account</Text>
          </View>
          
          <View style={styles.formContainer}>
            {/* Name Input */}
            <View style={styles.inputContainer}>
              <Text style={styles.label}>Full Name</Text>
              <TextInput
                style={[styles.input, errors.name ? styles.inputError : null]}
                placeholder="Enter your full name"
                value={formData.name}
                onChangeText={(text) => handleChange('name', text)}
              />
              {errors.name ? <Text style={styles.errorText}>{errors.name}</Text> : null}
            </View>
            
            {/* Email Input */}
            <View style={styles.inputContainer}>
              <Text style={styles.label}>Email</Text>
              <TextInput
                style={[styles.input, errors.email ? styles.inputError : null]}
                placeholder="Enter your email"
                value={formData.email}
                onChangeText={(text) => handleChange('email', text)}
                autoCapitalize="none"
                keyboardType="email-address"
              />
              {errors.email ? <Text style={styles.errorText}>{errors.email}</Text> : null}
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
                  onPress={toggleShowPassword}
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
            
            {/* Confirm Password Input */}
            <View style={styles.inputContainer}>
              <Text style={styles.label}>Confirm Password</Text>
              <View style={[styles.passwordContainer, errors.confirmPassword ? styles.inputError : null]}>
                <TextInput
                  style={styles.passwordInput}
                  placeholder="Confirm your password"
                  value={formData.confirmPassword}
                  onChangeText={(text) => handleChange('confirmPassword', text)}
                  secureTextEntry={!showConfirmPassword}
                />
                <TouchableOpacity
                  style={styles.passwordVisibilityButton}
                  onPress={toggleShowConfirmPassword}
                >
                  <Ionicons
                    name={showConfirmPassword ? 'eye-off' : 'eye'}
                    size={24}
                    color="#666"
                  />
                </TouchableOpacity>
              </View>
              {errors.confirmPassword ? <Text style={styles.errorText}>{errors.confirmPassword}</Text> : null}
            </View>
            
            {/* Phone Number Input */}
            <View style={styles.inputContainer}>
              <Text style={styles.label}>Phone Number</Text>
              <View style={styles.phoneContainer}>
                <CountryCodePicker 
                  selectedCode={formData.countryCode}
                  onSelectCountry={(code) => handleChange('countryCode', code)}
                />
                <TextInput
                  style={styles.phoneInput}
                  placeholder="Enter your phone number"
                  value={formData.phoneNumber}
                  onChangeText={(text) => handleChange('phoneNumber', text)}
                  keyboardType="phone-pad"
                />
              </View>
            </View>
            
            {/* Address Input */}
            <View style={styles.inputContainer}>
              <Text style={styles.label}>Address</Text>
              <TextInput
                style={[styles.input, errors.address ? styles.inputError : null]}
                placeholder="Enter your address"
                value={formData.address}
                onChangeText={(text) => handleChange('address', text)}
              />
              {errors.address ? <Text style={styles.errorText}>{errors.address}</Text> : null}
            </View>
            
            {/* Date of Birth Input */}
            <View style={styles.inputContainer}>
              <Text style={styles.label}>Date of Birth</Text>
              <Text style={styles.helperText}>(Must be at least 18 years old)</Text>
              <TouchableOpacity 
                style={[styles.input, styles.dateInput, errors.dateOfBirth ? styles.inputError : null]} 
                onPress={() => setShowDatePicker(true)}
              >
                <Text>{formatDate(formData.dateOfBirth)}</Text>
              </TouchableOpacity>
              {errors.dateOfBirth ? <Text style={styles.errorText}>{errors.dateOfBirth}</Text> : null}
              
              {showDatePicker && (
                <DateTimePicker
                  value={formData.dateOfBirth}
                  mode="date"
                  display="default"
                  onChange={handleDateChange}
                  maximumDate={getMaximumDate()}
                />
              )}
            </View>

            <TouchableOpacity
              style={styles.registerButton}
              onPress={handleRegister}
              disabled={isLoading}
            >
              {isLoading ? (
                <ActivityIndicator color="#ffffff" />
              ) : (
                <Text style={styles.registerButtonText}>Create Account</Text>
              )}
            </TouchableOpacity>

            <View style={styles.footer}>
              <Text style={styles.footerText}>Already have an account? </Text>
              <TouchableOpacity onPress={() => navigation.navigate('Login')}>
                <Text style={styles.signinText}>Sign in</Text>
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
  phoneContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 5,
  },
  phoneInput: {
    flex: 1,
    height: 50,
    borderWidth: 1,
    borderColor: '#ddd',
    borderRadius: 8,
    paddingHorizontal: 15,
    fontSize: 16,
    backgroundColor: '#fafafa',
  },
  dateInput: {
    justifyContent: 'center',
  },
  inputError: {
    borderColor: '#ff3b30',
  },
  errorText: {
    color: '#ff3b30',
    fontSize: 14,
    marginTop: 5,
  },
  registerButton: {
    backgroundColor: '#3970BE',
    height: 50,
    borderRadius: 8,
    justifyContent: 'center',
    alignItems: 'center',
    marginTop: 15,
    marginBottom: 10,
  },
  registerButtonText: {
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
  signinText: {
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
  helperText: {
    color: '#666',
    fontSize: 12,
    marginBottom: 5,
  },
});

export default RegisterScreen; 