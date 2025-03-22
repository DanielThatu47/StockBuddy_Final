import React from 'react';
import { createNativeStackNavigator } from '@react-navigation/native-stack';

// Import screens
import LoginScreen from '../screens/auth/LoginScreen';
import RegisterScreen from '../screens/auth/RegisterScreen';
import CaptchaScreen from '../screens/auth/CaptchaScreen';

const Stack = createNativeStackNavigator();

const AuthNavigator = () => {
  return (
    <Stack.Navigator
      screenOptions={{
        headerShown: false,
      }}
    >
      <Stack.Screen name="Login" component={LoginScreen} />
      <Stack.Screen name="Register" component={RegisterScreen} />
      <Stack.Screen name="Captcha" component={CaptchaScreen} />
    </Stack.Navigator>
  );
};

export default AuthNavigator; 