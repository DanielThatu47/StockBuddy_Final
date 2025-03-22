import React from 'react';
import { StatusBar } from 'react-native';
import { SafeAreaProvider } from 'react-native-safe-area-context';
import AppNavigator from './src/navigation/AppNavigator';
import Colors from './src/constants/colors';
import { AuthProvider } from './src/context/AuthContext';

const App = () => {
  return (
    <AuthProvider>
      <SafeAreaProvider>
        <StatusBar barStyle="light-content" backgroundColor={Colors.primary} />
        <AppNavigator />
      </SafeAreaProvider>
    </AuthProvider>
  );
};

export default App;
