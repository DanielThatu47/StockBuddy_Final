import React from 'react';
import { NavigationContainer } from '@react-navigation/native';
import { createNativeStackNavigator } from '@react-navigation/native-stack';
import { ActivityIndicator, View, StyleSheet } from 'react-native';

// Import navigators
import TabNavigator from './TabNavigator';
import AuthNavigator from './AuthNavigator';

// Import screens
import SearchScreen from '../screens/main/SearchScreen';
import AccountInfoScreen from '../screens/settings/AccountInfoScreen';
import PreferencesScreen from '../screens/settings/PreferencesScreen';
import NotificationsScreen from '../screens/settings/NotificationsScreen';
import PrivacySecurityScreen from '../screens/settings/PrivacySecurityScreen';
import HelpSupportScreen from '../screens/settings/HelpSupportScreen';
import AboutScreen from '../screens/settings/AboutScreen';
import NewsDetailScreen from '../screens/news/NewsDetailScreen';

// Import GUID screens
import GuidScreen from '../screens/guid/GuidScreen';
import GettingStartedScreen from '../screens/guid/GettingStartedScreen';

// Import new screens
import FAQScreen from '../screens/settings/FAQScreen';
import LiveChartScreen from '../screens/main/LiveChartScreen';
import LiveChatScreen from '../screens/settings/LiveChatScreen';
import NotifyView from '../screens/main/NotifyView';
import PredictionHistoryScreen from '../screens/PredictionHistoryScreen';
import PredictionDetailsScreen from '../screens/PredictionDetailsScreen';

// Import AuthContext
import { useAuth } from '../context/AuthContext';

const Stack = createNativeStackNavigator();

const AppNavigator = () => {
  const { loading, user } = useAuth();

  // Show loading screen while checking authentication status
  if (loading) {
    return (
      <View style={styles.loadingContainer}>
        <ActivityIndicator size="large" color="#3970BE" />
      </View>
    );
  }

  return (
    <NavigationContainer>
      {user ? (
        // User is authenticated, show main app screens
        <Stack.Navigator
          screenOptions={{
            headerStyle: {
              backgroundColor: '#FFFFFF',
              elevation: 0,
              shadowOpacity: 0,
            },
            headerTintColor: '#3970BE',
            headerTitleStyle: {
              fontWeight: '600',
            },
            headerBackTitleVisible: false,
            headerLeftContainerStyle: {
              paddingLeft: 10,
            },
          }}
        >
          <Stack.Screen
            name="MainTabs"
            component={TabNavigator}
            options={{ headerShown: false }}
          />
          <Stack.Screen
            name="Search"
            component={SearchScreen}
            options={{ headerShown: false }}
          />
          <Stack.Screen
            name="AccountInfo"
            component={AccountInfoScreen}
            options={{ headerShown: false }}
          />
          <Stack.Screen
            name="Preferences"
            component={PreferencesScreen}
            options={{ headerShown: false }}
          />
          <Stack.Screen
            name="Notifications"
            component={NotificationsScreen}
            options={{ headerShown: false }}
          />
          <Stack.Screen
            name="PrivacySecurity"
            component={PrivacySecurityScreen}
            options={{ headerShown: false }}
          />
          <Stack.Screen
            name="HelpSupport"
            component={HelpSupportScreen}
            options={{ headerShown: false }}
          />
          <Stack.Screen
            name="About"
            component={AboutScreen}
            options={{ headerShown: false }}
          />
          <Stack.Screen
            name="NewsDetail"
            component={NewsDetailScreen}
            options={{ headerShown: false }}
          />
          <Stack.Screen
            name="Guid"
            component={GuidScreen}
            options={{ headerShown: false }}
          />
          <Stack.Screen
            name="GettingStarted"
            component={GettingStartedScreen}
            options={{ headerShown: false }}
          />
          <Stack.Screen
            name="FAQ"
            component={FAQScreen}
            options={{ headerShown: false }}
          />
          <Stack.Screen
            name="LiveChart"
            component={LiveChartScreen}
            options={{ headerShown: false }}
          />
          <Stack.Screen
            name="LiveChat"
            component={LiveChatScreen}
            options={{ headerShown: false }}
          />
          <Stack.Screen
            name="NotifyView"
            component={NotifyView}
            options={{ headerShown: false }}
          />
          <Stack.Screen
            name="PredictionHistory"
            component={PredictionHistoryScreen}
            options={{ headerShown: false }}
          />
          <Stack.Screen
            name="PredictionDetails"
            component={PredictionDetailsScreen}
            options={{ headerShown: false }}
          />
        </Stack.Navigator>
      ) : (
        // User is not authenticated, show auth screens
        <AuthNavigator />
      )}
    </NavigationContainer>
  );
};

const styles = StyleSheet.create({
  loadingContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: '#FFFFFF',
  },
});

export default AppNavigator; 