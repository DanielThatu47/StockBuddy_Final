import React from 'react';
import { createBottomTabNavigator } from '@react-navigation/bottom-tabs';
import { Ionicons } from '@expo/vector-icons';
import { View, StyleSheet } from 'react-native';

// Import screens
import HomeScreen from '../screens/HomeScreen';
import WatchlistScreen from '../screens/WatchlistScreen';
import PortfolioScreen from '../screens/portfolio/PortfolioScreen';
import NewsScreen from '../screens/news/NewsScreen';
import ProfileScreen from '../screens/ProfileScreen';
import AIPredictionScreen from '../screens/AIPredictionScreen';

const Tab = createBottomTabNavigator();

const TabNavigator = () => {
  return (
    <Tab.Navigator
      screenOptions={{
        tabBarActiveTintColor: '#3970BE',
        tabBarInactiveTintColor: '#8A8A8A',
        tabBarStyle: {
          backgroundColor: '#FFFFFF',
          borderTopWidth: 0,
          height: 60,
          paddingBottom: 10,
        },
        headerShown: false,
      }}
    >
      <Tab.Screen
        name="Home"
        component={HomeScreen}
        options={{
          tabBarIcon: ({ color, focused }) => (
            <View style={focused ? styles.activeIconContainer : null}>
              <Ionicons name="home" size={24} color={color} />
            </View>
          ),
        }}
      />
      <Tab.Screen
        name="Watchlist"
        component={WatchlistScreen}
        options={{
          tabBarIcon: ({ color, focused }) => (
            <View style={focused ? styles.activeIconContainer : null}>
              <Ionicons name="star" size={24} color={color} />
            </View>
          ),
        }}
      />
      <Tab.Screen
        name="Portfolio"
        component={PortfolioScreen}
        options={{
          tabBarIcon: ({ color, focused }) => (
            <View style={focused ? styles.activeIconContainer : null}>
              <Ionicons name="pie-chart" size={24} color={color} />
            </View>
          ),
        }}
      />
      <Tab.Screen
        name="AIPrediction"
        component={AIPredictionScreen}
        options={{
          title: "AI Predict",
          tabBarIcon: ({ color, focused }) => (
            <View style={focused ? styles.activeIconContainer : null}>
              <Ionicons name="analytics" size={24} color={color} />
            </View>
          ),
        }}
      />
      <Tab.Screen
        name="News"
        component={NewsScreen}
        options={{
          tabBarIcon: ({ color, focused }) => (
            <View style={focused ? styles.activeIconContainer : null}>
              <Ionicons name="newspaper" size={24} color={color} />
            </View>
          ),
        }}
      />
      <Tab.Screen
        name="Profile"
        component={ProfileScreen}
        options={{
          tabBarIcon: ({ color, focused }) => (
            <View style={focused ? styles.activeIconContainer : null}>
              <Ionicons name="person" size={24} color={color} />
            </View>
          ),
        }}
      />
    </Tab.Navigator>
  );
};

const styles = StyleSheet.create({
  activeIconContainer: {
    borderBottomWidth: 2,
    borderBottomColor: '#3970BE',
    paddingBottom: 5,
  },
});

export default TabNavigator; 