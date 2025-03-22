import React, { useState } from 'react';
import { View, Text, StyleSheet, Switch, TouchableOpacity, ScrollView } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import Header from '../../components/Header';
import Colors from '../../constants/colors';

const NotificationsScreen = ({ navigation }) => {
  const [notifications, setNotifications] = useState({
    all: true,
    marketUpdates: true,
    portfolioAlerts: true,
    priceAlerts: true,
    newsUpdates: true,
    tradingSignals: true,
    systemUpdates: true,
    marketing: false,
  });

  const toggleNotification = (key) => {
    setNotifications(prev => ({
      ...prev,
      [key]: !prev[key]
    }));
  };

  const renderNotificationItem = (title, description, value, onToggle, icon) => (
    <View style={styles.notificationItem}>
      <View style={styles.notificationLeft}>
        <View style={styles.iconContainer}>
          <Ionicons name={icon} size={24} color={Colors.primary} />
        </View>
        <View style={styles.notificationContent}>
          <Text style={styles.notificationTitle}>{title}</Text>
          <Text style={styles.notificationDescription}>{description}</Text>
        </View>
      </View>
      <Switch
        value={value}
        onValueChange={onToggle}
        trackColor={{ false: Colors.gray, true: Colors.primary }}
        thumbColor={Colors.white}
      />
    </View>
  );

  return (
    <View style={styles.container}>
      <Header 
        title="Notifications" 
        showBackButton={true}
        onBackPress={() => navigation.goBack()}
      />

      <ScrollView style={styles.content} showsVerticalScrollIndicator={false}>
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>General Notifications</Text>
          {renderNotificationItem(
            'All Notifications',
            'Enable or disable all notifications',
            notifications.all,
            () => toggleNotification('all'),
            'notifications'
          )}
        </View>

        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Market & Trading</Text>
          {renderNotificationItem(
            'Market Updates',
            'Get notified about major market movements',
            notifications.marketUpdates,
            () => toggleNotification('marketUpdates'),
            'trending-up'
          )}
          {renderNotificationItem(
            'Portfolio Alerts',
            'Receive updates about your portfolio performance',
            notifications.portfolioAlerts,
            () => toggleNotification('portfolioAlerts'),
            'pie-chart'
          )}
          {renderNotificationItem(
            'Price Alerts',
            'Get notified when stocks reach your target prices',
            notifications.priceAlerts,
            () => toggleNotification('priceAlerts'),
            'cash'
          )}
          {renderNotificationItem(
            'Trading Signals',
            'Receive AI-powered trading recommendations',
            notifications.tradingSignals,
            () => toggleNotification('tradingSignals'),
            'analytics'
          )}
        </View>

        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Updates & News</Text>
          {renderNotificationItem(
            'News Updates',
            'Stay informed with latest market news',
            notifications.newsUpdates,
            () => toggleNotification('newsUpdates'),
            'newspaper'
          )}
          {renderNotificationItem(
            'System Updates',
            'Get notified about app updates and maintenance',
            notifications.systemUpdates,
            () => toggleNotification('systemUpdates'),
            'construct'
          )}
        </View>

        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Marketing</Text>
          {renderNotificationItem(
            'Marketing Communications',
            'Receive updates about new features and promotions',
            notifications.marketing,
            () => toggleNotification('marketing'),
            'megaphone'
          )}
        </View>

        <TouchableOpacity style={styles.clearButton}>
          <Ionicons name="trash-outline" size={20} color={Colors.error} />
          <Text style={styles.clearButtonText}>Clear All Notifications</Text>
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
  notificationItem: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    padding: 20,
    borderTopWidth: 1,
    borderTopColor: Colors.gray,
  },
  notificationLeft: {
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
  notificationContent: {
    flex: 1,
  },
  notificationTitle: {
    fontSize: 16,
    fontWeight: '500',
    color: Colors.secondary,
    marginBottom: 4,
  },
  notificationDescription: {
    fontSize: 14,
    color: Colors.darkGray,
  },
  clearButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: Colors.white,
    borderRadius: 12,
    padding: 16,
    marginBottom: 20,
    shadowColor: Colors.black,
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  clearButtonText: {
    fontSize: 16,
    color: Colors.error,
    marginLeft: 8,
    fontWeight: '500',
  },
});

export default NotificationsScreen; 
 