import React, { useState, useEffect } from 'react';
import {
  View,
  Text,
  StyleSheet,
  FlatList,
  TouchableOpacity,
  Image,
  Animated,
  ScrollView,
} from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import Header from '../../components/Header';
import Colors from '../../constants/colors';

const NotifyView = ({ navigation }) => {
  const [notifications, setNotifications] = useState([
    {
      id: '1',
      title: 'Price Alert: AAPL',
      message: 'Apple Inc. stock has reached your target price of $175.50',
      timestamp: new Date(Date.now() - 1000 * 60 * 5), // 5 minutes ago
      type: 'price_alert',
      read: false,
      icon: 'trending-up',
    },
    {
      id: '2',
      title: 'Portfolio Update',
      message: 'Your portfolio value has increased by 2.5% today',
      timestamp: new Date(Date.now() - 1000 * 60 * 30), // 30 minutes ago
      type: 'portfolio',
      read: false,
      icon: 'wallet',
    },
    {
      id: '3',
      title: 'AI Prediction',
      message: 'New prediction available for MSFT stock',
      timestamp: new Date(Date.now() - 1000 * 60 * 60), // 1 hour ago
      type: 'ai_prediction',
      read: true,
      icon: 'analytics',
    },
    {
      id: '4',
      title: 'Market News',
      message: 'Breaking: Major market movement detected',
      timestamp: new Date(Date.now() - 1000 * 60 * 120), // 2 hours ago
      type: 'news',
      read: true,
      icon: 'newspaper',
    },
  ]);

  const [unreadCount, setUnreadCount] = useState(0);
  const [selectedFilter, setSelectedFilter] = useState('all');
  const fadeAnim = new Animated.Value(0);

  useEffect(() => {
    // Calculate unread notifications
    const unread = notifications.filter(n => !n.read).length;
    setUnreadCount(unread);

    // Animate fade in with a shorter duration
    Animated.timing(fadeAnim, {
      toValue: 1,
      duration: 150, // Reduced duration for faster transition
      useNativeDriver: true,
    }).start();
  }, [notifications]);

  const markAllAsRead = () => {
    setNotifications(notifications.map(notification => ({
      ...notification,
      read: true
    })));
  };

  const markAsRead = (id) => {
    setNotifications(notifications.map(notification => 
      notification.id === id ? { ...notification, read: true } : notification
    ));
  };

  const getTimeAgo = (timestamp) => {
    const now = new Date();
    const diff = now - timestamp;
    const minutes = Math.floor(diff / 60000);
    const hours = Math.floor(minutes / 60);
    const days = Math.floor(hours / 24);

    if (minutes < 60) return `${minutes}m ago`;
    if (hours < 24) return `${hours}h ago`;
    return `${days}d ago`;
  };

  const filteredNotifications = notifications.filter(notification => {
    if (selectedFilter === 'all') return true;
    if (selectedFilter === 'unread') return !notification.read;
    return notification.type === selectedFilter;
  });

  const renderNotification = ({ item }) => (
    <TouchableOpacity
      style={[styles.notificationItem, !item.read && styles.unreadItem]}
      onPress={() => markAsRead(item.id)}
    >
      <View style={styles.notificationIcon}>
        <Ionicons name={item.icon} size={24} color={Colors.primary} />
      </View>
      <View style={styles.notificationContent}>
        <View style={styles.notificationHeader}>
          <Text style={styles.notificationTitle}>{item.title}</Text>
          {!item.read && <View style={styles.unreadDot} />}
        </View>
        <Text style={styles.notificationMessage}>{item.message}</Text>
        <Text style={styles.timestamp}>{getTimeAgo(item.timestamp)}</Text>
      </View>
    </TouchableOpacity>
  );

  return (
    <View style={styles.container}>
      <Header
        title="Notifications"
        showBackButton={true}
        onBackPress={() => navigation.goBack()}
      >
        <TouchableOpacity style={styles.markAllRead} onPress={markAllAsRead}>
          <Text style={styles.markAllReadText}>Mark all as read</Text>
        </TouchableOpacity>
      </Header>

      <View style={styles.filterContainer}>
        <ScrollView horizontal showsHorizontalScrollIndicator={false}>
          {['all', 'unread', 'price_alert', 'portfolio', 'ai_prediction', 'news'].map((filter) => (
            <TouchableOpacity
              key={filter}
              style={[
                styles.filterButton,
                selectedFilter === filter && styles.filterButtonActive
              ]}
              onPress={() => setSelectedFilter(filter)}
            >
              <Text style={[
                styles.filterText,
                selectedFilter === filter && styles.filterTextActive
              ]}>
                {filter.charAt(0).toUpperCase() + filter.slice(1).replace('_', ' ')}
              </Text>
            </TouchableOpacity>
          ))}
        </ScrollView>
      </View>

      <FlatList
        data={filteredNotifications}
        renderItem={renderNotification}
        keyExtractor={item => item.id}
        contentContainerStyle={styles.notificationsList}
        showsVerticalScrollIndicator={false}
      />
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: Colors.background,
  },
  markAllRead: {
    padding: 8,
  },
  markAllReadText: {
    color: Colors.primary,
    fontSize: 14,
    fontWeight: '600',
  },
  filterContainer: {
    paddingVertical: 12,
    borderBottomWidth: 1,
    borderBottomColor: Colors.gray,
    backgroundColor: Colors.white,
  },
  filterButton: {
    paddingHorizontal: 16,
    paddingVertical: 8,
    marginHorizontal: 4,
    borderRadius: 20,
    backgroundColor: Colors.background,
    borderWidth: 1,
    borderColor: Colors.gray,
  },
  filterButtonActive: {
    backgroundColor: Colors.primary,
    borderColor: Colors.primary,
  },
  filterText: {
    color: Colors.secondary,
    fontSize: 14,
    fontWeight: '500',
  },
  filterTextActive: {
    color: Colors.white,
  },
  notificationsList: {
    padding: 16,
  },
  notificationItem: {
    flexDirection: 'row',
    padding: 16,
    backgroundColor: Colors.white,
    borderRadius: 12,
    marginBottom: 12,
    shadowColor: Colors.black,
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  unreadItem: {
    backgroundColor: Colors.background,
  },
  notificationIcon: {
    width: 40,
    height: 40,
    borderRadius: 20,
    backgroundColor: Colors.background,
    justifyContent: 'center',
    alignItems: 'center',
    marginRight: 12,
  },
  notificationContent: {
    flex: 1,
  },
  notificationHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 4,
  },
  notificationTitle: {
    fontSize: 16,
    fontWeight: '600',
    color: Colors.secondary,
    flex: 1,
  },
  notificationMessage: {
    fontSize: 14,
    color: Colors.darkGray,
    marginBottom: 4,
  },
  timestamp: {
    fontSize: 12,
    color: Colors.gray,
  },
  unreadDot: {
    width: 8,
    height: 8,
    borderRadius: 4,
    backgroundColor: Colors.primary,
    marginLeft: 8,
  },
});

export default NotifyView; 