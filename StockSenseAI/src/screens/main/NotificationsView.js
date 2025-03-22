import React, { useState, useEffect } from 'react';
import {
  View,
  Text,
  StyleSheet,
  FlatList,
  TouchableOpacity,
  SafeAreaView,
  Image,
  Animated,
} from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import Header from '../../components/Header';
import Colors from '../../constants/colors';

const NotificationsView = ({ navigation }) => {
  const [notifications, setNotifications] = useState([
    {
      id: '1',
      title: 'Stock Alert: AAPL',
      message: 'Apple Inc. stock has reached your target price of $180',
      type: 'price_alert',
      timestamp: new Date(Date.now() - 1000 * 60 * 5), // 5 minutes ago
      read: false,
      icon: 'trending-up',
      color: Colors.primary,
    },
    {
      id: '2',
      title: 'Portfolio Update',
      message: 'Your portfolio value has increased by 5% this week',
      type: 'portfolio',
      timestamp: new Date(Date.now() - 1000 * 60 * 60), // 1 hour ago
      read: false,
      icon: 'pie-chart',
      color: Colors.success,
    },
    {
      id: '3',
      title: 'News Alert',
      message: 'Breaking: New market analysis available for your watchlist stocks',
      type: 'news',
      timestamp: new Date(Date.now() - 1000 * 60 * 60 * 2), // 2 hours ago
      read: true,
      icon: 'newspaper',
      color: Colors.warning,
    },
    {
      id: '4',
      title: 'AI Prediction',
      message: 'New AI prediction available for MSFT',
      type: 'ai',
      timestamp: new Date(Date.now() - 1000 * 60 * 60 * 24), // 1 day ago
      read: true,
      icon: 'analytics',
      color: Colors.info,
    },
  ]);

  const [unreadCount, setUnreadCount] = useState(0);
  const [showFilters, setShowFilters] = useState(false);
  const [selectedFilter, setSelectedFilter] = useState('all');

  useEffect(() => {
    updateUnreadCount();
  }, [notifications]);

  const updateUnreadCount = () => {
    const count = notifications.filter(n => !n.read).length;
    setUnreadCount(count);
  };

  const markAsRead = (id) => {
    setNotifications(notifications.map(notification =>
      notification.id === id ? { ...notification, read: true } : notification
    ));
  };

  const markAllAsRead = () => {
    setNotifications(notifications.map(notification => ({
      ...notification,
      read: true
    })));
  };

  const clearAll = () => {
    setNotifications([]);
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
    return notification.type === selectedFilter;
  });

  const renderNotification = ({ item }) => (
    <TouchableOpacity
      style={[
        styles.notificationItem,
        !item.read && styles.unreadNotification
      ]}
      onPress={() => markAsRead(item.id)}
    >
      <View style={[styles.iconContainer, { backgroundColor: item.color + '20' }]}>
        <Ionicons name={item.icon} size={24} color={item.color} />
      </View>
      <View style={styles.notificationContent}>
        <View style={styles.notificationHeader}>
          <Text style={styles.notificationTitle}>{item.title}</Text>
          <Text style={styles.timestamp}>{getTimeAgo(item.timestamp)}</Text>
        </View>
        <Text style={styles.notificationMessage}>{item.message}</Text>
      </View>
      {!item.read && <View style={styles.unreadDot} />}
    </TouchableOpacity>
  );

  const renderFilterButton = (type, label) => (
    <TouchableOpacity
      style={[
        styles.filterButton,
        selectedFilter === type && styles.filterButtonActive
      ]}
      onPress={() => setSelectedFilter(type)}
    >
      <Text style={[
        styles.filterButtonText,
        selectedFilter === type && styles.filterButtonTextActive
      ]}>
        {label}
      </Text>
    </TouchableOpacity>
  );

  return (
    <SafeAreaView style={styles.container}>
      <Header
        title="Notifications"
        showBackButton={true}
        onBackPress={() => navigation.goBack()}
        rightComponent={
          <View style={styles.headerActions}>
            <TouchableOpacity
              style={styles.headerButton}
              onPress={() => setShowFilters(!showFilters)}
            >
              <Ionicons name="filter" size={24} color={Colors.primary} />
            </TouchableOpacity>
            <TouchableOpacity
              style={styles.headerButton}
              onPress={markAllAsRead}
            >
              <Ionicons name="checkmark-done" size={24} color={Colors.primary} />
            </TouchableOpacity>
            <TouchableOpacity
              style={styles.headerButton}
              onPress={clearAll}
            >
              <Ionicons name="trash-outline" size={24} color={Colors.primary} />
            </TouchableOpacity>
          </View>
        }
      />

      {showFilters && (
        <View style={styles.filtersContainer}>
          {renderFilterButton('all', 'All')}
          {renderFilterButton('price_alert', 'Price Alerts')}
          {renderFilterButton('portfolio', 'Portfolio')}
          {renderFilterButton('news', 'News')}
          {renderFilterButton('ai', 'AI Predictions')}
        </View>
      )}

      <FlatList
        data={filteredNotifications}
        renderItem={renderNotification}
        keyExtractor={item => item.id}
        contentContainerStyle={styles.listContainer}
        ListEmptyComponent={
          <View style={styles.emptyContainer}>
            <Ionicons name="notifications-off" size={64} color={Colors.gray} />
            <Text style={styles.emptyText}>No notifications</Text>
          </View>
        }
      />
    </SafeAreaView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: Colors.background,
  },
  headerActions: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  headerButton: {
    padding: 8,
    marginLeft: 8,
  },
  filtersContainer: {
    flexDirection: 'row',
    padding: 12,
    backgroundColor: Colors.white,
    borderBottomWidth: 1,
    borderBottomColor: Colors.gray,
  },
  filterButton: {
    paddingHorizontal: 16,
    paddingVertical: 8,
    borderRadius: 20,
    backgroundColor: Colors.background,
    marginRight: 8,
  },
  filterButtonActive: {
    backgroundColor: Colors.primary,
  },
  filterButtonText: {
    color: Colors.secondary,
    fontSize: 14,
    fontWeight: '500',
  },
  filterButtonTextActive: {
    color: Colors.white,
  },
  listContainer: {
    padding: 16,
  },
  notificationItem: {
    flexDirection: 'row',
    padding: 16,
    backgroundColor: Colors.white,
    borderRadius: 12,
    marginBottom: 12,
    alignItems: 'center',
    shadowColor: Colors.black,
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  unreadNotification: {
    backgroundColor: Colors.primary + '10',
  },
  iconContainer: {
    width: 48,
    height: 48,
    borderRadius: 24,
    justifyContent: 'center',
    alignItems: 'center',
    marginRight: 12,
  },
  notificationContent: {
    flex: 1,
  },
  notificationHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 4,
  },
  notificationTitle: {
    fontSize: 16,
    fontWeight: '600',
    color: Colors.secondary,
  },
  notificationMessage: {
    fontSize: 14,
    color: Colors.darkGray,
    lineHeight: 20,
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
  emptyContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    paddingTop: 100,
  },
  emptyText: {
    fontSize: 16,
    color: Colors.gray,
    marginTop: 16,
  },
});

export default NotificationsView; 