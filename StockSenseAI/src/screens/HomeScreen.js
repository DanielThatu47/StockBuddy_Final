import React, { useState, useEffect } from 'react';
import { View, Text, ScrollView, StyleSheet, TouchableOpacity, RefreshControl } from 'react-native';
import { Ionicons, MaterialCommunityIcons, FontAwesome5 } from '@expo/vector-icons';
import Header from '../components/Header';
import SectionHeader from '../components/SectionHeader';
import StockCard from '../components/StockCard';
import StatusCard from '../components/StatusCard';
import FeatureCard from '../components/FeatureCard';
import NewsCard from '../components/NewsCard';
import Colors from '../constants/colors';

const HomeScreen = ({ navigation }) => {
  const [refreshing, setRefreshing] = useState(false);
  const [unreadNotifications, setUnreadNotifications] = useState(0);

  // Mock data for trending stocks
  const trendingStocks = [
    { symbol: 'MSFT', name: 'Microsoft', price: 150.25, priceChange: 2.5, volume: '2.5M' },
    { symbol: 'AMZN', name: 'Amazon', price: 150.25, priceChange: -1.2, volume: '1.8M' },
    { symbol: 'TSLA', name: 'Tesla', price: 150.25, priceChange: 2.5, volume: '3.2M' },
    { symbol: 'AAPL', name: 'Apple', price: 175.50, priceChange: 1.8, volume: '4.1M' },
    { symbol: 'GOOGL', name: 'Alphabet', price: 2800.75, priceChange: -0.5, volume: '1.2M' },
  ];

  // Mock data for features
  const features = [
    { id: 1, title: 'AIPredict', icon: 'brain', type: 'MaterialCommunityIcons', color: '#8e44ad' },
    { id: 2, title: 'Portfolio Analysis', icon: 'chart-line-variant', type: 'MaterialCommunityIcons', color: '#3498db' },
    { id: 3, title: 'Risk Assessment', icon: 'shield-check', type: 'MaterialCommunityIcons', color: '#2ecc71' },
    { id: 4, title: 'Market Scanner', icon: 'radar', type: 'MaterialCommunityIcons', color: '#e74c3c' },
    { id: 5, title: 'Trading Bots', icon: 'robot', type: 'MaterialCommunityIcons', color: '#f39c12' },
    { id: 6, title: 'Alerts', icon: 'bell-ring', type: 'MaterialCommunityIcons', color: '#1abc9c' },
  ];

  // Mock data for news
  const news = [
    {
      id: 1,
      title: 'Latest Market Updates and Trends',
      summary: 'Get the latest insights on market movements and expert analysis.',
      time: '2 hours ago',
      sentiment: 'Positive',
    },
    {
      id: 2,
      title: 'Latest Market Updates and Trends',
      summary: 'Get the latest insights on market movements and expert analysis.',
      time: '3 hours ago',
      sentiment: 'Neutral',
    },
  ];

  const onRefresh = React.useCallback(() => {
    setRefreshing(true);
    // Simulate data refresh
    setTimeout(() => {
      setRefreshing(false);
    }, 2000);
  }, []);

  // Mock function to get unread notifications count
  useEffect(() => {
    // In a real app, this would come from your backend
    setUnreadNotifications(2); // Mock value
  }, []);

  return (
    <View style={styles.container}>
      <Header
        title="StockBuddy"
        subtitle="Smart Stock Predictions"
        onSearchPress={() => navigation.navigate('Search')}
      >
        <TouchableOpacity 
          style={styles.headerIcon}
          onPress={() => navigation.navigate('NotifyView')}
        >
          <View style={styles.notificationContainer}>
            <Ionicons name="notifications-outline" size={24} color={Colors.white} />
            {unreadNotifications > 0 && (
              <View style={styles.notificationBadge}>
                <Text style={styles.notificationBadgeText}>
                  {unreadNotifications > 9 ? '9+' : unreadNotifications}
                </Text>
              </View>
            )}
          </View>
        </TouchableOpacity>
      </Header>

      <ScrollView 
        style={styles.content}
        showsVerticalScrollIndicator={false}
        refreshControl={
          <RefreshControl refreshing={refreshing} onRefresh={onRefresh} />
        }
      >
        <SectionHeader title="Market Overview" />
        
        <StatusCard 
          status="Market is currently closed" 
          message="Opens in 2 hours 30 minutes" 
        />

        <SectionHeader title="Quick Access" />
        
        <View style={styles.featuresGrid}>
          {features.map((feature) => (
            <FeatureCard
              key={feature.id}
              title={feature.title}
              icon={feature.icon}
              backgroundColor={feature.color}
              onPress={() => {
                if (feature.id === 1) { // AIPredict
                  navigation.navigate('AIPrediction');
                }
                // Add other feature navigations here as needed
              }}
            />
          ))}
        </View>

        <SectionHeader 
          title="Trending Stocks" 
          actionText="See All"
          onActionPress={() => {}}
        />
        
        <ScrollView 
          horizontal 
          showsHorizontalScrollIndicator={false}
          style={styles.horizontalScroll}
          contentContainerStyle={styles.horizontalScrollContent}
        >
          {trendingStocks.map((stock, index) => (
            <TouchableOpacity
              key={index}
              style={styles.stockCard}
              onPress={() => {}}
            >
              <View style={styles.stockHeader}>
                <Text style={styles.stockSymbol}>{stock.symbol}</Text>
                <Text style={styles.stockName}>{stock.name}</Text>
              </View>
              <View style={styles.stockPriceContainer}>
                <Text style={styles.stockPrice}>${stock.price}</Text>
                <View style={[
                  styles.priceChangeContainer,
                  { backgroundColor: stock.priceChange >= 0 ? Colors.success : Colors.error }
                ]}>
                  <Ionicons 
                    name={stock.priceChange >= 0 ? 'arrow-up' : 'arrow-down'} 
                    size={12} 
                    color={Colors.white} 
                  />
                  <Text style={styles.priceChange}>
                    {Math.abs(stock.priceChange)}%
                  </Text>
                </View>
              </View>
              <Text style={styles.stockVolume}>Vol: {stock.volume}</Text>
            </TouchableOpacity>
          ))}
        </ScrollView>

        <SectionHeader 
          title="Market News"
          actionText="More News"
          onActionPress={() => {}}
        />
        
        {news.map((item) => (
          <NewsCard
            key={item.id}
            title={item.title}
            summary={item.summary}
            time={item.time}
            sentiment={item.sentiment}
            onPress={() => {}}
          />
        ))}
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
  headerIcon: {
    padding: 4,
  },
  notificationContainer: {
    position: 'relative',
  },
  notificationBadge: {
    position: 'absolute',
    top: -5,
    right: -5,
    backgroundColor: Colors.error,
    borderRadius: 10,
    minWidth: 20,
    height: 20,
    justifyContent: 'center',
    alignItems: 'center',
    paddingHorizontal: 4,
  },
  notificationBadgeText: {
    color: Colors.white,
    fontSize: 12,
    fontWeight: 'bold',
  },
  featuresGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    justifyContent: 'space-between',
    marginVertical: 10,
  },
  horizontalScroll: {
    marginVertical: 10,
  },
  horizontalScrollContent: {
    paddingRight: 20,
  },
  stockCard: {
    width: 160,
    backgroundColor: Colors.white,
    borderRadius: 12,
    padding: 15,
    marginRight: 12,
    shadowColor: Colors.black,
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  stockHeader: {
    marginBottom: 10,
  },
  stockSymbol: {
    fontSize: 18,
    fontWeight: 'bold',
    color: Colors.secondary,
  },
  stockName: {
    fontSize: 12,
    color: Colors.darkGray,
    marginTop: 2,
  },
  stockPriceContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    marginBottom: 5,
  },
  stockPrice: {
    fontSize: 16,
    fontWeight: '600',
    color: Colors.secondary,
  },
  priceChangeContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: 6,
    paddingVertical: 3,
    borderRadius: 4,
  },
  priceChange: {
    fontSize: 12,
    color: Colors.white,
    marginLeft: 2,
  },
  stockVolume: {
    fontSize: 12,
    color: Colors.darkGray,
  },
});

export default HomeScreen;