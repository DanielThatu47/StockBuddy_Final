import React, { useState } from 'react';
import { View, ScrollView, StyleSheet, TouchableOpacity } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import Header from '../components/Header';
import SearchInput from '../components/SearchInput';
import StockCard from '../components/StockCard';
import Colors from '../constants/colors';

const WatchlistScreen = ({ navigation }) => {
  const [searchQuery, setSearchQuery] = useState('');

  // Mock data for watchlist stocks
  const watchlistStocks = [
    { symbol: 'GOOGL', companyName: 'Alphabet Inc.', price: 2830.45, priceChange: 2.3 },
    { symbol: 'MSFT', companyName: 'Microsoft Corp.', price: 305.75, priceChange: -0.8 },
    { symbol: 'AMZN', companyName: 'Amazon.com Inc.', price: 3380.20, priceChange: -1.2 },
    { symbol: 'TSLA', companyName: 'Tesla Inc.', price: 750.35, priceChange: 3.5 },
    { symbol: 'FB', companyName: 'Meta Platforms Inc.', price: 325.45, priceChange: 0.7 },
    { symbol: 'NFLX', companyName: 'Netflix Inc.', price: 540.82, priceChange: -2.1 },
  ];

  return (
    <View style={styles.container}>
      <Header title="Watchlist" subtitle="Track your favorite stocks" />

      <View style={styles.content}>
        <SearchInput
          placeholder="Search stocks..."
          value={searchQuery}
          onChangeText={setSearchQuery}
          onSubmit={() => {}}
          style={styles.searchInput}
        />

        <ScrollView showsVerticalScrollIndicator={false}>
          {watchlistStocks.map((stock, index) => (
            <StockCard
              key={index}
              symbol={stock.symbol}
              companyName={stock.companyName}
              price={stock.price}
              priceChange={stock.priceChange}
              onPress={() => navigation.navigate('StockDetail', { symbol: stock.symbol })}
            />
          ))}
        </ScrollView>
      </View>

      <TouchableOpacity style={styles.addButton} onPress={() => {}}>
        <Ionicons name="add" size={28} color={Colors.white} />
      </TouchableOpacity>
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
    paddingTop: 20,
  },
  searchInput: {
    marginBottom: 16,
  },
  addButton: {
    position: 'absolute',
    bottom: 20,
    right: 20,
    width: 56,
    height: 56,
    borderRadius: 28,
    backgroundColor: Colors.primary,
    justifyContent: 'center',
    alignItems: 'center',
    shadowColor: Colors.black,
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.3,
    shadowRadius: 4,
    elevation: 5,
  },
});

export default WatchlistScreen;