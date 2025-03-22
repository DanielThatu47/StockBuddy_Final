import React, { useState } from 'react';
import { View, Text, ScrollView, StyleSheet, TouchableOpacity } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import Header from '../../components/Header';
import SectionHeader from '../../components/SectionHeader';
import StockCard from '../../components/StockCard';
import StockChart from '../../components/StockChart';
import Colors from '../../constants/colors';

const PortfolioScreen = ({ navigation }) => {
  const [chartPeriod, setChartPeriod] = useState('1D');
  
  // Mock data for portfolio value
  const portfolioValue = 15420.75;
  const portfolioChange = 3.2;
  
  // Mock data for chart
  const chartData = {
    labels: ['9AM', '11AM', '1PM', '3PM', '5PM'],
    values: [142.5, 143.1, 144.7, 143.8, 145.2],
  };
  
  // Mock data for portfolio stocks
  const portfolioStocks = [
    { symbol: 'AAPL', companyName: 'Apple Inc.', price: 145.20, priceChange: 1.8, shares: 10 },
    { symbol: 'MSFT', companyName: 'Microsoft Corp.', price: 305.75, priceChange: -0.8, shares: 5 },
    { symbol: 'GOOGL', companyName: 'Alphabet Inc.', price: 2830.45, priceChange: 2.3, shares: 2 },
  ];
  
  // Calculate portfolio statistics
  const totalInvested = 12500;
  const totalGain = portfolioValue - totalInvested;
  const totalGainPercentage = (totalGain / totalInvested) * 100;

  return (
    <View style={styles.container}>
      <Header title="Portfolio" subtitle="Track your investments" />
      
      <ScrollView style={styles.content} showsVerticalScrollIndicator={false}>
        <View style={styles.portfolioSummary}>
          <Text style={styles.portfolioValue}>${portfolioValue.toFixed(2)}</Text>
          <View style={styles.changeContainer}>
            <Ionicons 
              name={portfolioChange >= 0 ? "arrow-up" : "arrow-down"} 
              size={16} 
              color={portfolioChange >= 0 ? Colors.positive : Colors.negative} 
            />
            <Text style={[
              styles.changeText, 
              { color: portfolioChange >= 0 ? Colors.positive : Colors.negative }
            ]}>
              {portfolioChange >= 0 ? '+' : ''}{portfolioChange.toFixed(2)}%
            </Text>
          </View>
        </View>
        
        <StockChart 
          data={chartData} 
          period={chartPeriod}
          onPeriodChange={setChartPeriod}
        />
        
        <View style={styles.statsContainer}>
          <View style={styles.statItem}>
            <Text style={styles.statLabel}>Invested</Text>
            <Text style={styles.statValue}>${totalInvested.toFixed(2)}</Text>
          </View>
          <View style={styles.statItem}>
            <Text style={styles.statLabel}>Gain/Loss</Text>
            <Text style={[
              styles.statValue, 
              { color: totalGain >= 0 ? Colors.positive : Colors.negative }
            ]}>
              {totalGain >= 0 ? '+' : ''}${Math.abs(totalGain).toFixed(2)}
            </Text>
          </View>
          <View style={styles.statItem}>
            <Text style={styles.statLabel}>Return</Text>
            <Text style={[
              styles.statValue, 
              { color: totalGainPercentage >= 0 ? Colors.positive : Colors.negative }
            ]}>
              {totalGainPercentage >= 0 ? '+' : ''}{totalGainPercentage.toFixed(2)}%
            </Text>
          </View>
        </View>
        
        <SectionHeader 
          title="Your Stocks" 
          actionText="See All"
          onActionPress={() => {}}
        />
        
        {portfolioStocks.map((stock, index) => (
          <View key={index} style={styles.portfolioStockContainer}>
            <StockCard
              symbol={stock.symbol}
              companyName={stock.companyName}
              price={stock.price}
              priceChange={stock.priceChange}
              onPress={() => navigation.navigate('StockDetail', { symbol: stock.symbol })}
            />
            <View style={styles.sharesContainer}>
              <Text style={styles.sharesLabel}>Shares</Text>
              <Text style={styles.sharesValue}>{stock.shares}</Text>
              <Text style={styles.sharesValue}>${(stock.price * stock.shares).toFixed(2)}</Text>
            </View>
          </View>
        ))}
      </ScrollView>
      
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
  },
  portfolioSummary: {
    marginTop: 20,
    alignItems: 'center',
  },
  portfolioValue: {
    fontSize: 32,
    fontWeight: 'bold',
    color: Colors.secondary,
  },
  changeContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    marginTop: 8,
  },
  changeText: {
    fontSize: 16,
    fontWeight: '600',
    marginLeft: 4,
  },
  statsContainer: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginVertical: 16,
    backgroundColor: Colors.white,
    borderRadius: 8,
    padding: 16,
    shadowColor: Colors.black,
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 2,
  },
  statItem: {
    alignItems: 'center',
  },
  statLabel: {
    fontSize: 12,
    color: Colors.darkGray,
    marginBottom: 4,
  },
  statValue: {
    fontSize: 16,
    fontWeight: 'bold',
    color: Colors.secondary,
  },
  portfolioStockContainer: {
    marginBottom: 8,
  },
  sharesContainer: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    paddingHorizontal: 16,
    paddingVertical: 8,
    backgroundColor: Colors.primaryLight,
    borderBottomLeftRadius: 8,
    borderBottomRightRadius: 8,
    marginTop: -8,
  },
  sharesLabel: {
    fontSize: 12,
    color: Colors.darkGray,
  },
  sharesValue: {
    fontSize: 12,
    fontWeight: '600',
    color: Colors.secondary,
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

export default PortfolioScreen; 