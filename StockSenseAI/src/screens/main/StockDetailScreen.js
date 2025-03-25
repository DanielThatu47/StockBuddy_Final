import React, { useState, useEffect } from 'react';
import { 
  View, 
  Text, 
  StyleSheet, 
  ScrollView, 
  TouchableOpacity, 
  ActivityIndicator 
} from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import Header from '../../components/Header';
import Colors from '../../constants/colors';
import FinnhubService from '../../services/FinnhubService';

const StockDetailScreen = ({ route, navigation }) => {
  const { symbol } = route.params;
  
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [companyProfile, setCompanyProfile] = useState(null);
  const [quoteData, setQuoteData] = useState(null);
  const [selectedTimeframe, setSelectedTimeframe] = useState('oneDay');
  const [chartData, setChartData] = useState([]);
  
  const timeframes = FinnhubService.getTimePeriods();
  
  // Fetch stock data on component mount
  useEffect(() => {
    const fetchStockData = async () => {
      setLoading(true);
      setError('');
      
      try {
        // Fetch company profile and quote data in parallel
        const [profileResponse, quoteResponse] = await Promise.all([
          FinnhubService.getCompanyProfile(symbol),
          FinnhubService.getQuote(symbol)
        ]);
        
        setCompanyProfile(profileResponse);
        setQuoteData(quoteResponse);
        
        // Load chart data for the selected timeframe
        await loadChartData(selectedTimeframe);
      } catch (err) {
        setError('Failed to load stock data. Please try again.');
        console.error(err);
      } finally {
        setLoading(false);
      }
    };
    
    fetchStockData();
  }, [symbol]);
  
  // Load chart data for a specific timeframe
  const loadChartData = async (timeframeKey) => {
    try {
      const timeframe = timeframes[timeframeKey];
      const historicalData = await FinnhubService.getHistoricalData(
        symbol,
        timeframe.resolution,
        timeframe.from,
        timeframe.to
      );
      
      const formattedData = FinnhubService.formatCandleData(historicalData);
      const lineChartData = FinnhubService.formatForLineChart(formattedData);
      
      setChartData(lineChartData);
      setSelectedTimeframe(timeframeKey);
    } catch (err) {
      console.error('Error loading chart data:', err);
      setError('Failed to load chart data');
    }
  };
  
  // Calculate price change and percent change
  const getPriceChange = () => {
    if (!quoteData) return { change: 0, percent: 0 };
    
    const change = quoteData.c - quoteData.pc;
    const percentChange = (change / quoteData.pc) * 100;
    
    return {
      change: change.toFixed(2),
      percent: percentChange.toFixed(2)
    };
  };
  
  const priceChange = getPriceChange();
  const isPositive = parseFloat(priceChange.change) >= 0;
  
  // Render timeframe selector buttons
  const renderTimeframeButtons = () => {
    return Object.keys(timeframes).map((key) => (
      <TouchableOpacity
        key={key}
        style={[
          styles.timeframeButton,
          selectedTimeframe === key && styles.selectedTimeframeButton
        ]}
        onPress={() => loadChartData(key)}
      >
        <Text
          style={[
            styles.timeframeButtonText,
            selectedTimeframe === key && styles.selectedTimeframeButtonText
          ]}
        >
          {timeframes[key].label}
        </Text>
      </TouchableOpacity>
    ));
  };
  
  if (loading) {
    return (
      <View style={styles.container}>
        <Header
          title={symbol}
          showBackButton={true}
          onBackPress={() => navigation.goBack()}
        />
        <View style={styles.loadingContainer}>
          <ActivityIndicator size="large" color={Colors.primary} />
          <Text style={styles.loadingText}>Loading stock data...</Text>
        </View>
      </View>
    );
  }
  
  if (error) {
    return (
      <View style={styles.container}>
        <Header
          title={symbol}
          showBackButton={true}
          onBackPress={() => navigation.goBack()}
        />
        <View style={styles.errorContainer}>
          <Ionicons name="alert-circle-outline" size={60} color={Colors.error} />
          <Text style={styles.errorText}>{error}</Text>
          <TouchableOpacity
            style={styles.retryButton}
            onPress={() => loadChartData(selectedTimeframe)}
          >
            <Text style={styles.retryButtonText}>Retry</Text>
          </TouchableOpacity>
        </View>
      </View>
    );
  }
  
  return (
    <View style={styles.container}>
      <Header
        title={companyProfile?.name || symbol}
        showBackButton={true}
        onBackPress={() => navigation.goBack()}
      />
      
      <ScrollView style={styles.scrollContainer}>
        {/* Stock price section */}
        <View style={styles.priceSection}>
          <Text style={styles.currentPrice}>${quoteData?.c.toFixed(2) || '0.00'}</Text>
          <View style={styles.priceChangeContainer}>
            <Ionicons
              name={isPositive ? 'caret-up' : 'caret-down'}
              size={16}
              color={isPositive ? Colors.success : Colors.error}
            />
            <Text
              style={[
                styles.priceChangeText,
                { color: isPositive ? Colors.success : Colors.error }
              ]}
            >
              ${Math.abs(priceChange.change)} ({Math.abs(priceChange.percent)}%)
            </Text>
          </View>
        </View>
        
        {/* Chart section - In a real app, you would render a chart component here */}
        <View style={styles.chartContainer}>
          <View style={styles.chartPlaceholder}>
            <Text style={styles.chartPlaceholderText}>
              Chart would be displayed here
            </Text>
            <Text style={styles.chartPlaceholderSubtext}>
              Data points: {chartData.length}
            </Text>
          </View>
          
          <View style={styles.timeframeButtonsContainer}>
            {renderTimeframeButtons()}
          </View>
        </View>
        
        {/* Company info section */}
        {companyProfile && (
          <View style={styles.companyInfoSection}>
            <Text style={styles.sectionTitle}>Company Information</Text>
            
            <View style={styles.infoRow}>
              <Text style={styles.infoLabel}>Symbol</Text>
              <Text style={styles.infoValue}>{companyProfile.ticker}</Text>
            </View>
            
            <View style={styles.infoRow}>
              <Text style={styles.infoLabel}>Industry</Text>
              <Text style={styles.infoValue}>{companyProfile.finnhubIndustry}</Text>
            </View>
            
            <View style={styles.infoRow}>
              <Text style={styles.infoLabel}>Market Cap</Text>
              <Text style={styles.infoValue}>
                ${(companyProfile.marketCapitalization / 1000).toFixed(2)}B
              </Text>
            </View>
            
            <View style={styles.infoRow}>
              <Text style={styles.infoLabel}>Country</Text>
              <Text style={styles.infoValue}>{companyProfile.country}</Text>
            </View>
            
            <View style={styles.infoRow}>
              <Text style={styles.infoLabel}>Exchange</Text>
              <Text style={styles.infoValue}>{companyProfile.exchange}</Text>
            </View>
            
            <View style={styles.infoRow}>
              <Text style={styles.infoLabel}>IPO Date</Text>
              <Text style={styles.infoValue}>{companyProfile.ipo}</Text>
            </View>
            
            <View style={styles.infoRow}>
              <Text style={styles.infoLabel}>Website</Text>
              <Text style={[styles.infoValue, styles.linkText]}>{companyProfile.weburl}</Text>
            </View>
          </View>
        )}
        
        {/* Trading stats section */}
        {quoteData && (
          <View style={styles.tradingStatsSection}>
            <Text style={styles.sectionTitle}>Trading Stats</Text>
            
            <View style={styles.statsGrid}>
              <View style={styles.statItem}>
                <Text style={styles.statLabel}>Open</Text>
                <Text style={styles.statValue}>${quoteData.o.toFixed(2)}</Text>
              </View>
              
              <View style={styles.statItem}>
                <Text style={styles.statLabel}>High</Text>
                <Text style={styles.statValue}>${quoteData.h.toFixed(2)}</Text>
              </View>
              
              <View style={styles.statItem}>
                <Text style={styles.statLabel}>Low</Text>
                <Text style={styles.statValue}>${quoteData.l.toFixed(2)}</Text>
              </View>
              
              <View style={styles.statItem}>
                <Text style={styles.statLabel}>Prev Close</Text>
                <Text style={styles.statValue}>${quoteData.pc.toFixed(2)}</Text>
              </View>
            </View>
          </View>
        )}
      </ScrollView>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: Colors.background,
  },
  scrollContainer: {
    flex: 1,
  },
  loadingContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  loadingText: {
    marginTop: 10,
    fontSize: 16,
    color: Colors.secondary,
  },
  errorContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    padding: 20,
  },
  errorText: {
    marginTop: 10,
    fontSize: 16,
    color: Colors.secondary,
    textAlign: 'center',
  },
  retryButton: {
    marginTop: 20,
    backgroundColor: Colors.primary,
    paddingVertical: 10,
    paddingHorizontal: 20,
    borderRadius: 8,
  },
  retryButtonText: {
    color: Colors.white,
    fontWeight: '600',
  },
  priceSection: {
    padding: 20,
  },
  currentPrice: {
    fontSize: 32,
    fontWeight: 'bold',
    color: Colors.secondary,
  },
  priceChangeContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    marginTop: 5,
  },
  priceChangeText: {
    fontSize: 16,
    marginLeft: 5,
  },
  chartContainer: {
    backgroundColor: Colors.white,
    marginHorizontal: 15,
    borderRadius: 15,
    padding: 15,
    shadowColor: Colors.black,
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
    marginBottom: 20,
  },
  chartPlaceholder: {
    height: 200,
    backgroundColor: Colors.lightGray,
    borderRadius: 8,
    justifyContent: 'center',
    alignItems: 'center',
  },
  chartPlaceholderText: {
    fontSize: 16,
    fontWeight: '500',
    color: Colors.darkGray,
  },
  chartPlaceholderSubtext: {
    fontSize: 14,
    color: Colors.darkGray,
    marginTop: 5,
  },
  timeframeButtonsContainer: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginTop: 15,
  },
  timeframeButton: {
    paddingVertical: 6,
    paddingHorizontal: 12,
    borderRadius: 20,
    backgroundColor: Colors.lightGray,
  },
  selectedTimeframeButton: {
    backgroundColor: Colors.primary,
  },
  timeframeButtonText: {
    fontSize: 12,
    fontWeight: '500',
    color: Colors.darkGray,
  },
  selectedTimeframeButtonText: {
    color: Colors.white,
  },
  companyInfoSection: {
    backgroundColor: Colors.white,
    marginHorizontal: 15,
    borderRadius: 15,
    padding: 15,
    shadowColor: Colors.black,
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
    marginBottom: 20,
  },
  sectionTitle: {
    fontSize: 18,
    fontWeight: '600',
    color: Colors.secondary,
    marginBottom: 15,
  },
  infoRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    paddingVertical: 10,
    borderBottomWidth: 1,
    borderBottomColor: Colors.lightGray,
  },
  infoLabel: {
    fontSize: 14,
    color: Colors.darkGray,
  },
  infoValue: {
    fontSize: 14,
    color: Colors.secondary,
    fontWeight: '500',
  },
  linkText: {
    color: Colors.primary,
  },
  tradingStatsSection: {
    backgroundColor: Colors.white,
    marginHorizontal: 15,
    borderRadius: 15,
    padding: 15,
    shadowColor: Colors.black,
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
    marginBottom: 20,
  },
  statsGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
  },
  statItem: {
    width: '50%',
    paddingVertical: 10,
    paddingRight: 10,
  },
  statLabel: {
    fontSize: 14,
    color: Colors.darkGray,
  },
  statValue: {
    fontSize: 16,
    color: Colors.secondary,
    fontWeight: '500',
    marginTop: 5,
  },
});

export default StockDetailScreen; 