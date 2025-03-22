import React, { useState } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  TouchableOpacity,
  SafeAreaView,
  Platform,
  Dimensions,
} from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { useNavigation } from '@react-navigation/native';
import { LineChart } from 'react-native-chart-kit';

const LiveChartScreen = () => {
  const navigation = useNavigation();
  const [selectedTimeframe, setSelectedTimeframe] = useState('1D');

  // Mock data for the chart
  const chartData = {
    labels: ['9:30', '10:00', '10:30', '11:00', '11:30', '12:00', '12:30', '13:00'],
    datasets: [
      {
        data: [150, 152, 151, 153, 155, 154, 156, 158],
      },
    ],
  };

  const timeframes = ['1D', '1W', '1M', '3M', '1Y', 'ALL'];

  const renderTimeframeButton = (timeframe) => (
    <TouchableOpacity
      key={timeframe}
      style={[
        styles.timeframeButton,
        selectedTimeframe === timeframe && styles.selectedTimeframe,
      ]}
      onPress={() => setSelectedTimeframe(timeframe)}
    >
      <Text
        style={[
          styles.timeframeText,
          selectedTimeframe === timeframe && styles.selectedTimeframeText,
        ]}
      >
        {timeframe}
      </Text>
    </TouchableOpacity>
  );

  return (
    <SafeAreaView style={styles.container}>
      <View style={styles.header}>
        <TouchableOpacity
          style={styles.backButton}
          onPress={() => navigation.goBack()}
        >
          <Ionicons name="arrow-back" size={24} color="#007AFF" />
        </TouchableOpacity>
        <Text style={styles.headerTitle}>Live Chart</Text>
      </View>

      <ScrollView 
        style={styles.content}
        showsVerticalScrollIndicator={false}
        contentContainerStyle={styles.scrollContent}
      >
        <View style={styles.chartContainer}>
          <View style={styles.stockInfo}>
            <Text style={styles.stockSymbol}>AAPL</Text>
            <Text style={styles.stockPrice}>$158.45</Text>
            <Text style={styles.priceChange}>+2.34 (1.50%)</Text>
          </View>

          <View style={styles.timeframeContainer}>
            {timeframes.map(renderTimeframeButton)}
          </View>

          <View style={styles.chartWrapper}>
            <LineChart
              data={chartData}
              width={Dimensions.get('window').width - 32}
              height={220}
              chartConfig={{
                backgroundColor: '#FFFFFF',
                backgroundGradientFrom: '#FFFFFF',
                backgroundGradientTo: '#FFFFFF',
                decimalPlaces: 2,
                color: (opacity = 1) => `rgba(0, 122, 255, ${opacity})`,
                style: {
                  borderRadius: 16,
                },
              }}
              bezier
              style={styles.chart}
            />
          </View>
        </View>

        <View style={styles.documentationSection}>
          <Text style={styles.sectionTitle}>Chart Documentation</Text>
          
          <View style={styles.docItem}>
            <Text style={styles.docTitle}>Understanding the Chart</Text>
            <Text style={styles.docContent}>
              The chart displays real-time price movements of the selected stock. The blue line represents the stock's price over time, and you can use different timeframes to analyze various periods.
            </Text>
          </View>

          <View style={styles.docItem}>
            <Text style={styles.docTitle}>Timeframe Selection</Text>
            <Text style={styles.docContent}>
              Choose different timeframes to view price movements over various periods:
              • 1D: Last 24 hours
              • 1W: Last week
              • 1M: Last month
              • 3M: Last 3 months
              • 1Y: Last year
              • ALL: All available data
            </Text>
          </View>

          <View style={styles.docItem}>
            <Text style={styles.docTitle}>Price Information</Text>
            <Text style={styles.docContent}>
              The current price, price change, and percentage change are displayed above the chart. Green indicates a positive change, while red indicates a negative change.
            </Text>
          </View>

          <View style={styles.docItem}>
            <Text style={styles.docTitle}>Interactive Features</Text>
            <Text style={styles.docContent}>
              • Tap and hold on the chart to see detailed price information
              • Pinch to zoom in/out
              • Swipe left/right to view different time periods
            </Text>
          </View>
        </View>
      </ScrollView>
    </SafeAreaView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#F5F5F5',
  },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    padding: 16,
    backgroundColor: '#FFFFFF',
    borderBottomWidth: 1,
    borderBottomColor: '#E0E0E0',
    ...Platform.select({
      ios: {
        paddingTop: 44,
      },
      android: {
        paddingTop: 16,
      },
    }),
  },
  backButton: {
    padding: 8,
    marginRight: 8,
  },
  headerTitle: {
    fontSize: 20,
    fontWeight: 'bold',
    color: '#333333',
  },
  content: {
    flex: 1,
  },
  scrollContent: {
    padding: 16,
    paddingBottom: 32,
  },
  chartContainer: {
    backgroundColor: '#FFFFFF',
    borderRadius: 12,
    padding: 16,
    marginBottom: 24,
    shadowColor: '#000',
    shadowOffset: {
      width: 0,
      height: 2,
    },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  stockInfo: {
    marginBottom: 16,
  },
  stockSymbol: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#333333',
    marginBottom: 4,
  },
  stockPrice: {
    fontSize: 20,
    color: '#333333',
    marginBottom: 4,
  },
  priceChange: {
    fontSize: 16,
    color: '#34C759',
  },
  timeframeContainer: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: 16,
  },
  timeframeButton: {
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 16,
    backgroundColor: '#F5F5F5',
  },
  selectedTimeframe: {
    backgroundColor: '#007AFF',
  },
  timeframeText: {
    fontSize: 14,
    color: '#666666',
  },
  selectedTimeframeText: {
    color: '#FFFFFF',
  },
  chartWrapper: {
    alignItems: 'center',
  },
  chart: {
    marginVertical: 8,
    borderRadius: 16,
  },
  documentationSection: {
    backgroundColor: '#FFFFFF',
    borderRadius: 12,
    padding: 20,
    shadowColor: '#000',
    shadowOffset: {
      width: 0,
      height: 2,
    },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  sectionTitle: {
    fontSize: 20,
    fontWeight: 'bold',
    color: '#333333',
    marginBottom: 16,
  },
  docItem: {
    marginBottom: 20,
  },
  docTitle: {
    fontSize: 16,
    fontWeight: '600',
    color: '#333333',
    marginBottom: 8,
  },
  docContent: {
    fontSize: 14,
    color: '#666666',
    lineHeight: 20,
  },
});

export default LiveChartScreen; 