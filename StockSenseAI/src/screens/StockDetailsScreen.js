import React, { useEffect, useState } from 'react';
import { View, Text, StyleSheet } from 'react-native';
import { LineChart } from 'react-native-chart-kit';
import { fetchHistoricalData } from '../services/predictionService'; // New service function

const StockDetailsScreen = ({ route }) => {
  const { symbol } = route.params;
  const [historicalData, setHistoricalData] = useState([]);

  useEffect(() => {
    const loadHistoricalData = async () => {
      const data = await fetchHistoricalData(symbol);
      setHistoricalData(data);
    };
    loadHistoricalData();
  }, [symbol]);

  return (
    <View style={styles.container}>
      <Text style={styles.title}>{symbol} Historical Data</Text>
      <LineChart
        data={{
          labels: historicalData.map(data => data.date),
          datasets: [{
            data: historicalData.map(data => data.close),
          }],
        }}
        width={400}
        height={220}
        chartConfig={{
          backgroundColor: '#ffffff',
          backgroundGradientFrom: '#ffffff',
          backgroundGradientTo: '#ffffff',
          decimalPlaces: 2,
          color: (opacity = 1) => `rgba(0, 122, 255, ${opacity})`,
          style: {
            borderRadius: 16,
          },
        }}
        bezier
      />
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    padding: 20,
  },
  title: {
    fontSize: 24,
    fontWeight: 'bold',
  },
});

export default StockDetailsScreen; 