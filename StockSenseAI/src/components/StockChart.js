import React from 'react';
import { View, Text, StyleSheet, Dimensions } from 'react-native';
import { LineChart } from 'react-native-chart-kit';
import Colors from '../constants/colors';

const { width } = Dimensions.get('window');

const StockChart = ({ data, period = '1D', onPeriodChange }) => {
  const chartConfig = {
    backgroundGradientFrom: Colors.white,
    backgroundGradientTo: Colors.white,
    decimalPlaces: 2,
    color: () => Colors.chartBlue,
    labelColor: () => Colors.darkGray,
    style: {
      borderRadius: 16,
    },
    propsForDots: {
      r: '5',
      strokeWidth: '2',
      stroke: Colors.chartBlue,
    },
    propsForBackgroundLines: {
      strokeDasharray: '',
      strokeWidth: 0.5,
      stroke: Colors.gray,
    },
  };

  return (
    <View style={styles.container}>
      <View style={styles.periodSelector}>
        {['1D', '1W', '1M', '3M', '1Y', 'All'].map((p) => (
          <View
            key={p}
            style={[
              styles.periodButton,
              period === p && styles.activePeriodButton,
            ]}
          >
            <Text
              style={[
                styles.periodText,
                period === p && styles.activePeriodText,
              ]}
              onPress={() => onPeriodChange(p)}
            >
              {p}
            </Text>
          </View>
        ))}
      </View>

      <LineChart
        data={{
          labels: data.labels,
          datasets: [
            {
              data: data.values,
              color: (opacity = 1) => Colors.primary,
              strokeWidth: 2,
            },
          ],
        }}
        width={width - 40}
        height={220}
        chartConfig={chartConfig}
        bezier
        withInnerLines={true}
        withOuterLines={false}
        withVerticalLines={false}
        withHorizontalLines={true}
        withDots={true}
        withShadow={false}
        style={styles.chart}
      />
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    backgroundColor: Colors.white,
    borderRadius: 16,
    padding: 16,
    marginVertical: 8,
  },
  periodSelector: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: 16,
  },
  periodButton: {
    paddingVertical: 8,
    paddingHorizontal: 12,
    borderRadius: 20,
  },
  activePeriodButton: {
    backgroundColor: Colors.primary,
  },
  periodText: {
    fontSize: 14,
    color: Colors.darkGray,
    fontWeight: '500',
  },
  activePeriodText: {
    color: Colors.white,
    fontWeight: '600',
  },
  chart: {
    marginVertical: 8,
    borderRadius: 16,
  },
});

export default StockChart; 