import React from 'react';
import { View, Text, StyleSheet, TouchableOpacity } from 'react-native';
import Colors from '../constants/colors';

const StockCard = ({ 
  symbol, 
  companyName = '', 
  price, 
  priceChange, 
  onPress, 
  containerStyle = {} 
}) => {
  const isPositive = priceChange >= 0;
  const priceChangeColor = isPositive ? Colors.positive : Colors.negative;
  const priceChangeSymbol = isPositive ? '+' : '';

  return (
    <TouchableOpacity 
      style={[styles.container, containerStyle]} 
      onPress={onPress}
      activeOpacity={0.7}
    >
      <View style={styles.leftContent}>
        <Text style={styles.symbolText}>{symbol}</Text>
        {companyName ? <Text style={styles.companyText}>{companyName}</Text> : null}
      </View>
      <View style={styles.rightContent}>
        <Text style={styles.priceText}>${parseFloat(price).toFixed(2)}</Text>
        <Text style={[styles.changeText, { color: priceChangeColor }]}>
          {priceChangeSymbol}{parseFloat(priceChange).toFixed(1)}%
        </Text>
      </View>
    </TouchableOpacity>
  );
};

const styles = StyleSheet.create({
  container: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    backgroundColor: Colors.cardBackground,
    borderRadius: 8,
    padding: 16,
    marginVertical: 8,
    shadowColor: Colors.black,
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 2,
  },
  leftContent: {
    flex: 1,
  },
  rightContent: {
    alignItems: 'flex-end',
  },
  symbolText: {
    fontSize: 18,
    fontWeight: 'bold',
    color: Colors.secondary,
  },
  companyText: {
    fontSize: 14,
    color: Colors.darkGray,
    marginTop: 4,
  },
  priceText: {
    fontSize: 18,
    fontWeight: 'bold',
    color: Colors.secondary,
  },
  changeText: {
    fontSize: 14,
    fontWeight: '600',
    marginTop: 4,
  },
});

export default StockCard; 