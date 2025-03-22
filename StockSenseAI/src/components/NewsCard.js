import React from 'react';
import { View, Text, StyleSheet, TouchableOpacity } from 'react-native';
import Colors from '../constants/colors';

const NewsCard = ({ title, summary, time, sentiment, onPress }) => {
  const getSentimentColor = () => {
    switch (sentiment) {
      case 'Positive':
        return Colors.positive;
      case 'Negative':
        return Colors.negative;
      default:
        return Colors.neutral;
    }
  };

  return (
    <TouchableOpacity 
      style={styles.container} 
      onPress={onPress}
      activeOpacity={0.7}
    >
      <View style={styles.content}>
        <Text style={styles.title}>{title}</Text>
        <Text style={styles.summary} numberOfLines={2}>{summary}</Text>
        <View style={styles.footer}>
          <Text style={styles.time}>{time}</Text>
          <View style={[styles.sentimentBadge, { backgroundColor: getSentimentColor() + '20' }]}>
            <Text style={[styles.sentimentText, { color: getSentimentColor() }]}>
              {sentiment}
            </Text>
          </View>
        </View>
      </View>
    </TouchableOpacity>
  );
};

const styles = StyleSheet.create({
  container: {
    backgroundColor: Colors.white,
    borderRadius: 8,
    marginVertical: 8,
    shadowColor: Colors.black,
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 2,
    overflow: 'hidden',
  },
  content: {
    padding: 16,
  },
  title: {
    fontSize: 16,
    fontWeight: 'bold',
    color: Colors.secondary,
    marginBottom: 8,
  },
  summary: {
    fontSize: 14,
    color: Colors.darkGray,
    marginBottom: 12,
  },
  footer: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  time: {
    fontSize: 12,
    color: Colors.darkGray,
  },
  sentimentBadge: {
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderRadius: 12,
  },
  sentimentText: {
    fontSize: 12,
    fontWeight: '600',
  },
});

export default NewsCard; 