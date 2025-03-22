import React, { useState } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  TouchableOpacity,
  SafeAreaView,
  Platform,
  Animated,
} from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { useNavigation } from '@react-navigation/native';

const FAQScreen = () => {
  const navigation = useNavigation();
  const [expandedItems, setExpandedItems] = useState({});

  const faqItems = [
    {
      id: '1',
      question: 'How do I get started with StockSenseAI?',
      answer: 'Getting started is easy! Simply download the app, create an account, and complete your profile. You can then start tracking your investments, analyzing stocks, and receiving personalized recommendations.',
    },
    {
      id: '2',
      question: 'What features are available in the free version?',
      answer: 'The free version includes basic portfolio tracking, market overview, and news feed. You can track up to 10 stocks in your watchlist and access basic market analysis tools.',
    },
    {
      id: '3',
      question: 'How do I add stocks to my portfolio?',
      answer: 'To add stocks, go to the Search tab, find the stock you want to add, and tap the + button. You can then enter the purchase details including price, quantity, and date.',
    },
    {
      id: '4',
      question: 'What payment methods are accepted?',
      answer: 'We accept all major credit cards, PayPal, and bank transfers for premium subscriptions. All payments are processed securely through our payment partners.',
    },
    {
      id: '5',
      question: 'How do I contact customer support?',
      answer: 'You can reach our customer support team through the Help & Support section in your profile. We typically respond within 24 hours during business days.',
    },
  ];

  const toggleItem = (id) => {
    setExpandedItems(prev => ({
      ...prev,
      [id]: !prev[id]
    }));
  };

  const renderFAQItem = (item) => {
    const isExpanded = expandedItems[item.id];
    const rotateAnimation = new Animated.Value(isExpanded ? 1 : 0);

    React.useEffect(() => {
      Animated.timing(rotateAnimation, {
        toValue: isExpanded ? 1 : 0,
        duration: 300,
        useNativeDriver: true,
      }).start();
    }, [isExpanded]);

    const rotate = rotateAnimation.interpolate({
      inputRange: [0, 1],
      outputRange: ['0deg', '90deg'],
    });

    return (
      <View key={item.id} style={styles.faqItem}>
        <TouchableOpacity
          style={styles.faqHeader}
          onPress={() => toggleItem(item.id)}
        >
          <Text style={styles.questionText}>{item.question}</Text>
          <Animated.View style={{ transform: [{ rotate }] }}>
            <Ionicons name="chevron-forward" size={24} color="#007AFF" />
          </Animated.View>
        </TouchableOpacity>
        {isExpanded && (
          <View style={styles.answerContainer}>
            <Text style={styles.answerText}>{item.answer}</Text>
          </View>
        )}
      </View>
    );
  };

  return (
    <SafeAreaView style={styles.container}>
      <View style={styles.header}>
        <TouchableOpacity
          style={styles.backButton}
          onPress={() => navigation.goBack()}
        >
          <Ionicons name="arrow-back" size={24} color="#007AFF" />
        </TouchableOpacity>
        <Text style={styles.headerTitle}>Frequently Asked Questions</Text>
      </View>

      <ScrollView 
        style={styles.content}
        showsVerticalScrollIndicator={false}
        contentContainerStyle={styles.scrollContent}
      >
        <Text style={styles.introText}>
          Find answers to common questions about using StockSenseAI.
        </Text>

        {faqItems.map(renderFAQItem)}
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
  introText: {
    fontSize: 16,
    color: '#666666',
    marginBottom: 24,
    lineHeight: 24,
  },
  faqItem: {
    backgroundColor: '#FFFFFF',
    borderRadius: 12,
    marginBottom: 16,
    overflow: 'hidden',
    shadowColor: '#000',
    shadowOffset: {
      width: 0,
      height: 2,
    },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  faqHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: 16,
  },
  questionText: {
    fontSize: 16,
    fontWeight: '600',
    color: '#333333',
    flex: 1,
    marginRight: 16,
  },
  answerContainer: {
    padding: 16,
    paddingTop: 0,
    borderTopWidth: 1,
    borderTopColor: '#E0E0E0',
  },
  answerText: {
    fontSize: 16,
    color: '#666666',
    lineHeight: 24,
  },
});

export default FAQScreen; 
 