import React from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  TouchableOpacity,
  SafeAreaView,
  Image,
  Platform,
} from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { useNavigation, useRoute } from '@react-navigation/native';

const GettingStartedScreen = () => {
  const navigation = useNavigation();
  const route = useRoute();
  const { sectionId } = route.params || { sectionId: 'getting-started' };

  const getSectionContent = () => {
    switch (sectionId) {
      case 'getting-started':
        return {
          title: 'Getting Started',
          steps: [
            {
              id: 1,
              title: 'Welcome to StockSenseAI',
              content: 'StockSenseAI is your intelligent companion for stock market analysis and portfolio management. Let\'s get you started with the basics.',
              icon: 'rocket-outline',
            },
            {
              id: 2,
              title: 'Create Your Account',
              content: 'Sign up for a free account to access all features. You\'ll need to provide your email and create a secure password.',
              icon: 'person-add-outline',
            },
            {
              id: 3,
              title: 'Set Up Your Profile',
              content: 'Customize your profile with your investment preferences and risk tolerance level to get personalized recommendations.',
              icon: 'settings-outline',
            },
            {
              id: 4,
              title: 'Explore the Dashboard',
              content: 'The main dashboard shows your portfolio summary, watchlist, and market overview. Take time to familiarize yourself with the layout.',
              icon: 'apps-outline',
            },
            {
              id: 5,
              title: 'Add Your First Stock',
              content: 'Start by adding stocks to your watchlist. Use the search function to find stocks and tap the + button to add them.',
              icon: 'add-circle-outline',
            },
          ],
        };
      case 'portfolio-management':
        return {
          title: 'Portfolio Management',
          steps: [
            {
              id: 1,
              title: 'View Your Portfolio',
              content: 'Access your portfolio from the main tab bar. Here you can see all your investments and their current performance.',
              icon: 'pie-chart-outline',
            },
            {
              id: 2,
              title: 'Add Investments',
              content: 'Track your investments by adding them to your portfolio. Include purchase price, quantity, and date for accurate tracking.',
              icon: 'add-circle-outline',
            },
            {
              id: 3,
              title: 'Monitor Performance',
              content: 'View detailed performance metrics including profit/loss, return on investment, and portfolio allocation.',
              icon: 'trending-up-outline',
            },
            {
              id: 4,
              title: 'Portfolio Analytics',
              content: 'Use advanced analytics to understand your portfolio\'s risk profile and diversification.',
              icon: 'analytics-outline',
            },
          ],
        };
      case 'market-analysis':
        return {
          title: 'Market Analysis',
          steps: [
            {
              id: 1,
              title: 'Market Overview',
              content: 'Get a comprehensive view of market trends, indices, and sector performance.',
              icon: 'globe-outline',
            },
            {
              id: 2,
              title: 'Stock Analysis',
              content: 'Access detailed stock information including price history, technical indicators, and fundamental data.',
              icon: 'bar-chart-outline',
            },
            {
              id: 3,
              title: 'Technical Indicators',
              content: 'Use various technical indicators to analyze price trends and make informed decisions.',
              icon: 'trending-up-outline',
            },
            {
              id: 4,
              title: 'Market News',
              content: 'Stay updated with relevant news that might impact your investments.',
              icon: 'newspaper-outline',
            },
          ],
        };
      case 'news-feed':
        return {
          title: 'News Feed',
          steps: [
            {
              id: 1,
              title: 'Access News',
              content: 'Find the news tab in the bottom navigation to access the latest market news.',
              icon: 'newspaper-outline',
            },
            {
              id: 2,
              title: 'Customize Feed',
              content: 'Set up your news preferences to see relevant information about your watchlist and portfolio.',
              icon: 'settings-outline',
            },
            {
              id: 3,
              title: 'Save Articles',
              content: 'Bookmark important articles for future reference.',
              icon: 'bookmark-outline',
            },
            {
              id: 4,
              title: 'Share News',
              content: 'Share relevant news with other investors or save for later.',
              icon: 'share-outline',
            },
          ],
        };
      case 'settings':
        return {
          title: 'Settings & Preferences',
          steps: [
            {
              id: 1,
              title: 'Account Settings',
              content: 'Manage your account information, password, and personal details.',
              icon: 'person-outline',
            },
            {
              id: 2,
              title: 'Notification Preferences',
              content: 'Customize which notifications you want to receive and how often.',
              icon: 'notifications-outline',
            },
            {
              id: 3,
              title: 'Display Settings',
              content: 'Adjust the app\'s appearance and layout to your preferences.',
              icon: 'color-palette-outline',
            },
            {
              id: 4,
              title: 'Privacy Settings',
              content: 'Control your privacy settings and data sharing preferences.',
              icon: 'shield-outline',
            },
          ],
        };
      default:
        return {
          title: 'Getting Started',
          steps: [],
        };
    }
  };

  const sectionContent = getSectionContent();

  return (
    <SafeAreaView style={styles.container}>
      <View style={styles.header}>
        <TouchableOpacity
          style={styles.backButton}
          onPress={() => navigation.goBack()}
        >
          <Ionicons name="arrow-back" size={24} color="#007AFF" />
        </TouchableOpacity>
        <Text style={styles.headerTitle}>{sectionContent.title}</Text>
      </View>

      <ScrollView 
        style={styles.content}
        showsVerticalScrollIndicator={false}
        contentContainerStyle={styles.scrollContent}
      >
        <View style={styles.introSection}>
          <Text style={styles.introTitle}>Quick Start Guide</Text>
          <Text style={styles.introText}>
            Follow these simple steps to make the most of {sectionContent.title.toLowerCase()}.
          </Text>
        </View>

        {sectionContent.steps.map((step) => (
          <View key={step.id} style={styles.stepCard}>
            <View style={styles.stepHeader}>
              <View style={styles.stepIconContainer}>
                <Ionicons name={step.icon} size={24} color="#007AFF" />
              </View>
              <Text style={styles.stepNumber}>Step {step.id}</Text>
            </View>
            <Text style={styles.stepTitle}>{step.title}</Text>
            <Text style={styles.stepContent}>{step.content}</Text>
          </View>
        ))}

        <View style={styles.tipsSection}>
          <Text style={styles.tipsTitle}>Pro Tips</Text>
          <View style={styles.tipItem}>
            <Ionicons name="bulb-outline" size={20} color="#FFD700" />
            <Text style={styles.tipText}>
              Take time to explore all features before making your first trade
            </Text>
          </View>
          <View style={styles.tipItem}>
            <Ionicons name="bulb-outline" size={20} color="#FFD700" />
            <Text style={styles.tipText}>
              Use the practice mode to familiarize yourself with trading features
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
  introSection: {
    backgroundColor: '#FFFFFF',
    borderRadius: 12,
    padding: 20,
    marginBottom: 20,
    shadowColor: '#000',
    shadowOffset: {
      width: 0,
      height: 2,
    },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  introTitle: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#333333',
    marginBottom: 8,
  },
  introText: {
    fontSize: 16,
    color: '#666666',
    lineHeight: 24,
  },
  stepCard: {
    backgroundColor: '#FFFFFF',
    borderRadius: 12,
    padding: 16,
    marginBottom: 16,
    shadowColor: '#000',
    shadowOffset: {
      width: 0,
      height: 2,
    },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  stepHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 12,
  },
  stepIconContainer: {
    width: 40,
    height: 40,
    borderRadius: 20,
    backgroundColor: '#F0F8FF',
    justifyContent: 'center',
    alignItems: 'center',
    marginRight: 12,
  },
  stepNumber: {
    fontSize: 14,
    fontWeight: '600',
    color: '#007AFF',
  },
  stepTitle: {
    fontSize: 18,
    fontWeight: '600',
    color: '#333333',
    marginBottom: 8,
  },
  stepContent: {
    fontSize: 16,
    color: '#666666',
    lineHeight: 24,
  },
  tipsSection: {
    backgroundColor: '#FFFFFF',
    borderRadius: 12,
    padding: 20,
    marginTop: 20,
    marginBottom: 16,
    shadowColor: '#000',
    shadowOffset: {
      width: 0,
      height: 2,
    },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  tipsTitle: {
    fontSize: 20,
    fontWeight: 'bold',
    color: '#333333',
    marginBottom: 16,
  },
  tipItem: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 12,
  },
  tipText: {
    fontSize: 16,
    color: '#666666',
    marginLeft: 12,
    flex: 1,
    lineHeight: 24,
  },
});

export default GettingStartedScreen; 