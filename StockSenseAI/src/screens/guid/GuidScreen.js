import React, { useState } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  TouchableOpacity,
  SafeAreaView,
  Platform,
} from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { useNavigation } from '@react-navigation/native';

const GuidScreen = () => {
  const navigation = useNavigation();
  const [selectedSection, setSelectedSection] = useState(null);

  const sections = [
    {
      id: 'getting-started',
      title: 'Getting Started',
      content: 'Learn the basics of using StockSenseAI and how to navigate through the app effectively.',
      screen: 'GettingStarted',
    },
    {
      id: 'portfolio-management',
      title: 'Portfolio Management',
      content: 'Understand how to track and manage your investment portfolio, including adding stocks and monitoring performance.',
      screen: 'GettingStarted',
    },
    {
      id: 'market-analysis',
      title: 'Market Analysis',
      content: 'Explore the various tools and features available for analyzing market trends and making informed investment decisions.',
      screen: 'GettingStarted',
    },
    {
      id: 'news-feed',
      title: 'News Feed',
      content: 'Stay updated with the latest market news and how to use the news feed to make better investment choices.',
      screen: 'GettingStarted',
    },
    {
      id: 'settings',
      title: 'Settings & Preferences',
      content: 'Customize your app experience and manage your account settings.',
      screen: 'GettingStarted',
    },
  ];

  const handleSectionPress = (section) => {
    navigation.navigate(section.screen, { sectionId: section.id });
  };

  const renderSection = (section) => (
    <TouchableOpacity
      key={section.id}
      style={[
        styles.sectionCard,
        selectedSection === section.id && styles.selectedSection,
      ]}
      onPress={() => handleSectionPress(section)}
    >
      <View style={styles.sectionHeader}>
        <Text style={styles.sectionTitle}>{section.title}</Text>
        <Ionicons
          name="chevron-forward"
          size={24}
          color="#007AFF"
        />
      </View>
      <Text style={styles.sectionContent}>{section.content}</Text>
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
        <Text style={styles.headerTitle}>User Guide</Text>
      </View>

      <ScrollView 
        style={styles.content}
        showsVerticalScrollIndicator={false}
        contentContainerStyle={styles.scrollContent}
      >
        <Text style={styles.welcomeText}>
          Welcome to StockSenseAI! This guide will help you make the most of our app.
        </Text>
        
        {sections.map(renderSection)}
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
  welcomeText: {
    fontSize: 16,
    color: '#666666',
    marginBottom: 24,
    lineHeight: 24,
  },
  sectionCard: {
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
  selectedSection: {
    borderColor: '#007AFF',
    borderWidth: 1,
  },
  sectionHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 8,
  },
  sectionTitle: {
    fontSize: 18,
    fontWeight: '600',
    color: '#333333',
  },
  sectionContent: {
    fontSize: 16,
    color: '#666666',
    lineHeight: 24,
  },
});

export default GuidScreen; 