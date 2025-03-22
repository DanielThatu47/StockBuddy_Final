import React from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  TouchableOpacity,
  SafeAreaView,
  Platform,
  Linking,
} from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { useNavigation } from '@react-navigation/native';
import Header from '../../components/Header';
import Colors from '../../constants/colors';

const HelpSupportScreen = ({ navigation }) => {
  const handleContact = (method) => {
    switch (method) {
      case 'email':
        Linking.openURL('mailto:support@stocksenseai.com');
        break;
      case 'phone':
        Linking.openURL('tel:+1234567890');
        break;
      case 'chat':
        navigation.navigate('LiveChat');
        break;
      default:
        break;
    }
  };

  const renderContactMethod = (icon, title, description, onPress) => (
    <TouchableOpacity style={styles.contactMethod} onPress={onPress}>
      <View style={styles.iconContainer}>
        <Ionicons name={icon} size={24} color={Colors.primary} />
      </View>
      <View style={styles.contactContent}>
        <Text style={styles.contactTitle}>{title}</Text>
        <Text style={styles.contactDescription}>{description}</Text>
      </View>
      <Ionicons name="chevron-forward" size={20} color={Colors.darkGray} />
    </TouchableOpacity>
  );

  const renderFAQItem = (question, answer) => (
    <TouchableOpacity 
      style={styles.faqItem}
      onPress={() => navigation.navigate('FAQ')}
    >
      <View style={styles.faqHeader}>
        <Text style={styles.faqQuestion}>{question}</Text>
        <Ionicons name="chevron-forward" size={20} color={Colors.darkGray} />
      </View>
      <Text style={styles.faqAnswer}>{answer}</Text>
    </TouchableOpacity>
  );

  return (
    <SafeAreaView style={styles.container}>
      <Header 
        title="Help & Support" 
        showBackButton={true}
        onBackPress={() => navigation.goBack()}
      />

      <ScrollView style={styles.content} showsVerticalScrollIndicator={false}>
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Contact Us</Text>
          {renderContactMethod(
            'mail',
            'Email Support',
            'Get help via email',
            () => handleContact('email')
          )}
          {renderContactMethod(
            'call',
            'Phone Support',
            'Call our support team',
            () => handleContact('phone')
          )}
          {renderContactMethod(
            'chatbubbles',
            'Live Chat',
            'Chat with our support team',
            () => handleContact('chat')
          )}
        </View>

        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Frequently Asked Questions</Text>
          {renderFAQItem(
            'How do I get started with StockSense AI?',
            'To get started, simply create an account, complete your profile, and you can begin using our AI-powered stock analysis features.'
          )}
          {renderFAQItem(
            'How accurate are the AI predictions?',
            'Our AI predictions are based on advanced machine learning algorithms and historical data analysis. While we strive for accuracy, past performance does not guarantee future results.'
          )}
          {renderFAQItem(
            'Can I sync my existing portfolio?',
            'Yes, you can import your portfolio from various trading platforms or manually add your holdings.'
          )}
          <TouchableOpacity 
            style={[styles.menuItem, styles.viewAllButton]}
            onPress={() => navigation.navigate('FAQ')}
          >
            <Text style={styles.viewAllText}>View All FAQs</Text>
            <Ionicons name="chevron-forward" size={20} color={Colors.primary} />
          </TouchableOpacity>
        </View>

        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Chart Support</Text>
          <TouchableOpacity 
            style={styles.menuItem}
            onPress={() => navigation.navigate('LiveChart')}
          >
            <View style={styles.menuItemLeft}>
              <Ionicons name="analytics" size={24} color={Colors.primary} />
              <Text style={styles.menuItemText}>Live Chart Guide</Text>
            </View>
            <Ionicons name="chevron-forward" size={20} color={Colors.darkGray} />
          </TouchableOpacity>
        </View>

        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Resources</Text>
          <TouchableOpacity style={styles.menuItem}>
            <View style={styles.menuItemLeft}>
              <Ionicons name="book" size={24} color={Colors.primary} />
              <Text style={styles.menuItemText}>User Guide</Text>
            </View>
            <Ionicons name="chevron-forward" size={20} color={Colors.darkGray} />
          </TouchableOpacity>
          <TouchableOpacity style={styles.menuItem}>
            <View style={styles.menuItemLeft}>
              <Ionicons name="videocam" size={24} color={Colors.primary} />
              <Text style={styles.menuItemText}>Video Tutorials</Text>
            </View>
            <Ionicons name="chevron-forward" size={20} color={Colors.darkGray} />
          </TouchableOpacity>
          <TouchableOpacity style={styles.menuItem}>
            <View style={styles.menuItemLeft}>
              <Ionicons name="document-text" size={24} color={Colors.primary} />
              <Text style={styles.menuItemText}>Documentation</Text>
            </View>
            <Ionicons name="chevron-forward" size={20} color={Colors.darkGray} />
          </TouchableOpacity>
        </View>

        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Community</Text>
          <TouchableOpacity style={styles.menuItem}>
            <View style={styles.menuItemLeft}>
              <Ionicons name="people" size={24} color={Colors.primary} />
              <Text style={styles.menuItemText}>Join Our Community</Text>
            </View>
            <Ionicons name="chevron-forward" size={20} color={Colors.darkGray} />
          </TouchableOpacity>
          <TouchableOpacity style={styles.menuItem}>
            <View style={styles.menuItemLeft}>
              <Ionicons name="chatbubbles" size={24} color={Colors.primary} />
              <Text style={styles.menuItemText}>Discussion Forum</Text>
            </View>
            <Ionicons name="chevron-forward" size={20} color={Colors.darkGray} />
          </TouchableOpacity>
        </View>

        <TouchableOpacity style={styles.feedbackButton}>
          <Ionicons name="star" size={20} color={Colors.white} />
          <Text style={styles.feedbackButtonText}>Rate the App</Text>
        </TouchableOpacity>
      </ScrollView>
    </SafeAreaView>
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
  section: {
    backgroundColor: Colors.white,
    borderRadius: 12,
    overflow: 'hidden',
    marginBottom: 20,
    shadowColor: Colors.black,
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  sectionTitle: {
    fontSize: 18,
    fontWeight: '600',
    color: Colors.secondary,
    padding: 20,
    paddingBottom: 10,
  },
  contactMethod: {
    flexDirection: 'row',
    alignItems: 'center',
    padding: 20,
    borderTopWidth: 1,
    borderTopColor: Colors.gray,
  },
  iconContainer: {
    width: 40,
    height: 40,
    borderRadius: 20,
    backgroundColor: Colors.primary + '20',
    justifyContent: 'center',
    alignItems: 'center',
    marginRight: 12,
  },
  contactContent: {
    flex: 1,
  },
  contactTitle: {
    fontSize: 16,
    fontWeight: '500',
    color: Colors.secondary,
    marginBottom: 4,
  },
  contactDescription: {
    fontSize: 14,
    color: Colors.darkGray,
  },
  faqItem: {
    padding: 20,
    borderTopWidth: 1,
    borderTopColor: Colors.gray,
  },
  faqHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  faqQuestion: {
    fontSize: 16,
    fontWeight: '500',
    color: Colors.secondary,
    flex: 1,
    marginRight: 12,
  },
  faqAnswer: {
    fontSize: 14,
    color: Colors.darkGray,
    marginTop: 8,
  },
  menuItem: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    padding: 20,
    borderTopWidth: 1,
    borderTopColor: Colors.gray,
  },
  menuItemLeft: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  menuItemText: {
    fontSize: 16,
    color: Colors.secondary,
    marginLeft: 12,
  },
  viewAllButton: {
    backgroundColor: Colors.primary + '10',
  },
  viewAllText: {
    fontSize: 16,
    color: Colors.primary,
    fontWeight: '500',
  },
  feedbackButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: Colors.primary,
    borderRadius: 12,
    padding: 16,
    marginBottom: 30,
  },
  feedbackButtonText: {
    fontSize: 16,
    color: Colors.white,
    marginLeft: 8,
    fontWeight: '500',
  },
});

export default HelpSupportScreen; 