import React from 'react';
import { View, Text, StyleSheet, TouchableOpacity, ScrollView, Image, Linking } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import Header from '../../components/Header';
import Colors from '../../constants/colors';

const AboutScreen = ({ navigation }) => {
  const handleLink = (url) => {
    Linking.openURL(url);
  };

  const renderMenuItem = (icon, title, onPress) => (
    <TouchableOpacity style={styles.menuItem} onPress={onPress}>
      <View style={styles.menuItemLeft}>
        <Ionicons name={icon} size={24} color={Colors.primary} />
        <Text style={styles.menuItemText}>{title}</Text>
      </View>
      <Ionicons name="chevron-forward" size={20} color={Colors.darkGray} />
    </TouchableOpacity>
  );

  return (
    <View style={styles.container}>
      <Header 
        title="About" 
        showBackButton={true}
        onBackPress={() => navigation.goBack()}
      />

      <ScrollView style={styles.content} showsVerticalScrollIndicator={false}>
        <View style={styles.logoContainer}>
          <View style={styles.logoIconContainer}>
            <Ionicons name="analytics" size={60} color={Colors.primary} />
          </View>
          <Text style={styles.appName}>StockSense AI</Text>
          <Text style={styles.version}>Version 1.0.0</Text>
        </View>

        <View style={styles.section}>
          <Text style={styles.sectionTitle}>About the App</Text>
          <Text style={styles.description}>
            StockSense AI is your intelligent companion for stock market analysis and trading. 
            Powered by advanced artificial intelligence, we provide real-time insights, 
            predictions, and portfolio management tools to help you make informed investment decisions.
          </Text>
        </View>

        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Legal</Text>
          {renderMenuItem(
            'document-text',
            'Terms of Service',
            () => handleLink('https://stocksenseai.com/terms')
          )}
          {renderMenuItem(
            'shield-checkmark',
            'Privacy Policy',
            () => handleLink('https://stocksenseai.com/privacy')
          )}
          {renderMenuItem(
            'information-circle',
            'Disclaimer',
            () => handleLink('https://stocksenseai.com/disclaimer')
          )}
        </View>

        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Connect With Us</Text>
          <View style={styles.socialLinks}>
            <TouchableOpacity
              style={styles.socialButton}
              onPress={() => handleLink('https://twitter.com/stocksenseai')}
            >
              <Ionicons name="logo-twitter" size={24} color={Colors.primary} />
            </TouchableOpacity>
            <TouchableOpacity
              style={styles.socialButton}
              onPress={() => handleLink('https://linkedin.com/company/stocksenseai')}
            >
              <Ionicons name="logo-linkedin" size={24} color={Colors.primary} />
            </TouchableOpacity>
            <TouchableOpacity
              style={styles.socialButton}
              onPress={() => handleLink('https://facebook.com/stocksenseai')}
            >
              <Ionicons name="logo-facebook" size={24} color={Colors.primary} />
            </TouchableOpacity>
          </View>
        </View>

        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Support</Text>
          {renderMenuItem(
            'mail',
            'Contact Support',
            () => handleLink('mailto:support@stocksenseai.com')
          )}
          {renderMenuItem(
            'help-circle',
            'FAQ',
            () => handleLink('https://stocksenseai.com/faq')
          )}
          {renderMenuItem(
            'bug',
            'Report a Bug',
            () => handleLink('https://stocksenseai.com/bug-report')
          )}
        </View>

        <View style={styles.footer}>
          <Text style={styles.footerText}>
            © 2024 StockSense AI. All rights reserved.
          </Text>
          <Text style={styles.footerText}>
            Made with ❤️ for traders worldwide
          </Text>
        </View>
      </ScrollView>
    </View>
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
  logoContainer: {
    alignItems: 'center',
    marginTop: 20,
    marginBottom: 30,
  },
  logoIconContainer: {
    width: 100,
    height: 100,
    borderRadius: 50,
    backgroundColor: Colors.primary + '20',
    justifyContent: 'center',
    alignItems: 'center',
    marginBottom: 16,
  },
  appName: {
    fontSize: 24,
    fontWeight: 'bold',
    color: Colors.secondary,
    marginBottom: 4,
  },
  version: {
    fontSize: 14,
    color: Colors.darkGray,
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
  description: {
    fontSize: 14,
    color: Colors.darkGray,
    lineHeight: 20,
    padding: 20,
    paddingTop: 0,
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
  socialLinks: {
    flexDirection: 'row',
    justifyContent: 'center',
    padding: 20,
    paddingTop: 10,
  },
  socialButton: {
    width: 50,
    height: 50,
    borderRadius: 25,
    backgroundColor: Colors.primary + '20',
    justifyContent: 'center',
    alignItems: 'center',
    marginHorizontal: 10,
  },
  footer: {
    alignItems: 'center',
    marginBottom: 30,
  },
  footerText: {
    fontSize: 12,
    color: Colors.darkGray,
    marginBottom: 4,
  },
});

export default AboutScreen; 