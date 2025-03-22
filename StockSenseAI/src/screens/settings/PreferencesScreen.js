import React, { useState } from 'react';
import { View, Text, StyleSheet, Switch, TouchableOpacity, ScrollView } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import Header from '../../components/Header';
import Colors from '../../constants/colors';

const PreferencesScreen = ({ navigation }) => {
  const [preferences, setPreferences] = useState({
    darkMode: false,
    pushNotifications: true,
    emailNotifications: true,
    marketAlerts: true,
    portfolioUpdates: true,
    newsUpdates: true,
    language: 'English',
    currency: 'USD',
    timezone: 'UTC-5',
  });

  const togglePreference = (key) => {
    setPreferences(prev => ({
      ...prev,
      [key]: !prev[key]
    }));
  };

  const renderPreferenceItem = (title, value, onToggle, icon) => (
    <View style={styles.preferenceItem}>
      <View style={styles.preferenceLeft}>
        <Ionicons name={icon} size={24} color={Colors.primary} />
        <Text style={styles.preferenceText}>{title}</Text>
      </View>
      <Switch
        value={value}
        onValueChange={onToggle}
        trackColor={{ false: Colors.gray, true: Colors.primary }}
        thumbColor={Colors.white}
      />
    </View>
  );

  const renderSettingItem = (title, value, icon, onPress) => (
    <TouchableOpacity style={styles.settingItem} onPress={onPress}>
      <View style={styles.settingLeft}>
        <Ionicons name={icon} size={24} color={Colors.primary} />
        <Text style={styles.settingText}>{title}</Text>
      </View>
      <View style={styles.settingRight}>
        <Text style={styles.settingValue}>{value}</Text>
        <Ionicons name="chevron-forward" size={20} color={Colors.darkGray} />
      </View>
    </TouchableOpacity>
  );

  return (
    <View style={styles.container}>
      <Header 
        title="Preferences" 
        showBackButton={true}
        onBackPress={() => navigation.goBack()}
      />

      <ScrollView style={styles.content} showsVerticalScrollIndicator={false}>
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Appearance</Text>
          {renderPreferenceItem(
            'Dark Mode',
            preferences.darkMode,
            () => togglePreference('darkMode'),
            'moon'
          )}
        </View>

        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Notifications</Text>
          {renderPreferenceItem(
            'Push Notifications',
            preferences.pushNotifications,
            () => togglePreference('pushNotifications'),
            'notifications'
          )}
          {renderPreferenceItem(
            'Email Notifications',
            preferences.emailNotifications,
            () => togglePreference('emailNotifications'),
            'mail'
          )}
          {renderPreferenceItem(
            'Market Alerts',
            preferences.marketAlerts,
            () => togglePreference('marketAlerts'),
            'trending-up'
          )}
          {renderPreferenceItem(
            'Portfolio Updates',
            preferences.portfolioUpdates,
            () => togglePreference('portfolioUpdates'),
            'pie-chart'
          )}
          {renderPreferenceItem(
            'News Updates',
            preferences.newsUpdates,
            () => togglePreference('newsUpdates'),
            'newspaper'
          )}
        </View>

        <View style={styles.section}>
          <Text style={styles.sectionTitle}>General Settings</Text>
          {renderSettingItem(
            'Language',
            preferences.language,
            'language',
            () => {}
          )}
          {renderSettingItem(
            'Currency',
            preferences.currency,
            'cash',
            () => {}
          )}
          {renderSettingItem(
            'Timezone',
            preferences.timezone,
            'time',
            () => {}
          )}
        </View>

        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Data Management</Text>
          <TouchableOpacity style={styles.menuItem}>
            <View style={styles.menuItemLeft}>
              <Ionicons name="cloud-download" size={24} color={Colors.primary} />
              <Text style={styles.menuItemText}>Export Data</Text>
            </View>
            <Ionicons name="chevron-forward" size={20} color={Colors.darkGray} />
          </TouchableOpacity>
          <TouchableOpacity style={styles.menuItem}>
            <View style={styles.menuItemLeft}>
              <Ionicons name="trash" size={24} color={Colors.error} />
              <Text style={[styles.menuItemText, { color: Colors.error }]}>Clear All Data</Text>
            </View>
            <Ionicons name="chevron-forward" size={20} color={Colors.darkGray} />
          </TouchableOpacity>
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
  preferenceItem: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    padding: 20,
    borderTopWidth: 1,
    borderTopColor: Colors.gray,
  },
  preferenceLeft: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  preferenceText: {
    fontSize: 16,
    color: Colors.secondary,
    marginLeft: 12,
  },
  settingItem: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    padding: 20,
    borderTopWidth: 1,
    borderTopColor: Colors.gray,
  },
  settingLeft: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  settingText: {
    fontSize: 16,
    color: Colors.secondary,
    marginLeft: 12,
  },
  settingRight: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  settingValue: {
    fontSize: 16,
    color: Colors.darkGray,
    marginRight: 8,
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
});

export default PreferencesScreen; 