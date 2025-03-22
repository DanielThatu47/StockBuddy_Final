import React from 'react';
import { View, Text, StyleSheet, TouchableOpacity } from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';
import { Ionicons } from '@expo/vector-icons';
import Colors from '../constants/colors';

const Header = ({ title, subtitle, children, onSearchPress, showBackButton, onBackPress }) => {
  return (
    <LinearGradient
      colors={[Colors.primary, Colors.primaryDark]}
      style={styles.container}
      start={{ x: 0, y: 0 }}
      end={{ x: 1, y: 0 }}
    >
      <View style={styles.contentContainer}>
        <View style={styles.leftContainer}>
          {showBackButton && (
            <TouchableOpacity style={styles.backButton} onPress={onBackPress}>
              <Ionicons name="arrow-back" size={24} color={Colors.white} />
            </TouchableOpacity>
          )}
          <View style={styles.titleContainer}>
            <Text style={styles.title}>{title}</Text>
            {subtitle && <Text style={styles.subtitle}>{subtitle}</Text>}
          </View>
        </View>
        <View style={styles.actionsContainer}>
          {onSearchPress && (
            <TouchableOpacity style={styles.iconButton} onPress={onSearchPress}>
              <Ionicons name="search" size={24} color={Colors.white} />
            </TouchableOpacity>
          )}
          {children}
        </View>
      </View>
    </LinearGradient>
  );
};

const styles = StyleSheet.create({
  container: {
    paddingTop: 45,
    paddingBottom: 20,
    paddingHorizontal: 20,
    borderBottomLeftRadius: 20,
    borderBottomRightRadius: 20,
  },
  contentContainer: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  leftContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    flex: 1,
  },
  backButton: {
    padding: 8,
    marginRight: 8,
  },
  titleContainer: {
    flex: 1,
  },
  title: {
    fontSize: 24,
    fontWeight: 'bold',
    color: Colors.white,
  },
  subtitle: {
    fontSize: 14,
    color: Colors.white,
    opacity: 0.8,
    marginTop: 2,
  },
  actionsContainer: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  iconButton: {
    padding: 8,
    marginRight: 8,
  },
});

export default Header; 