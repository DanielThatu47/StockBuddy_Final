import React from 'react';
import { View, Text, StyleSheet, TouchableOpacity } from 'react-native';
import Colors from '../constants/colors';

const SectionHeader = ({ title, actionText, onActionPress }) => {
  return (
    <View style={styles.container}>
      <Text style={styles.title}>{title}</Text>
      {actionText && (
        <TouchableOpacity onPress={onActionPress}>
          <Text style={styles.actionText}>{actionText}</Text>
        </TouchableOpacity>
      )}
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginVertical: 16,
  },
  title: {
    fontSize: 20,
    fontWeight: 'bold',
    color: Colors.secondary,
  },
  actionText: {
    fontSize: 14,
    color: Colors.primary,
    fontWeight: '600',
  },
});

export default SectionHeader; 