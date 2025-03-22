import React from 'react';
import { View, Text, StyleSheet } from 'react-native';
import Colors from '../constants/colors';

const StatusCard = ({ status, message }) => {
  return (
    <View style={styles.container}>
      <Text style={styles.statusText}>{status}</Text>
      {message && <Text style={styles.messageText}>{message}</Text>}
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    backgroundColor: Colors.white,
    borderRadius: 8,
    padding: 16,
    marginVertical: 8,
    shadowColor: Colors.black,
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 2,
  },
  statusText: {
    fontSize: 18,
    fontWeight: 'bold',
    color: Colors.secondary,
    marginBottom: 4,
  },
  messageText: {
    fontSize: 14,
    color: Colors.darkGray,
  },
});

export default StatusCard; 