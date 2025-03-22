import React from 'react';
import { View, TextInput, StyleSheet, TouchableOpacity } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import Colors from '../constants/colors';

const SearchInput = ({ placeholder, value, onChangeText, onSubmit, style }) => {
  return (
    <View style={[styles.container, style]}>
      <Ionicons name="search" size={20} color={Colors.darkGray} style={styles.icon} />
      <TextInput
        style={styles.input}
        placeholder={placeholder || "Search..."}
        placeholderTextColor={Colors.darkGray}
        value={value}
        onChangeText={onChangeText}
        onSubmitEditing={onSubmit}
        returnKeyType="search"
        clearButtonMode="while-editing"
      />
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: Colors.white,
    borderRadius: 8,
    paddingHorizontal: 12,
    paddingVertical: 8,
    shadowColor: Colors.black,
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 2,
  },
  icon: {
    marginRight: 8,
  },
  input: {
    flex: 1,
    fontSize: 16,
    color: Colors.secondary,
    padding: 8,
  },
});

export default SearchInput; 