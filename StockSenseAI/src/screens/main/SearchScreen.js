import React, { useState } from 'react';
import { View, Text, StyleSheet, TextInput, TouchableOpacity, FlatList } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import Header from '../../components/Header';
import Colors from '../../constants/colors';

const SearchScreen = ({ navigation }) => {
  const [searchQuery, setSearchQuery] = useState('');
  const [recentSearches] = useState(['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA']);

  const handleSearch = (query) => {
    // Implement search functionality
    console.log('Searching for:', query);
  };

  const clearSearch = () => {
    setSearchQuery('');
  };

  const renderRecentSearch = ({ item }) => (
    <TouchableOpacity
      style={styles.recentSearchItem}
      onPress={() => {
        setSearchQuery(item);
        handleSearch(item);
      }}
    >
      <Ionicons name="time-outline" size={20} color={Colors.darkGray} />
      <Text style={styles.recentSearchText}>{item}</Text>
    </TouchableOpacity>
  );

  return (
    <View style={styles.container}>
      <Header 
        title="Search Stocks" 
        showBackButton={true}
        onSearchPress={null}
        onBackPress={() => navigation.goBack()}
      />

      <View style={styles.searchContainer}>
        <View style={styles.searchInputContainer}>
          <Ionicons name="search" size={20} color={Colors.darkGray} style={styles.searchIcon} />
          <TextInput
            style={styles.searchInput}
            placeholder="Search stocks, companies, or symbols"
            value={searchQuery}
            onChangeText={setSearchQuery}
            autoFocus
          />
          {searchQuery.length > 0 && (
            <TouchableOpacity onPress={clearSearch} style={styles.clearButton}>
              <Ionicons name="close-circle" size={20} color={Colors.darkGray} />
            </TouchableOpacity>
          )}
        </View>
      </View>

      <View style={styles.content}>
        <Text style={styles.sectionTitle}>Recent Searches</Text>
        <FlatList
          data={recentSearches}
          renderItem={renderRecentSearch}
          keyExtractor={(item) => item}
          showsVerticalScrollIndicator={false}
        />
      </View>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: Colors.background,
  },
  searchContainer: {
    padding: 20,
    paddingTop: 10,
  },
  searchInputContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: Colors.white,
    borderRadius: 12,
    paddingHorizontal: 15,
    height: 50,
    shadowColor: Colors.black,
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  searchIcon: {
    marginRight: 10,
  },
  searchInput: {
    flex: 1,
    fontSize: 16,
    color: Colors.secondary,
  },
  clearButton: {
    padding: 5,
  },
  content: {
    flex: 1,
    paddingHorizontal: 20,
  },
  sectionTitle: {
    fontSize: 18,
    fontWeight: '600',
    color: Colors.secondary,
    marginBottom: 15,
  },
  recentSearchItem: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingVertical: 12,
    borderBottomWidth: 1,
    borderBottomColor: Colors.gray,
  },
  recentSearchText: {
    fontSize: 16,
    color: Colors.secondary,
    marginLeft: 10,
  },
});

export default SearchScreen; 