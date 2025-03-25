import React, { useState, useEffect } from 'react';
import { View, Text, StyleSheet, TextInput, TouchableOpacity, FlatList, ActivityIndicator } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import Header from '../../components/Header';
import Colors from '../../constants/colors';
import FinnhubService from '../../services/FinnhubService';

const SearchScreen = ({ navigation }) => {
  const [searchQuery, setSearchQuery] = useState('');
  const [recentSearches, setRecentSearches] = useState(['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA']);
  const [searchResults, setSearchResults] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  // Search for stocks using the API
  const handleSearch = async (query) => {
    if (!query.trim()) return;
    
    setLoading(true);
    setError('');
    
    try {
      const results = await FinnhubService.searchStocks(query);
      setSearchResults(FinnhubService.formatSearchResults(results));
      
      // Add to recent searches if not already present
      if (!recentSearches.includes(query) && query.length > 1) {
        setRecentSearches(prev => [query, ...prev].slice(0, 5));
      }
    } catch (err) {
      setError('Failed to search stocks. Please try again.');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  // Debounce search to prevent too many API calls
  useEffect(() => {
    const timeoutId = setTimeout(() => {
      if (searchQuery.length > 1) {
        handleSearch(searchQuery);
      } else {
        setSearchResults([]);
      }
    }, 500);
    
    return () => clearTimeout(timeoutId);
  }, [searchQuery]);

  const clearSearch = () => {
    setSearchQuery('');
    setSearchResults([]);
  };

  const handleStockSelect = (stock) => {
    navigation.navigate('StockDetail', { symbol: stock.symbol });
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

  const renderSearchResult = ({ item }) => (
    <TouchableOpacity
      style={styles.searchResultItem}
      onPress={() => handleStockSelect(item)}
    >
      <View style={styles.searchResultContent}>
        <Text style={styles.symbolText}>{item.symbol}</Text>
        <Text style={styles.descriptionText} numberOfLines={1}>{item.description}</Text>
      </View>
      <Ionicons name="chevron-forward" size={20} color={Colors.darkGray} />
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
        {loading && (
          <View style={styles.loadingContainer}>
            <ActivityIndicator size="large" color={Colors.primary} />
          </View>
        )}
        
        {error ? (
          <Text style={styles.errorText}>{error}</Text>
        ) : null}

        {searchResults.length > 0 ? (
          <>
            <Text style={styles.sectionTitle}>Results</Text>
            <FlatList
              data={searchResults}
              renderItem={renderSearchResult}
              keyExtractor={(item) => item.symbol}
              showsVerticalScrollIndicator={false}
            />
          </>
        ) : (
          searchQuery.length === 0 && (
            <>
              <Text style={styles.sectionTitle}>Recent Searches</Text>
              <FlatList
                data={recentSearches}
                renderItem={renderRecentSearch}
                keyExtractor={(item) => item}
                showsVerticalScrollIndicator={false}
              />
            </>
          )
        )}
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
  loadingContainer: {
    padding: 20,
    alignItems: 'center',
  },
  errorText: {
    color: 'red',
    textAlign: 'center',
    marginVertical: 10,
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
  searchResultItem: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingVertical: 15,
    borderBottomWidth: 1,
    borderBottomColor: Colors.gray,
  },
  searchResultContent: {
    flex: 1,
  },
  symbolText: {
    fontSize: 16,
    fontWeight: 'bold',
    color: Colors.secondary,
  },
  descriptionText: {
    fontSize: 14,
    color: Colors.darkGray,
    marginTop: 2,
  },
});

export default SearchScreen; 