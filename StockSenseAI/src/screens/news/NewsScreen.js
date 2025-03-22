import React, { useState, useMemo, useEffect } from 'react';
import {
  View,
  Text,
  StyleSheet,
  FlatList,
  TouchableOpacity,
  Image,
  RefreshControl,
  ScrollView,
  ActivityIndicator,
  TextInput,
} from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import Colors from '../../constants/colors';
import Header from '../../components/Header';
import { NEWS_CATEGORIES } from '../../config/api';
import { fetchNews } from '../../services/newsService';
import { addBookmark, removeBookmark, isBookmarked } from '../../services/bookmarkService';

const NewsScreen = ({ navigation }) => {
  const [refreshing, setRefreshing] = useState(false);
  const [selectedCategory, setSelectedCategory] = useState('all');
  const [newsData, setNewsData] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [searchQuery, setSearchQuery] = useState('');
  const [bookmarkedNews, setBookmarkedNews] = useState(new Set());

  const categories = Object.values(NEWS_CATEGORIES);

  const loadNews = async (category = 'all', query = '') => {
    try {
      setLoading(true);
      setError(null);
      
      const selectedCategory = categories.find(cat => cat.id === category);
      const searchQuery = query || selectedCategory?.query || 'finance';
      
      const news = await fetchNews(searchQuery);
      setNewsData(news.map(item => ({ ...item, category })));
      
      // Update bookmarked status
      const bookmarkedIds = new Set();
      for (const item of news) {
        if (await isBookmarked(item.id)) {
          bookmarkedIds.add(item.id);
        }
      }
      setBookmarkedNews(bookmarkedIds);
    } catch (err) {
      setError(err.message || 'Error fetching news');
      console.error('Error fetching news:', err);
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  };

  useEffect(() => {
    loadNews(selectedCategory, searchQuery);
  }, [selectedCategory, searchQuery]);

  const onRefresh = React.useCallback(() => {
    setRefreshing(true);
    loadNews(selectedCategory, searchQuery);
  }, [selectedCategory, searchQuery]);

  const handleSearch = (text) => {
    setSearchQuery(text);
    if (text.trim()) {
      loadNews('all', text);
    } else {
      loadNews(selectedCategory);
    }
  };

  const handleBookmark = async (news) => {
    try {
      if (bookmarkedNews.has(news.id)) {
        await removeBookmark(news.id);
        setBookmarkedNews(prev => {
          const newSet = new Set(prev);
          newSet.delete(news.id);
          return newSet;
        });
      } else {
        await addBookmark(news);
        setBookmarkedNews(prev => new Set([...prev, news.id]));
      }
    } catch (error) {
      console.error('Error handling bookmark:', error);
    }
  };

  const handleNewsPress = (news) => {
    navigation.navigate('NewsDetail', { news });
  };

  // Filter news based on selected category and search query
  const filteredNews = useMemo(() => {
    let filtered = newsData;
    
    if (selectedCategory !== 'all') {
      filtered = filtered.filter(news => news.category === selectedCategory);
    }
    
    if (searchQuery.trim()) {
      const query = searchQuery.toLowerCase();
      filtered = filtered.filter(news => 
        news.title.toLowerCase().includes(query) ||
        news.summary.toLowerCase().includes(query) ||
        news.source.toLowerCase().includes(query)
      );
    }
    
    return filtered;
  }, [selectedCategory, newsData, searchQuery]);

  const renderNewsItem = ({ item }) => (
    <TouchableOpacity 
      style={styles.newsCard}
      onPress={() => handleNewsPress(item)}
    >
      <Image 
        source={{ uri: item.image }} 
        style={styles.newsImage}
        defaultSource={require('../../../assets/placeholder-news.png')}
      />
      <View style={styles.newsContent}>
        <Text style={styles.newsSource}>{item.source}</Text>
        <Text style={styles.newsTitle}>{item.title}</Text>
        <Text style={styles.newsSummary}>{item.summary}</Text>
        <View style={styles.newsFooter}>
          <Text style={styles.newsTime}>{item.time}</Text>
          <TouchableOpacity onPress={() => handleBookmark(item)}>
            <Ionicons 
              name={bookmarkedNews.has(item.id) ? "bookmark" : "bookmark-outline"} 
              size={20} 
              color={Colors.primary} 
            />
          </TouchableOpacity>
        </View>
      </View>
    </TouchableOpacity>
  );

  const renderCategoryItem = ({ item }) => (
    <TouchableOpacity
      key={item.id}
      style={[
        styles.categoryButton,
        selectedCategory === item.id && styles.selectedCategory,
      ]}
      onPress={() => setSelectedCategory(item.id)}
    >
      <Ionicons
        name={item.icon}
        size={16}
        color={selectedCategory === item.id ? Colors.white : Colors.primary}
        style={styles.categoryIcon}
      />
      <Text
        style={[
          styles.categoryText,
          selectedCategory === item.id && styles.selectedCategoryText,
        ]}
      >
        {item.label}
      </Text>
    </TouchableOpacity>
  );

  const renderContent = () => {
    if (loading && !refreshing) {
      return (
        <View style={styles.centerContainer}>
          <ActivityIndicator size="large" color={Colors.primary} />
          <Text style={styles.loadingText}>Loading news...</Text>
        </View>
      );
    }

    if (error) {
      return (
        <View style={styles.centerContainer}>
          <Ionicons name="alert-circle" size={48} color={Colors.error} />
          <Text style={styles.errorText}>{error}</Text>
          <TouchableOpacity 
            style={styles.retryButton}
            onPress={() => loadNews(selectedCategory, searchQuery)}
          >
            <Text style={styles.retryButtonText}>Retry</Text>
          </TouchableOpacity>
        </View>
      );
    }

    return (
      <FlatList
        data={filteredNews}
        renderItem={renderNewsItem}
        keyExtractor={(item) => item.id}
        contentContainerStyle={styles.newsList}
        refreshControl={
          <RefreshControl refreshing={refreshing} onRefresh={onRefresh} />
        }
        ListEmptyComponent={() => (
          <View style={styles.emptyContainer}>
            <Ionicons name="newspaper-outline" size={48} color={Colors.gray} />
            <Text style={styles.emptyText}>No news found for this category</Text>
          </View>
        )}
      />
    );
  };

  return (
    <View style={styles.container}>
      <Header 
        title="Financial News" 
        showBackButton={false}
      />

      <View style={styles.searchContainer}>
        <Ionicons name="search" size={20} color={Colors.black} style={styles.searchIcon} />
        <TextInput
          style={styles.searchInput}
          placeholder="Search news..."
          value={searchQuery}
          onChangeText={handleSearch}
          placeholderTextColor={Colors.black}
        />
        {searchQuery ? (
          <TouchableOpacity onPress={() => handleSearch('')}>
            <Ionicons name="close-circle" size={20} color={Colors.red} />
          </TouchableOpacity>
        ) : null}
      </View>

      <View style={styles.categoriesContainer}>
        <ScrollView
          horizontal
          showsHorizontalScrollIndicator={false}
          contentContainerStyle={styles.categoriesList}
        >
          {categories.map((category) => renderCategoryItem({ item: category }))}
        </ScrollView>
      </View>

      {renderContent()}
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: Colors.background,
  },
  searchContainer: {
    padding:10,
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: Colors.white,
    marginHorizontal: 15,
    marginVertical: 10,
    paddingHorizontal: 15,
    borderRadius: 10,
    borderWidth: 1,
    borderColor: Colors.gray,
  },
  searchIcon: {
    marginRight: 10,
  },
  searchInput: {
    flex: 1,
    height: 40,
    fontSize: 16,
    color: Colors.secondary,
  },
  categoriesContainer: {
    backgroundColor: Colors.white,
    borderBottomWidth: 1,
    borderBottomColor: Colors.gray,
  },
  categoriesList: {
    paddingHorizontal: 15,
    paddingVertical: 12,
  },
  categoryButton: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: 16,
    paddingVertical: 8,
    marginRight: 10,
    borderRadius: 20,
    backgroundColor: Colors.gray,
  },
  categoryIcon: {
    marginRight: 6,
  },
  selectedCategory: {
    backgroundColor: Colors.primary,
  },
  categoryText: {
    color: Colors.secondary,
    fontWeight: '500',
    fontSize: 14,
  },
  selectedCategoryText: {
    color: Colors.white,
  },
  newsList: {
    padding: 15,
  },
  newsCard: {
    backgroundColor: Colors.white,
    borderRadius: 12,
    marginBottom: 15,
    overflow: 'hidden',
    shadowColor: Colors.black,
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  newsImage: {
    width: '100%',
    height: 200,
  },
  newsContent: {
    padding: 15,
  },
  newsSource: {
    fontSize: 14,
    color: Colors.primary,
    fontWeight: '500',
    marginBottom: 5,
  },
  newsTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    color: Colors.secondary,
    marginBottom: 8,
  },
  newsSummary: {
    fontSize: 14,
    color: Colors.darkGray,
    marginBottom: 10,
    lineHeight: 20,
  },
  newsFooter: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  newsTime: {
    fontSize: 12,
    color: Colors.gray,
  },
  emptyContainer: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    paddingVertical: 40,
  },
  emptyText: {
    marginTop: 10,
    fontSize: 16,
    color: Colors.gray,
    textAlign: 'center',
  },
  centerContainer: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    padding: 20,
  },
  loadingText: {
    marginTop: 10,
    fontSize: 16,
    color: Colors.gray,
  },
  errorText: {
    marginTop: 10,
    fontSize: 16,
    color: Colors.error,
    textAlign: 'center',
  },
  retryButton: {
    marginTop: 20,
    paddingHorizontal: 20,
    paddingVertical: 10,
    backgroundColor: Colors.primary,
    borderRadius: 8,
  },
  retryButtonText: {
    color: Colors.white,
    fontSize: 16,
    fontWeight: '500',
  },
});

export default NewsScreen; 