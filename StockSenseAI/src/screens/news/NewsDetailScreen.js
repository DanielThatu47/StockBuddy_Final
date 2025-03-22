import React from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  Image,
  TouchableOpacity,
  Linking,
  Share,
  Dimensions,
  Platform
} from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import Colors from '../../constants/colors';
import Header from '../../components/Header';

const { width } = Dimensions.get('window');

const NewsDetailScreen = ({ route, navigation }) => {
  const { news } = route.params;

  const handleShare = async () => {
    try {
      await Share.share({
        message: `${news.title}\n\nRead more: ${news.url}`,
        title: news.title,
      });
    } catch (error) {
      console.error('Error sharing:', error);
    }
  };

  const handleOpenBrowser = async () => {
    try {
      await Linking.openURL(news.url);
    } catch (error) {
      console.error('Error opening URL:', error);
    }
  };

  return (
    <View style={styles.container}>
      <Header 
        title="News Detail" 
        showBackButton={true}
        onBackPress={() => navigation.goBack()}
      />

      <ScrollView style={styles.scrollView} showsVerticalScrollIndicator={false}>
        <View style={styles.imageContainer}>
          <Image 
            source={{ uri: news.image }} 
            style={styles.image}
            defaultSource={require('../../../assets/placeholder-news.png')}
          />
          <View style={styles.imageOverlay} />
        </View>

        <View style={styles.contentContainer}>
          <View style={styles.sourceContainer}>
            <Text style={styles.source}>{news.source}</Text>
            <Text style={styles.time}>{news.time}</Text>
          </View>

          <Text style={styles.title}>{news.title}</Text>

          <View style={styles.authorContainer}>
            <Ionicons name="person-outline" size={16} color="#666" />
            <Text style={styles.author}>{news.author}</Text>
          </View>

          <View style={styles.divider} />

          <Text style={styles.summary}>{news.summary}</Text>

          <View style={styles.divider} />

          <Text style={styles.content}>{news.content}</Text>

          <View style={styles.readMoreContainer}>
            <TouchableOpacity 
              style={styles.readMoreButton}
              onPress={handleOpenBrowser}
            >
              <Text style={styles.readMoreText}>Read Full Article</Text>
              <Ionicons name="arrow-forward" size={16} color="#007AFF" />
            </TouchableOpacity>
          </View>
        </View>
      </ScrollView>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#fff',
  },
  scrollView: {
    flex: 1,
  },
  imageContainer: {
    position: 'relative',
    width: width,
    height: width * 0.6,
  },
  image: {
    width: '100%',
    height: '100%',
    resizeMode: 'cover',
  },
  imageOverlay: {
    ...StyleSheet.absoluteFillObject,
    backgroundColor: 'rgba(0,0,0,0.1)',
  },
  contentContainer: {
    padding: 20,
  },
  sourceContainer: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 12,
  },
  source: {
    fontSize: 14,
    fontWeight: '600',
    color: '#007AFF',
    textTransform: 'uppercase',
    letterSpacing: 0.5,
  },
  time: {
    fontSize: 14,
    color: '#666',
  },
  title: {
    fontSize: 24,
    fontWeight: '700',
    color: '#000',
    lineHeight: 32,
    marginBottom: 16,
  },
  authorContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 20,
  },
  author: {
    fontSize: 14,
    color: '#666',
    marginLeft: 6,
  },
  divider: {
    height: 1,
    backgroundColor: '#eee',
    marginVertical: 20,
  },
  summary: {
    fontSize: 18,
    lineHeight: 26,
    color: '#333',
    marginBottom: 20,
  },
  content: {
    fontSize: 16,
    lineHeight: 24,
    color: '#444',
  },
  readMoreContainer: {
    marginTop: 30,
    marginBottom: 40,
  },
  readMoreButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    padding: 16,
    backgroundColor: '#f8f9fa',
    borderRadius: 12,
  },
  readMoreText: {
    fontSize: 16,
    fontWeight: '600',
    color: '#007AFF',
    marginRight: 8,
  },
});

export default NewsDetailScreen; 