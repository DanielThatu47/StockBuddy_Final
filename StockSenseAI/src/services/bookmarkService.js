import AsyncStorage from '@react-native-async-storage/async-storage';

const BOOKMARKS_KEY = '@news_bookmarks';

export const getBookmarks = async () => {
  try {
    const bookmarks = await AsyncStorage.getItem(BOOKMARKS_KEY);
    return bookmarks ? JSON.parse(bookmarks) : [];
  } catch (error) {
    console.error('Error getting bookmarks:', error);
    return [];
  }
};

export const addBookmark = async (news) => {
  try {
    const bookmarks = await getBookmarks();
    if (!bookmarks.some(bookmark => bookmark.id === news.id)) {
      bookmarks.push(news);
      await AsyncStorage.setItem(BOOKMARKS_KEY, JSON.stringify(bookmarks));
      return true;
    }
    return false;
  } catch (error) {
    console.error('Error adding bookmark:', error);
    return false;
  }
};

export const removeBookmark = async (newsId) => {
  try {
    const bookmarks = await getBookmarks();
    const updatedBookmarks = bookmarks.filter(bookmark => bookmark.id !== newsId);
    await AsyncStorage.setItem(BOOKMARKS_KEY, JSON.stringify(updatedBookmarks));
    return true;
  } catch (error) {
    console.error('Error removing bookmark:', error);
    return false;
  }
};

export const isBookmarked = async (newsId) => {
  try {
    const bookmarks = await getBookmarks();
    return bookmarks.some(bookmark => bookmark.id === newsId);
  } catch (error) {
    console.error('Error checking bookmark:', error);
    return false;
  }
}; 