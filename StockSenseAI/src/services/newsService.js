import { getNewsApiUrl } from '../config/api';

export const fetchNews = async (query = '') => {
  try {
    // Default to stock market news if no specific query
    const searchQuery = query || 'stock market OR trading OR stocks OR financial markets';
    
    const response = await fetch(getNewsApiUrl(searchQuery));
    const data = await response.json();

    if (data.status === 'error') {
      throw new Error(data.message || 'Failed to fetch news');
    }

    // Filter and transform the articles
    const articles = data.articles
      .filter(article => {
        // Filter for stock market and trading related content
        const relevantKeywords = [
          'stock', 'trading', 'market', 'investor', 'shares', 
          'bull', 'bear', 'dow', 's&p', 'nasdaq', 'trading volume',
          'stock price', 'market cap', 'dividend', 'earnings'
        ];
        
        const title = article.title.toLowerCase();
        const description = (article.description || '').toLowerCase();
        const content = (article.content || '').toLowerCase();
        
        return relevantKeywords.some(keyword => 
          title.includes(keyword) || 
          description.includes(keyword) || 
          content.includes(keyword)
        );
      })
      .map(article => ({
        id: article.url,
        title: article.title,
        source: article.source.name,
        time: new Date(article.publishedAt).toLocaleDateString('en-US', {
          year: 'numeric',
          month: 'short',
          day: 'numeric',
          hour: '2-digit',
          minute: '2-digit'
        }),
        image: article.urlToImage || null,
        summary: article.description || '',
        url: article.url,
        content: article.content || '',
        author: article.author || 'Unknown',
        publishedAt: article.publishedAt
      }));

    return articles;
  } catch (error) {
    console.error('Error fetching news:', error);
    throw new Error('Failed to fetch news. Please try again later.');
  }
}; 