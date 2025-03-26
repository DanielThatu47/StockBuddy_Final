import React, { useState, useEffect, useRef } from 'react';
import {
  View,
  Text,
  TextInput,
  StyleSheet,
  ScrollView,
  TouchableOpacity,
  ActivityIndicator,
  Alert,
  Platform,
  StatusBar,
  FlatList,
  RefreshControl
} from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import * as Notifications from 'expo-notifications';
import { LineChart } from 'react-native-chart-kit';
import { Dimensions } from 'react-native';
import * as PredictionService from '../services/predictionService';
import Colors from '../constants/colors';
import AsyncStorage from '@react-native-async-storage/async-storage';

// Set up notifications
Notifications.setNotificationHandler({
  handleNotification: async () => ({
    shouldShowAlert: true,
    shouldPlaySound: true,
    shouldSetBadge: false,
  }),
});

// Storage keys
const ACTIVE_PREDICTION_KEY = 'active_prediction';

const AIPredictionScreen = ({ navigation }) => {
  const [symbol, setSymbol] = useState('');
  const [daysAhead, setDaysAhead] = useState('5');
  const [isLoading, setIsLoading] = useState(false);
  const [prediction, setPrediction] = useState(null);
  const [progress, setProgress] = useState(0);
  const [error, setError] = useState(null);
  const [recentPredictions, setRecentPredictions] = useState([]);
  const [loadingPredictions, setLoadingPredictions] = useState(false);
  const [activePredictionInfo, setActivePredictionInfo] = useState(null);
  const [refreshing, setRefreshing] = useState(false);
  
  const statusCheckInterval = useRef(null);
  const notificationListener = useRef();
  const responseListener = useRef();
  const isMounted = useRef(true);
  
  // Set up notification listeners, fetch recent predictions, check for active prediction
  useEffect(() => {
    isMounted.current = true;
    registerForPushNotificationsAsync();
    
    notificationListener.current = Notifications.addNotificationReceivedListener(
      notification => {
        console.log(notification);
      }
    );
    
    responseListener.current = Notifications.addNotificationResponseReceivedListener(
      response => {
        if (prediction && prediction.id) {
          navigation.navigate('PredictionDetails', { id: prediction.id });
        }
      }
    );
    
    // Load active prediction if exists
    loadActivePrediction();
    fetchRecentPredictions();
    
    // Clean up on unmount
    return () => {
      isMounted.current = false;
      Notifications.removeNotificationSubscription(notificationListener.current);
      Notifications.removeNotificationSubscription(responseListener.current);
      clearInterval(statusCheckInterval.current);
    };
  }, []);
  
  // Function to load active prediction from storage
  const loadActivePrediction = async () => {
    try {
      const storedPrediction = await AsyncStorage.getItem(ACTIVE_PREDICTION_KEY);
      if (storedPrediction) {
        const predictionData = JSON.parse(storedPrediction);
        
        // If the prediction data is valid and not too old (less than 30 minutes)
        const createdTime = new Date(predictionData.createdAt).getTime();
        const currentTime = new Date().getTime();
        const timeDiff = (currentTime - createdTime) / (1000 * 60); // minutes
        
        if (timeDiff < 30) {
          setActivePredictionInfo(predictionData);
          setSymbol(predictionData.symbol);
          setDaysAhead(predictionData.daysAhead.toString());
          setIsLoading(true);
          
          // Start checking the status
          startStatusCheck(predictionData.taskId);
        } else {
          // Clear stale prediction
          await AsyncStorage.removeItem(ACTIVE_PREDICTION_KEY);
        }
      }
    } catch (err) {
      console.error('Error loading active prediction:', err);
    }
  };
  
  // Function to save active prediction to storage
  const saveActivePrediction = async (predictionData) => {
    try {
      const predictionInfo = {
        taskId: predictionData.taskId,
        symbol: predictionData.symbol || symbol,
        daysAhead: predictionData.daysAhead || parseInt(daysAhead),
        createdAt: new Date().toISOString()
      };
      
      await AsyncStorage.setItem(ACTIVE_PREDICTION_KEY, JSON.stringify(predictionInfo));
      setActivePredictionInfo(predictionInfo);
    } catch (err) {
      console.error('Error saving active prediction:', err);
    }
  };
  
  // Function to clear active prediction from storage
  const clearActivePrediction = async () => {
    try {
      await AsyncStorage.removeItem(ACTIVE_PREDICTION_KEY);
      setActivePredictionInfo(null);
    } catch (err) {
      console.error('Error clearing active prediction:', err);
    }
  };
  
  // Start checking prediction status periodically
  const startStatusCheck = (taskId) => {
    if (statusCheckInterval.current) {
      clearInterval(statusCheckInterval.current);
    }
    
    // Immediately check status once
    checkPredictionStatus(taskId);
    
    statusCheckInterval.current = setInterval(() => {
      if (isMounted.current) {
        checkPredictionStatus(taskId);
      }
    }, 5000);
  };
  
  // Function to fetch recent predictions
  const fetchRecentPredictions = async () => {
    try {
      setLoadingPredictions(true);
      const predictions = await PredictionService.getUserPredictions();
      
      if (isMounted.current) {
        setRecentPredictions(predictions.slice(0, 5)); // Get the 5 most recent
      }
    } catch (err) {
      console.error('Error fetching recent predictions:', err);
    } finally {
      if (isMounted.current) {
        setLoadingPredictions(false);
      }
    }
  };
  
  // Function to check status periodically
  const checkPredictionStatus = async (taskId) => {
    try {
      const response = await PredictionService.getPredictionStatus(taskId);
      
      if (!isMounted.current) return;
      
      setProgress(response.progress || 0);
      
      if (response.status === 'completed') {
        // Prediction is complete
        setPrediction(response);
        clearInterval(statusCheckInterval.current);
        setIsLoading(false);
        
        // Clear active prediction
        clearActivePrediction();
        
        // Send notification
        await Notifications.scheduleNotificationAsync({
          content: {
            title: 'Prediction Complete!',
            body: `Your prediction for ${symbol} is ready.`,
            data: { id: response.id },
          },
          trigger: null,
        });
        
        // Refresh recent predictions
        fetchRecentPredictions();
      } else if (response.status === 'failed' || response.status === 'stopped') {
        // Prediction failed or was stopped
        clearInterval(statusCheckInterval.current);
        setIsLoading(false);
        setError(response.status === 'failed' ? 'Prediction failed' : 'Prediction stopped');
        
        // Clear active prediction
        clearActivePrediction();
        
        // Refresh recent predictions
        fetchRecentPredictions();
      }
    } catch (err) {
      console.error('Error checking prediction status:', err);
      
      // If the error persists for multiple checks, clear the interval
      if (err.message?.includes('not found') || err.message?.includes('404')) {
        clearInterval(statusCheckInterval.current);
        setIsLoading(false);
        clearActivePrediction();
      }
    }
  };
  
  // Start prediction
  const handleStartPrediction = async () => {
    if (!symbol) {
      Alert.alert('Error', 'Please enter a stock symbol');
      return;
    }
    
    // Check if there's already an active prediction running
    if (activePredictionInfo) {
      Alert.alert(
        'Prediction in Progress',
        `There's already a prediction running for ${activePredictionInfo.symbol}. Would you like to continue with it or start a new one?`,
        [
          {
            text: 'Continue Current',
            onPress: () => {
              // Just continue with the current prediction
              setSymbol(activePredictionInfo.symbol);
              setDaysAhead(activePredictionInfo.daysAhead.toString());
            }
          },
          {
            text: 'Start New',
            onPress: async () => {
              // Stop the current prediction
              try {
                await PredictionService.stopPrediction(activePredictionInfo.taskId);
              } catch (err) {
                console.error('Error stopping previous prediction:', err);
              }
              
              // Clear and start new
              clearActivePrediction();
              clearInterval(statusCheckInterval.current);
              startNewPrediction();
            }
          },
          {
            text: 'Cancel',
            style: 'cancel'
          }
        ]
      );
      return;
    }
    
    // Start a new prediction
    startNewPrediction();
  };
  
  // Function to start a new prediction
  const startNewPrediction = async () => {
    try {
      setIsLoading(true);
      setError(null);
      setPrediction(null);
      setProgress(0);
      
      const symbolToUse = symbol.toUpperCase().trim();
      const daysAheadToUse = parseInt(daysAhead) || 5;
      
      // Basic input validation
      if (!symbolToUse) {
        setIsLoading(false);
        setError('Please enter a valid stock symbol');
        return;
      }
      
      if (daysAheadToUse < 1 || daysAheadToUse > 30) {
        setIsLoading(false);
        setError('Days to predict must be between 1 and 30');
        return;
      }
      
      const result = await PredictionService.startPrediction(
        symbolToUse,
        daysAheadToUse
      );
      
      // Save active prediction
      saveActivePrediction({
        taskId: result.taskId,
        symbol: symbolToUse,
        daysAhead: daysAheadToUse
      });
      
      // Start checking status
      startStatusCheck(result.taskId);
      
    } catch (err) {
      setIsLoading(false);
      
      // Handle specific error messages
      if (err.message?.includes('Symbol') && err.message?.includes('not found')) {
        const errorMessage = `Symbol '${symbol}' not found. Please verify the stock symbol is correct and from a supported exchange (NASDAQ or BSE).`;
        setError(errorMessage);
        
        // Also show an alert for immediate visibility
        Alert.alert(
          'Symbol Not Found',
          errorMessage,
          [{ text: 'OK' }]
        );
      } else if (err.message?.includes('pending prediction')) {
        setError('You already have a pending prediction for this symbol. Please wait for it to complete or stop it first.');
      } else if (err.message?.includes('already exists')) {
        setError('A prediction with this ID already exists. Please try again in a moment.');
      } else if (err.message?.includes('API request limit')) {
        setError('API request limit reached. Please try again in a minute.');
        
        // Also show an alert for API limit errors
        Alert.alert(
          'API Limit Reached',
          'API request limit reached. Please try again in a minute.',
          [{ text: 'OK' }]
        );
      } else {
        setError(err.message || 'Failed to start prediction. Please try again later.');
      }
      
      console.error('Prediction error:', err);
      
      // Clear any stale active prediction info
      clearActivePrediction();
    }
  };
  
  // Function to refresh all data
  const refreshData = async () => {
    setRefreshing(true);
    setError(null);
    
    // Clear inputs when refreshing
    setSymbol('');
    setDaysAhead('5');
    setPrediction(null);
    
    try {
      // Check active prediction status if there is one
      if (activePredictionInfo) {
        await checkPredictionStatus(activePredictionInfo.taskId);
      }
      
      // Fetch recent predictions
      await fetchRecentPredictions();
    } catch (err) {
      console.error('Error refreshing data:', err);
    } finally {
      setRefreshing(false);
    }
  };
  
  // Pull-to-refresh handler
  const handleRefresh = () => {
    refreshData();
  };
  
  // Stop prediction
  const handleStopPrediction = async () => {
    const taskId = activePredictionInfo?.taskId || (prediction && prediction.taskId);
    if (!taskId) return;
    
    try {
      // First update UI to indicate stopping
      setError("Stopping prediction...");
      
      // Call the API to stop the prediction
      const response = await PredictionService.stopPrediction(taskId);
      
      if (response && (response.status === "stopped" || response.status === "stopping")) {
        // Successfully requested stop
        setIsLoading(false);
        clearInterval(statusCheckInterval.current);
        setError('Prediction stopped successfully');
        
        // Clear active prediction
        clearActivePrediction();
        
        // Delay refresh to ensure backend has updated status
        setTimeout(() => {
          fetchRecentPredictions();
        }, 2000);
      } else {
        setError('Could not stop prediction. Try again.');
      }
    } catch (err) {
      console.error('Error stopping prediction:', err);
      setError('Error stopping prediction: ' + (err.message || 'Unknown error'));
      
      // Even if there's an error, clear the active prediction from local storage
      // This prevents the UI from getting stuck
      clearActivePrediction();
      setIsLoading(false);
      clearInterval(statusCheckInterval.current);
    }
  };
  
  // Register for push notifications
  const registerForPushNotificationsAsync = async () => {
    let token;
    
    if (Platform.OS === 'android') {
      await Notifications.setNotificationChannelAsync('default', {
        name: 'default',
        importance: Notifications.AndroidImportance.MAX,
        vibrationPattern: [0, 250, 250, 250],
        lightColor: '#FF231F7C',
      });
    }
    
    const { status: existingStatus } = await Notifications.getPermissionsAsync();
    let finalStatus = existingStatus;
    
    if (existingStatus !== 'granted') {
      const { status } = await Notifications.requestPermissionsAsync();
      finalStatus = status;
    }
    
    if (finalStatus !== 'granted') {
      Alert.alert('Failed to get push token for notification!');
      return;
    }
    
    token = (await Notifications.getExpoPushTokenAsync()).data;
    console.log('Push Token:', token);
    
    return token;
  };
  
  // Function to format date
  const formatDate = (dateString) => {
    const date = new Date(dateString);
    return date.toLocaleDateString();
  };
  
  // Function to get status color
  const getStatusColor = (status) => {
    switch (status) {
      case 'completed':
        return '#4caf50';
      case 'pending':
      case 'running':
        return '#ff9800';
      case 'failed':
      case 'stopped':
        return '#f44336';
      default:
        return '#757575';
    }
  };
  
  // Render item for recent predictions
  const renderPredictionItem = ({ item }) => (
    <TouchableOpacity 
      style={styles.predictionItem}
      onPress={() => navigation.navigate('PredictionDetails', { id: item._id })}
    >
      <View style={styles.predictionItemContent}>
        <Text style={styles.predictionSymbol}>{item.symbol}</Text>
        <View style={[styles.statusBadge, { backgroundColor: getStatusColor(item.status) }]}>
          <Text style={styles.statusText}>{item.status}</Text>
        </View>
      </View>
      <View style={styles.predictionItemDetails}>
        <Text style={styles.predictionDate}>{formatDate(item.createdAt)}</Text>
        <Text style={styles.predictionDays}>{item.daysAhead} days prediction</Text>
      </View>
    </TouchableOpacity>
  );
  
  // When user focuses the screen (coming back to it)
  useEffect(() => {
    const unsubscribe = navigation.addListener('focus', () => {
      // Fetch recent predictions on focus
      fetchRecentPredictions();
      
      // If there's an active prediction, make sure we're still checking it
      if (activePredictionInfo && !statusCheckInterval.current) {
        setIsLoading(true);
        startStatusCheck(activePredictionInfo.taskId);
      }
    });

    return unsubscribe;
  }, [navigation, activePredictionInfo]);
  
  return (
    <View style={styles.container}>
      <StatusBar barStyle="light-content" backgroundColor={Colors.primary} />
      <View style={styles.header}>
        <TouchableOpacity 
          style={styles.backButton}
          onPress={() => navigation.goBack()}
        >
          <Ionicons name="arrow-back" size={24} color="white" />
        </TouchableOpacity>
        <Text style={styles.headerText}>AI Stock Predictions</Text>
        <TouchableOpacity 
          style={styles.refreshButton}
          onPress={refreshData}
        >
          <Ionicons name="refresh" size={24} color="white" />
        </TouchableOpacity>
      </View>
      
      <ScrollView 
        contentContainerStyle={styles.content}
        refreshControl={
          <RefreshControl 
            refreshing={refreshing}
            onRefresh={handleRefresh}
            colors={[Colors.primary]}
            tintColor={Colors.primary}
          />
        }
      >
        <Text style={styles.sectionTitle}>Enter Stock Details</Text>
        
        <View style={styles.inputContainer}>
          <Text style={styles.inputLabel}>Stock Symbol</Text>
          <TextInput
            style={styles.input}
            value={symbol}
            onChangeText={setSymbol}
            placeholder="Enter stock symbol (e.g., AAPL)"
            placeholderTextColor="#888"
            autoCapitalize="characters"
            maxLength={10}
            editable={!isLoading}
          />
          <Text style={styles.inputNote}>Currently supported: NASDAQ and BSE exchange symbols only</Text>
        </View>
        
        <View style={styles.inputContainer}>
          <Text style={styles.inputLabel}>Days to Predict</Text>
          <TextInput
            style={styles.input}
            value={daysAhead}
            onChangeText={setDaysAhead}
            placeholder="Number of days (1-30)"
            placeholderTextColor="#888"
            keyboardType="number-pad"
            maxLength={2}
            editable={!isLoading}
          />
        </View>
        
        {error && (
          <View style={styles.errorContainer}>
            <Text style={styles.errorText}>{error}</Text>
          </View>
        )}
        
        {isLoading ? (
          <View style={styles.progressContainer}>
            <Text style={styles.progressText}>
              Predicting {activePredictionInfo?.symbol || symbol}...
            </Text>
            <View style={styles.progressBarContainer}>
              <View style={[styles.progressBar, { width: `${progress}%` }]} />
            </View>
            <Text style={styles.progressPercentage}>{progress}%</Text>
            
            <Text style={styles.progressNote}>
              You can navigate away and check back later. The prediction will continue in the background.
            </Text>
            
            <TouchableOpacity 
              style={styles.stopButton}
              onPress={handleStopPrediction}
            >
              <Text style={styles.stopButtonText}>Stop Prediction</Text>
            </TouchableOpacity>
          </View>
        ) : (
          <TouchableOpacity 
            style={styles.predictButton}
            onPress={handleStartPrediction}
          >
            <Text style={styles.predictButtonText}>Start Prediction</Text>
          </TouchableOpacity>
        )}
        
        {prediction && prediction.result && (
          <View style={styles.resultContainer}>
            <Text style={styles.resultTitle}>Prediction Results for {prediction.result.symbol}</Text>
            
            {prediction.result.predictions.length > 0 && (
              <View style={styles.chartContainer}>
                <LineChart
                  data={{
                    labels: prediction.result.predictions.map(p => p.date.substring(5)),
                    datasets: [{
                      data: prediction.result.predictions.map(p => p.price)
                    }]
                  }}
                  width={Dimensions.get('window').width - 40}
                  height={220}
                  chartConfig={{
                    backgroundColor: Colors.primary,
                    backgroundGradientFrom: Colors.primary,
                    backgroundGradientTo: '#4680cc',
                    decimalPlaces: 2,
                    color: (opacity = 1) => `rgba(255, 255, 255, ${opacity})`,
                    labelColor: (opacity = 1) => `rgba(255, 255, 255, ${opacity})`,
                    style: {
                      borderRadius: 16
                    },
                    propsForDots: {
                      r: '6',
                      strokeWidth: '2',
                      stroke: '#ffa726'
                    }
                  }}
                  bezier
                  style={styles.chart}
                />
              </View>
            )}
            
            {prediction.result.sentiment && (
              <View style={styles.sentimentContainer}>
                <Text style={styles.sentimentTitle}>Sentiment Analysis</Text>
                
                <View style={styles.sentimentData}>
                  <View style={styles.sentimentItem}>
                    <Text style={styles.sentimentLabel}>Positive</Text>
                    <Text style={[styles.sentimentValue, { color: '#4caf50' }]}>
                      {prediction.result.sentiment.totals.positive || 0}
                    </Text>
                  </View>
                  
                  <View style={styles.sentimentItem}>
                    <Text style={styles.sentimentLabel}>Neutral</Text>
                    <Text style={[styles.sentimentValue, { color: '#ff9800' }]}>
                      {prediction.result.sentiment.totals.neutral || 0}
                    </Text>
                  </View>
                  
                  <View style={styles.sentimentItem}>
                    <Text style={styles.sentimentLabel}>Negative</Text>
                    <Text style={[styles.sentimentValue, { color: '#f44336' }]}>
                      {prediction.result.sentiment.totals.negative || 0}
                    </Text>
                  </View>
                </View>
                
                <Text style={styles.sentimentSummary}>
                  {prediction.result.sentiment.summary}
                </Text>
              </View>
            )}
            
            <TouchableOpacity 
              style={styles.viewDetailsButton}
              onPress={() => navigation.navigate('PredictionHistory')}
            >
              <Text style={styles.viewDetailsButtonText}>View Prediction History</Text>
            </TouchableOpacity>
          </View>
        )}
        
        {/* Recent Predictions Section */}
        <View style={styles.recentPredictionsContainer}>
          <Text style={styles.sectionTitle}>Recent Predictions</Text>
          
          {loadingPredictions ? (
            <ActivityIndicator size="small" color={Colors.primary} style={styles.loadingIndicator} />
          ) : recentPredictions.length > 0 ? (
            <FlatList
              data={recentPredictions}
              renderItem={renderPredictionItem}
              keyExtractor={(item) => item._id}
              scrollEnabled={false}
            />
          ) : (
            <Text style={styles.noPredictionsText}>No recent predictions found</Text>
          )}
          
          <TouchableOpacity 
            style={styles.viewAllButton}
            onPress={() => navigation.navigate('PredictionHistory')}
          >
            <Text style={styles.viewAllButtonText}>View All Predictions</Text>
          </TouchableOpacity>
        </View>
      </ScrollView>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f5f5f5',
  },
  header: {
    backgroundColor: Colors.primary,
    paddingTop: Platform.OS === 'ios' ? 50 : 25,
    paddingBottom: 15,
    paddingHorizontal: 20,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
  },
  backButton: {
    padding: 5,
  },
  refreshButton: {
    padding: 5,
  },
  headerText: {
    color: 'white',
    fontSize: 20,
    fontWeight: 'bold',
    flex: 1,
    textAlign: 'center',
  },
  content: {
    padding: 20,
  },
  sectionTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    marginBottom: 15,
    color: '#333',
  },
  inputContainer: {
    marginBottom: 15,
  },
  inputLabel: {
    fontSize: 16,
    marginBottom: 5,
    color: '#555',
  },
  input: {
    backgroundColor: 'white',
    borderWidth: 1,
    borderColor: '#ddd',
    borderRadius: 5,
    padding: 10,
    fontSize: 16,
  },
  inputNote: {
    fontSize: 12,
    color: '#757575',
    marginTop: 5,
  },
  predictButton: {
    backgroundColor: Colors.primary,
    padding: 15,
    borderRadius: 5,
    alignItems: 'center',
    marginTop: 10,
  },
  predictButtonText: {
    color: 'white',
    fontSize: 16,
    fontWeight: 'bold',
  },
  errorContainer: {
    padding: 10,
    backgroundColor: '#ffebee',
    borderRadius: 5,
    marginBottom: 15,
  },
  errorText: {
    color: '#d32f2f',
    fontSize: 14,
  },
  progressContainer: {
    marginTop: 20,
    alignItems: 'center',
  },
  progressText: {
    fontSize: 16,
    marginBottom: 10,
    color: '#555',
  },
  progressBarContainer: {
    width: '100%',
    height: 10,
    backgroundColor: '#e0e0e0',
    borderRadius: 5,
    overflow: 'hidden',
  },
  progressBar: {
    height: '100%',
    backgroundColor: Colors.primary,
  },
  progressPercentage: {
    marginTop: 5,
    fontSize: 14,
    color: '#555',
  },
  progressNote: {
    marginTop: 10,
    fontSize: 12,
    color: '#666',
    fontStyle: 'italic',
    textAlign: 'center',
  },
  stopButton: {
    marginTop: 15,
    padding: 10,
    backgroundColor: '#f44336',
    borderRadius: 5,
  },
  stopButtonText: {
    color: 'white',
    fontSize: 14,
    fontWeight: 'bold',
  },
  resultContainer: {
    marginTop: 30,
    padding: 15,
    backgroundColor: 'white',
    borderRadius: 10,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 3,
    elevation: 3,
  },
  resultTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    marginBottom: 15,
    color: '#333',
    textAlign: 'center',
  },
  chartContainer: {
    marginVertical: 15,
    alignItems: 'center',
  },
  chart: {
    borderRadius: 16,
  },
  sentimentContainer: {
    marginTop: 20,
  },
  sentimentTitle: {
    fontSize: 16,
    fontWeight: 'bold',
    marginBottom: 10,
    color: '#333',
  },
  sentimentData: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: 15,
  },
  sentimentItem: {
    alignItems: 'center',
    flex: 1,
  },
  sentimentLabel: {
    fontSize: 14,
    color: '#555',
    marginBottom: 5,
  },
  sentimentValue: {
    fontSize: 18,
    fontWeight: 'bold',
  },
  sentimentSummary: {
    fontSize: 14,
    color: '#555',
    lineHeight: 20,
  },
  viewDetailsButton: {
    backgroundColor: Colors.primary,
    padding: 12,
    borderRadius: 5,
    alignItems: 'center',
    marginTop: 20,
  },
  viewDetailsButtonText: {
    color: 'white',
    fontSize: 14,
    fontWeight: 'bold',
  },
  
  // Recent Predictions styles
  recentPredictionsContainer: {
    marginTop: 30,
    backgroundColor: 'white',
    borderRadius: 10,
    padding: 15,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 3,
    elevation: 3,
  },
  loadingIndicator: {
    marginVertical: 20,
  },
  noPredictionsText: {
    textAlign: 'center',
    color: '#757575',
    marginVertical: 20,
    fontStyle: 'italic',
  },
  predictionItem: {
    borderBottomWidth: 1,
    borderBottomColor: '#f0f0f0',
    paddingVertical: 12,
  },
  predictionItemContent: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 5,
  },
  predictionSymbol: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#333',
  },
  statusBadge: {
    paddingHorizontal: 8,
    paddingVertical: 3,
    borderRadius: 12,
  },
  statusText: {
    color: 'white',
    fontSize: 12,
    fontWeight: '500',
  },
  predictionItemDetails: {
    flexDirection: 'row',
    justifyContent: 'space-between',
  },
  predictionDate: {
    fontSize: 12,
    color: '#757575',
  },
  predictionDays: {
    fontSize: 12,
    color: '#757575',
  },
  viewAllButton: {
    marginTop: 15,
    padding: 10,
    backgroundColor: '#f0f0f0',
    borderRadius: 5,
    alignItems: 'center',
  },
  viewAllButtonText: {
    color: Colors.primary,
    fontWeight: '600',
  },
});

export default AIPredictionScreen; 