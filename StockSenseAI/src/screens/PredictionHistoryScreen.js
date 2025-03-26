import React, { useState, useEffect } from 'react';
import {
  View,
  Text,
  StyleSheet,
  FlatList,
  TouchableOpacity,
  ActivityIndicator,
  RefreshControl,
  Platform,
  StatusBar,
  Alert
} from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import * as PredictionService from '../services/predictionService';
import Colors from '../constants/colors';

const PredictionHistoryScreen = ({ navigation }) => {
  const [predictions, setPredictions] = useState([]);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [error, setError] = useState(null);
  const [selectedPredictions, setSelectedPredictions] = useState([]);
  const [isSelectionMode, setIsSelectionMode] = useState(false);
  const [deleting, setDeleting] = useState(false);

  const loadPredictions = async () => {
    try {
      setError(null);
      const data = await PredictionService.getUserPredictions();
      setPredictions(data);
      // Clear selection when reloading
      setSelectedPredictions([]);
      setIsSelectionMode(false);
    } catch (err) {
      setError('Failed to load predictions. Please try again.');
      console.error('Error loading predictions:', err);
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  };

  useEffect(() => {
    loadPredictions();
  }, []);

  const onRefresh = () => {
    setRefreshing(true);
    loadPredictions();
  };

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

  const formatDate = (dateString) => {
    const date = new Date(dateString);
    return date.toLocaleDateString() + ' ' + date.toLocaleTimeString();
  };

  // Handle selection of a prediction item
  const toggleSelection = (id) => {
    setSelectedPredictions(prevSelected => {
      if (prevSelected.includes(id)) {
        const filtered = prevSelected.filter(itemId => itemId !== id);
        if (filtered.length === 0) {
          // If no items left selected, exit selection mode
          setIsSelectionMode(false);
        }
        return filtered;
      } else {
        return [...prevSelected, id];
      }
    });
  };

  // Toggle selection mode
  const toggleSelectionMode = () => {
    setIsSelectionMode(!isSelectionMode);
    if (isSelectionMode) {
      // Clear selections when exiting selection mode
      setSelectedPredictions([]);
    }
  };

  // Delete selected predictions
  const deleteSelectedPredictions = async () => {
    if (selectedPredictions.length === 0) {
      return;
    }

    Alert.alert(
      'Delete Predictions',
      `Are you sure you want to delete ${selectedPredictions.length} prediction${selectedPredictions.length > 1 ? 's' : ''}?`,
      [
        { text: 'Cancel', style: 'cancel' },
        {
          text: 'Delete',
          style: 'destructive',
          onPress: async () => {
            try {
              setDeleting(true);
              await PredictionService.deleteMultiplePredictions(selectedPredictions);
              
              // Update UI
              setPredictions(prev => 
                prev.filter(item => !selectedPredictions.includes(item._id))
              );
              
              // Exit selection mode
              setSelectedPredictions([]);
              setIsSelectionMode(false);
              
              // Show success message
              Alert.alert('Success', 'Predictions deleted successfully');
            } catch (err) {
              console.error('Error deleting predictions:', err);
              Alert.alert('Error', 'Failed to delete predictions. Please try again.');
            } finally {
              setDeleting(false);
            }
          }
        }
      ]
    );
  };

  // Delete a single prediction
  const deletePrediction = async (id) => {
    Alert.alert(
      'Delete Prediction',
      'Are you sure you want to delete this prediction?',
      [
        { text: 'Cancel', style: 'cancel' },
        {
          text: 'Delete',
          style: 'destructive',
          onPress: async () => {
            try {
              setDeleting(true);
              await PredictionService.deletePrediction(id);
              
              // Update UI
              setPredictions(prev => prev.filter(item => item._id !== id));
              
              // Show success message
              Alert.alert('Success', 'Prediction deleted successfully');
            } catch (err) {
              console.error('Error deleting prediction:', err);
              Alert.alert('Error', 'Failed to delete prediction. Please try again.');
            } finally {
              setDeleting(false);
            }
          }
        }
      ]
    );
  };

  // Render each prediction item
  const renderPredictionItem = ({ item }) => {
    return (
      <TouchableOpacity
        style={[
          styles.predictionItem,
          selectedPredictions.includes(item._id) && styles.selectedItem
        ]}
        onPress={() => {
          if (isSelectionMode) {
            toggleSelection(item._id);
          } else {
            navigation.navigate('PredictionDetails', { id: item._id });
          }
        }}
        onLongPress={() => {
          if (!isSelectionMode) {
            setIsSelectionMode(true);
            toggleSelection(item._id);
          }
        }}
      >
        <View style={styles.predictionHeader}>
          {isSelectionMode && (
            <TouchableOpacity 
              style={styles.checkboxContainer}
              onPress={() => toggleSelection(item._id)}
            >
              <View style={[
                styles.checkbox,
                selectedPredictions.includes(item._id) && styles.checkboxSelected
              ]}>
                {selectedPredictions.includes(item._id) && (
                  <Ionicons name="checkmark" size={16} color="white" />
                )}
              </View>
            </TouchableOpacity>
          )}
          <Text style={styles.symbolText}>{item.symbol}</Text>
          <View style={[styles.statusBadge, { backgroundColor: getStatusColor(item.status) }]}>
            <Text style={styles.statusText}>{item.status}</Text>
          </View>
        </View>
        
        <View style={styles.predictionInfo}>
          <Text style={styles.infoLabel}>Prediction Period:</Text>
          <Text style={styles.infoValue}>{item.daysAhead} days</Text>
        </View>
        
        <View style={styles.predictionInfo}>
          <Text style={styles.infoLabel}>Created:</Text>
          <Text style={styles.infoValue}>{formatDate(item.createdAt)}</Text>
        </View>
        
        <View style={styles.viewButton}>
          <Text style={styles.viewButtonText}>View Details</Text>
          <Ionicons name="chevron-forward" size={16} color={Colors.primary} />
        </View>
      </TouchableOpacity>
    );
  };

  return (
    <View style={styles.container}>
      <StatusBar barStyle="light-content" backgroundColor={Colors.primary} />
      <View style={styles.header}>
        <TouchableOpacity 
          style={styles.backButton}
          onPress={() => {
            if (isSelectionMode) {
              // If in selection mode, exit selection mode instead of going back
              setIsSelectionMode(false);
              setSelectedPredictions([]);
            } else {
              navigation.goBack();
            }
          }}
        >
          <Ionicons name={isSelectionMode ? "close" : "arrow-back"} size={24} color="white" />
        </TouchableOpacity>
        <Text style={styles.headerText}>
          {isSelectionMode 
            ? `Selected (${selectedPredictions.length})` 
            : 'Prediction History'}
        </Text>
        {isSelectionMode ? (
          <TouchableOpacity 
            style={styles.actionButton}
            onPress={deleteSelectedPredictions}
            disabled={selectedPredictions.length === 0 || deleting}
          >
            {deleting ? (
              <ActivityIndicator size="small" color="white" />
            ) : (
              <Ionicons name="trash-outline" size={24} color="white" />
            )}
          </TouchableOpacity>
        ) : (
          <TouchableOpacity 
            style={styles.actionButton}
            onPress={toggleSelectionMode}
          >
            <Ionicons name="ellipsis-vertical" size={24} color="white" />
          </TouchableOpacity>
        )}
      </View>
      
      {loading && !refreshing ? (
        <View style={styles.loadingContainer}>
          <ActivityIndicator size="large" color={Colors.primary} />
          <Text style={styles.loadingText}>Loading predictions...</Text>
        </View>
      ) : error ? (
        <View style={styles.errorContainer}>
          <Text style={styles.errorText}>{error}</Text>
          <TouchableOpacity 
            style={styles.retryButton}
            onPress={loadPredictions}
          >
            <Text style={styles.retryButtonText}>Retry</Text>
          </TouchableOpacity>
        </View>
      ) : predictions.length === 0 ? (
        <View style={styles.emptyContainer}>
          <Ionicons name="analytics-outline" size={64} color="#bbb" />
          <Text style={styles.emptyText}>No predictions yet</Text>
          <TouchableOpacity 
            style={styles.newPredictionButton}
            onPress={() => navigation.navigate('AIPrediction')}
          >
            <Text style={styles.newPredictionButtonText}>Make a Prediction</Text>
          </TouchableOpacity>
        </View>
      ) : (
        <FlatList
          data={predictions}
          renderItem={renderPredictionItem}
          keyExtractor={item => item._id}
          contentContainerStyle={styles.listContent}
          refreshControl={
            <RefreshControl
              refreshing={refreshing}
              onRefresh={onRefresh}
              colors={[Colors.primary]}
            />
          }
        />
      )}
      
      {!isSelectionMode && (
        <TouchableOpacity 
          style={styles.fabButton}
          onPress={() => navigation.navigate('AIPrediction')}
        >
          <Ionicons name="add" size={24} color="white" />
        </TouchableOpacity>
      )}
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
  headerText: {
    color: 'white',
    fontSize: 20,
    fontWeight: 'bold',
    flex: 1,
    textAlign: 'center',
  },
  actionButton: {
    padding: 5,
  },
  loadingContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  loadingText: {
    marginTop: 10,
    fontSize: 16,
    color: '#555',
  },
  errorContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    padding: 20,
  },
  errorText: {
    fontSize: 16,
    color: '#d32f2f',
    textAlign: 'center',
    marginBottom: 15,
  },
  retryButton: {
    backgroundColor: Colors.primary,
    paddingVertical: 10,
    paddingHorizontal: 20,
    borderRadius: 5,
  },
  retryButtonText: {
    color: 'white',
    fontSize: 14,
    fontWeight: 'bold',
  },
  emptyContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    padding: 20,
  },
  emptyText: {
    fontSize: 18,
    color: '#757575',
    marginTop: 10,
    marginBottom: 20,
  },
  newPredictionButton: {
    backgroundColor: Colors.primary,
    paddingVertical: 12,
    paddingHorizontal: 20,
    borderRadius: 5,
  },
  newPredictionButtonText: {
    color: 'white',
    fontSize: 16,
    fontWeight: 'bold',
  },
  listContent: {
    padding: 15,
  },
  predictionItem: {
    backgroundColor: 'white',
    borderRadius: 10,
    padding: 15,
    marginBottom: 15,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 3,
    elevation: 3,
  },
  selectedItem: {
    backgroundColor: '#e3f2fd',
  },
  predictionHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 10,
  },
  symbolText: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#333',
    flex: 1,
  },
  statusBadge: {
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderRadius: 12,
  },
  statusText: {
    color: 'white',
    fontSize: 12,
    fontWeight: 'bold',
  },
  predictionInfo: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: 5,
  },
  infoLabel: {
    fontSize: 14,
    color: '#666',
  },
  infoValue: {
    fontSize: 14,
    color: '#333',
    fontWeight: '500',
  },
  viewButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'flex-end',
    marginTop: 10,
  },
  viewButtonText: {
    color: Colors.primary,
    fontWeight: '500',
    marginRight: 5,
  },
  fabButton: {
    position: 'absolute',
    bottom: 20,
    right: 20,
    width: 56,
    height: 56,
    borderRadius: 28,
    backgroundColor: Colors.primary,
    justifyContent: 'center',
    alignItems: 'center',
    elevation: 5,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.25,
    shadowRadius: 3.84,
  },
  checkboxContainer: {
    marginRight: 10,
  },
  checkbox: {
    width: 22,
    height: 22,
    borderRadius: 11,
    borderWidth: 2,
    borderColor: Colors.primary,
    justifyContent: 'center',
    alignItems: 'center',
  },
  checkboxSelected: {
    backgroundColor: Colors.primary,
  },
});

export default PredictionHistoryScreen; 