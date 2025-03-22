import React, { useState, useEffect } from 'react';
import { 
  View, 
  Text, 
  FlatList, 
  TouchableOpacity, 
  Modal, 
  StyleSheet, 
  TextInput,
  SafeAreaView 
} from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import countryCodes from '../utils/countryCodes';

const CountryCodePicker = ({ selectedCode, onSelectCountry }) => {
  const [modalVisible, setModalVisible] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');
  const [filteredCountries, setFilteredCountries] = useState(countryCodes);
  const [selectedCountry, setSelectedCountry] = useState(
    countryCodes.find(country => country.code === selectedCode) || countryCodes[0]
  );

  useEffect(() => {
    if (searchQuery) {
      const filtered = countryCodes.filter(
        item => 
          item.country.toLowerCase().includes(searchQuery.toLowerCase()) ||
          item.code.includes(searchQuery)
      );
      setFilteredCountries(filtered);
    } else {
      setFilteredCountries(countryCodes);
    }
  }, [searchQuery]);

  const handleSelectCountry = (country) => {
    setSelectedCountry(country);
    onSelectCountry(country.code);
    setModalVisible(false);
  };

  return (
    <View>
      <TouchableOpacity 
        style={styles.countryCodeButton} 
        onPress={() => setModalVisible(true)}
      >
        <Text style={styles.countryCodeText}>
          {selectedCountry.flag} {selectedCountry.code}
        </Text>
        <Ionicons name="chevron-down" size={16} color="#666" />
      </TouchableOpacity>

      <Modal
        animationType="slide"
        transparent={true}
        visible={modalVisible}
        onRequestClose={() => setModalVisible(false)}
      >
        <SafeAreaView style={styles.modalContainer}>
          <View style={styles.modalContent}>
            <View style={styles.modalHeader}>
              <TouchableOpacity 
                style={styles.closeButton}
                onPress={() => setModalVisible(false)}
              >
                <Ionicons name="close" size={24} color="#333" />
              </TouchableOpacity>
              <Text style={styles.modalTitle}>Select Country Code</Text>
            </View>

            <View style={styles.searchContainer}>
              <Ionicons name="search" size={20} color="#666" style={styles.searchIcon} />
              <TextInput
                style={styles.searchInput}
                placeholder="Search country or code..."
                value={searchQuery}
                onChangeText={setSearchQuery}
                autoCapitalize="none"
              />
              {searchQuery ? (
                <TouchableOpacity onPress={() => setSearchQuery('')}>
                  <Ionicons name="close-circle" size={20} color="#666" />
                </TouchableOpacity>
              ) : null}
            </View>

            <FlatList
              data={filteredCountries}
              keyExtractor={(item) => item.code}
              renderItem={({ item }) => (
                <TouchableOpacity
                  style={styles.countryItem}
                  onPress={() => handleSelectCountry(item)}
                >
                  <Text style={styles.countryFlag}>{item.flag}</Text>
                  <View style={styles.countryInfo}>
                    <Text style={styles.countryName}>{item.country}</Text>
                    <Text style={styles.countryCode}>{item.code}</Text>
                  </View>
                </TouchableOpacity>
              )}
              ItemSeparatorComponent={() => <View style={styles.separator} />}
              ListEmptyComponent={() => (
                <View style={styles.emptyContainer}>
                  <Text style={styles.emptyText}>No countries found</Text>
                </View>
              )}
            />
          </View>
        </SafeAreaView>
      </Modal>
    </View>
  );
};

const styles = StyleSheet.create({
  countryCodeButton: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#f0f0f0',
    borderRadius: 8,
    paddingHorizontal: 10,
    paddingVertical: 12,
    marginRight: 10,
    minWidth: 90,
    justifyContent: 'space-between'
  },
  countryCodeText: {
    fontSize: 16,
    color: '#333'
  },
  modalContainer: {
    flex: 1,
    backgroundColor: 'rgba(0, 0, 0, 0.5)',
  },
  modalContent: {
    flex: 1,
    backgroundColor: '#fff',
    marginTop: 50,
    borderTopLeftRadius: 20,
    borderTopRightRadius: 20,
  },
  modalHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    padding: 15,
    borderBottomWidth: 1,
    borderBottomColor: '#eee',
  },
  closeButton: {
    padding: 5,
  },
  modalTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#333',
    marginLeft: 10,
  },
  searchContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#f0f0f0',
    borderRadius: 10,
    margin: 15,
    paddingHorizontal: 10,
  },
  searchIcon: {
    marginRight: 10,
  },
  searchInput: {
    flex: 1,
    height: 50,
    fontSize: 16,
  },
  countryItem: {
    flexDirection: 'row',
    alignItems: 'center',
    padding: 15,
  },
  countryFlag: {
    fontSize: 24,
    marginRight: 15,
  },
  countryInfo: {
    flex: 1,
  },
  countryName: {
    fontSize: 16,
    color: '#333',
  },
  countryCode: {
    fontSize: 14,
    color: '#666',
    marginTop: 2,
  },
  separator: {
    height: 1,
    backgroundColor: '#eee',
  },
  emptyContainer: {
    padding: 20,
    alignItems: 'center',
  },
  emptyText: {
    color: '#666',
    fontSize: 16,
  },
});

export default CountryCodePicker; 