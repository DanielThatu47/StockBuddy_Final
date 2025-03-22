import React, { useState, useEffect } from 'react';
import { View, Text, StyleSheet, TextInput, TouchableOpacity, ScrollView, Image, Alert, ActivityIndicator } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import Header from '../../components/Header';
import Colors from '../../constants/colors';
import { useAuth } from '../../context/AuthContext';
import { apiRequest } from '../../services/api';
import * as ImagePicker from 'expo-image-picker';

const AccountInfoScreen = ({ navigation }) => {
  const { user, updateProfile } = useAuth();
  const [userInfo, setUserInfo] = useState({
    name: '',
    email: '',
    phoneNumber: '',
    avatar: '',
    countryCode: '',
    address: '',
    dateOfBirth: '',
  });
  const [isLoading, setIsLoading] = useState(false);

  useEffect(() => {
    if (user) {
      setUserInfo({
        name: user.name || '',
        email: user.email || '',
        phoneNumber: user.phoneNumber || '',
        avatar: user.profilePicture || '',
        countryCode: user.countryCode || '',
        address: user.address || '',
        dateOfBirth: user.dateOfBirth || '',
      });
    }
  }, [user]);

  const handleSave = async () => {
    try {
      setIsLoading(true);
      const result = await updateProfile(userInfo);
      
      if (result.success) {
        Alert.alert('Success', 'Profile updated successfully');
      } else {
        Alert.alert('Error', result.error || 'Failed to update profile');
      }
    } catch (error) {
      Alert.alert('Error', error.message || 'Failed to update profile');
    } finally {
      setIsLoading(false);
    }
  };

  const handleSelectProfilePicture = async () => {
    try {
      const permissionResult = await ImagePicker.requestMediaLibraryPermissionsAsync();
      
      if (!permissionResult.granted) {
        Alert.alert('Permission required', 'Please grant permission to access photos');
        return;
      }
      
      const result = await ImagePicker.launchImageLibraryAsync({
        mediaTypes: ImagePicker.MediaTypeOptions.Images,
        allowsEditing: true,
        aspect: [1, 1],
        quality: 0.8,
      });
      
      if (!result.canceled) {
        // Here you would upload the image to your backend
        // and update the profile picture URL
        setUserInfo(prev => ({
          ...prev,
          avatar: result.assets[0].uri
        }));
      }
    } catch (error) {
      console.error('Error selecting image:', error);
      Alert.alert('Error', 'Failed to select profile picture');
    }
  };

  return (
    <View style={styles.container}>
      <Header 
        title="Account Information" 
        showBackButton={true}
        onBackPress={() => navigation.goBack()}
      />

      <ScrollView style={styles.content} showsVerticalScrollIndicator={false}>
        <View style={styles.avatarContainer}>
          <Image 
            source={{ uri: userInfo.avatar || 'https://via.placeholder.com/100' }} 
            style={styles.avatar} 
          />
          <TouchableOpacity 
            style={styles.changePhotoButton}
            onPress={handleSelectProfilePicture}
          >
            <Text style={styles.changePhotoText}>Change Photo</Text>
          </TouchableOpacity>
        </View>

        <View style={styles.formContainer}>
          <View style={styles.inputGroup}>
            <Text style={styles.label}>Full Name</Text>
            <TextInput
              style={styles.input}
              value={userInfo.name}
              onChangeText={(text) => setUserInfo({ ...userInfo, name: text })}
              placeholder="Enter your full name"
            />
          </View>

          <View style={styles.inputGroup}>
            <Text style={styles.label}>Email Address</Text>
            <TextInput
              style={[styles.input, styles.disabledInput]}
              value={userInfo.email}
              editable={false}
              placeholder="Enter your email"
              keyboardType="email-address"
              autoCapitalize="none"
            />
          </View>

          <View style={styles.inputGroup}>
            <Text style={styles.label}>Phone Number</Text>
            <TextInput
              style={styles.input}
              value={userInfo.phoneNumber}
              onChangeText={(text) => setUserInfo({ ...userInfo, phoneNumber: text })}
              placeholder="Enter your phone number"
              keyboardType="phone-pad"
            />
          </View>

          <View style={styles.inputGroup}>
            <Text style={styles.label}>Address</Text>
            <TextInput
              style={styles.input}
              value={userInfo.address}
              onChangeText={(text) => setUserInfo({ ...userInfo, address: text })}
              placeholder="Enter your address"
              multiline
            />
          </View>

          <TouchableOpacity 
            style={[styles.saveButton, isLoading && styles.disabledButton]} 
            onPress={handleSave}
            disabled={isLoading}
          >
            {isLoading ? (
              <ActivityIndicator color="#ffffff" />
            ) : (
              <Text style={styles.saveButtonText}>Save Changes</Text>
            )}
          </TouchableOpacity>
        </View>

        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Account Security</Text>
          <TouchableOpacity 
            style={styles.menuItem}
            onPress={() => navigation.navigate('ChangePassword')}
          >
            <View style={styles.menuItemLeft}>
              <Ionicons name="lock-closed" size={24} color={Colors.primary} />
              <Text style={styles.menuItemText}>Change Password</Text>
            </View>
            <Ionicons name="chevron-forward" size={20} color={Colors.darkGray} />
          </TouchableOpacity>
          <TouchableOpacity style={styles.menuItem}>
            <View style={styles.menuItemLeft}>
              <Ionicons name="shield-checkmark" size={24} color={Colors.primary} />
              <Text style={styles.menuItemText}>Two-Factor Authentication</Text>
            </View>
            <Ionicons name="chevron-forward" size={20} color={Colors.darkGray} />
          </TouchableOpacity>
        </View>
      </ScrollView>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: Colors.background,
  },
  content: {
    flex: 1,
    paddingHorizontal: 20,
  },
  avatarContainer: {
    alignItems: 'center',
    marginTop: 20,
    marginBottom: 30,
  },
  avatar: {
    width: 100,
    height: 100,
    borderRadius: 50,
    backgroundColor: Colors.gray,
  },
  changePhotoButton: {
    marginTop: 10,
  },
  changePhotoText: {
    color: Colors.primary,
    fontSize: 16,
    fontWeight: '500',
  },
  formContainer: {
    backgroundColor: Colors.white,
    borderRadius: 12,
    padding: 20,
    marginBottom: 20,
    shadowColor: Colors.black,
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  inputGroup: {
    marginBottom: 20,
  },
  label: {
    fontSize: 14,
    color: Colors.darkGray,
    marginBottom: 8,
  },
  input: {
    borderWidth: 1,
    borderColor: Colors.gray,
    borderRadius: 8,
    padding: 12,
    fontSize: 16,
    color: Colors.secondary,
  },
  saveButton: {
    backgroundColor: Colors.primary,
    borderRadius: 8,
    padding: 16,
    alignItems: 'center',
    marginTop: 10,
  },
  saveButtonText: {
    color: Colors.white,
    fontSize: 16,
    fontWeight: '600',
  },
  section: {
    backgroundColor: Colors.white,
    borderRadius: 12,
    overflow: 'hidden',
    marginBottom: 20,
    shadowColor: Colors.black,
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  sectionTitle: {
    fontSize: 18,
    fontWeight: '600',
    color: Colors.secondary,
    padding: 20,
    paddingBottom: 10,
  },
  menuItem: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    padding: 20,
    borderTopWidth: 1,
    borderTopColor: Colors.gray,
  },
  menuItemLeft: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  menuItemText: {
    fontSize: 16,
    color: Colors.secondary,
    marginLeft: 12,
  },
  disabledInput: {
    backgroundColor: '#f5f5f5',
    color: '#666',
  },
  disabledButton: {
    opacity: 0.7,
  },
});

export default AccountInfoScreen; 