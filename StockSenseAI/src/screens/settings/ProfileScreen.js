import React, { useState, useEffect } from 'react';
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  ScrollView,
  Image,
  Alert,
  ActivityIndicator
} from 'react-native';
import { useAuth } from '../../context/AuthContext';
import { authService } from '../../services/api';
import * as ImagePicker from 'expo-image-picker';

const ProfileScreen = ({ navigation }) => {
  const { userInfo, logout } = useAuth();
  const [isLoading, setIsLoading] = useState(false);
  
  // Format date of birth
  const formatDate = (dateString) => {
    if (!dateString) return 'Not provided';
    
    const date = new Date(dateString);
    if (isNaN(date.getTime())) return 'Invalid date';
    
    return date.toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'long',
      day: 'numeric',
    });
  };

  // Handle logout
  const handleLogout = async () => {
    Alert.alert(
      'Confirm Logout',
      'Are you sure you want to log out?',
      [
        {
          text: 'Cancel',
          style: 'cancel',
        },
        {
          text: 'Logout',
          onPress: () => logout(),
          style: 'destructive',
        },
      ]
    );
  };

  // Handle profile picture selection
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
        Alert.alert('Success', 'Profile picture updated');
      }
    } catch (error) {
      console.error('Error selecting image:', error);
      Alert.alert('Error', 'Failed to select profile picture');
    }
  };

  return (
    <ScrollView style={styles.container}>
      <View style={styles.header}>
        <View style={styles.profileImageContainer}>
          <TouchableOpacity onPress={handleSelectProfilePicture}>
            {userInfo?.profilePicture ? (
              <Image 
                source={{ uri: userInfo.profilePicture }} 
                style={styles.profileImage} 
              />
            ) : (
              <View style={styles.profileImagePlaceholder}>
                <Text style={styles.profileImagePlaceholderText}>
                  {userInfo?.name ? userInfo.name.charAt(0).toUpperCase() : 'U'}
                </Text>
              </View>
            )}
            <View style={styles.editIconContainer}>
              <Text style={styles.editIconText}>✎</Text>
            </View>
          </TouchableOpacity>
        </View>
        <Text style={styles.name}>{userInfo?.name || 'User'}</Text>
        <Text style={styles.email}>{userInfo?.email || 'No email'}</Text>
      </View>
      
      <View style={styles.infoSection}>
        <Text style={styles.sectionTitle}>Personal Information</Text>
        
        <View style={styles.infoItem}>
          <Text style={styles.infoLabel}>Phone Number</Text>
          <Text style={styles.infoValue}>
            {userInfo?.countryCode || ''} {userInfo?.phoneNumber || 'Not provided'}
          </Text>
        </View>
        
        <View style={styles.infoItem}>
          <Text style={styles.infoLabel}>Address</Text>
          <Text style={styles.infoValue}>{userInfo?.address || 'Not provided'}</Text>
        </View>
        
        <View style={styles.infoItem}>
          <Text style={styles.infoLabel}>Date of Birth</Text>
          <Text style={styles.infoValue}>{formatDate(userInfo?.dateOfBirth)}</Text>
        </View>
        
        <View style={styles.infoItem}>
          <Text style={styles.infoLabel}>Joined</Text>
          <Text style={styles.infoValue}>{formatDate(userInfo?.createdAt)}</Text>
        </View>
      </View>
      
      <TouchableOpacity
        style={styles.button}
        onPress={() => navigation.navigate('AccountInfo')}
      >
        <Text style={styles.buttonText}>Edit Profile</Text>
      </TouchableOpacity>
      
      <TouchableOpacity
        style={[styles.button, styles.logoutButton]}
        onPress={handleLogout}
      >
        <Text style={[styles.buttonText, styles.logoutButtonText]}>Logout</Text>
      </TouchableOpacity>
    </ScrollView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#ffffff',
  },
  header: {
    alignItems: 'center',
    paddingVertical: 30,
    borderBottomWidth: 1,
    borderBottomColor: '#f0f0f0',
  },
  profileImageContainer: {
    position: 'relative',
    marginBottom: 15,
  },
  profileImage: {
    width: 100,
    height: 100,
    borderRadius: 50,
  },
  profileImagePlaceholder: {
    width: 100,
    height: 100,
    borderRadius: 50,
    backgroundColor: '#3970BE',
    justifyContent: 'center',
    alignItems: 'center',
  },
  profileImagePlaceholderText: {
    color: '#ffffff',
    fontSize: 40,
    fontWeight: 'bold',
  },
  editIconContainer: {
    position: 'absolute',
    bottom: 0,
    right: 0,
    backgroundColor: '#ffffff',
    width: 30,
    height: 30,
    borderRadius: 15,
    justifyContent: 'center',
    alignItems: 'center',
    borderWidth: 1,
    borderColor: '#ddd',
  },
  editIconText: {
    fontSize: 16,
    color: '#3970BE',
  },
  name: {
    fontSize: 22,
    fontWeight: 'bold',
    marginBottom: 5,
  },
  email: {
    fontSize: 16,
    color: '#666',
  },
  infoSection: {
    padding: 20,
  },
  sectionTitle: {
    fontSize: 18,
    fontWeight: '600',
    marginBottom: 15,
    color: '#333',
  },
  infoItem: {
    paddingVertical: 12,
    borderBottomWidth: 1,
    borderBottomColor: '#f0f0f0',
  },
  infoLabel: {
    fontSize: 14,
    color: '#666',
    marginBottom: 4,
  },
  infoValue: {
    fontSize: 16,
    color: '#333',
  },
  button: {
    backgroundColor: '#3970BE',
    paddingVertical: 14,
    paddingHorizontal: 20,
    borderRadius: 8,
    marginHorizontal: 20,
    marginTop: 20,
    alignItems: 'center',
  },
  buttonText: {
    color: '#ffffff',
    fontSize: 16,
    fontWeight: '600',
  },
  logoutButton: {
    backgroundColor: '#ffffff',
    borderWidth: 1,
    borderColor: '#ff3b30',
    marginBottom: 30,
  },
  logoutButtonText: {
    color: '#ff3b30',
  },
});

export default ProfileScreen; 