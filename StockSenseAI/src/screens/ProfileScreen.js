import React, { useState, useEffect } from 'react';
import { View, Text, StyleSheet, ScrollView, TouchableOpacity, Image, Alert } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import Header from '../components/Header';
import Colors from '../constants/colors';
import { useAuth } from '../context/AuthContext';
import * as ImagePicker from 'expo-image-picker';

const ProfileScreen = ({ navigation }) => {
  const { user, logout, updateProfile } = useAuth();
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
        setIsLoading(true);
        // Update profile picture in the backend
        const updateResult = await updateProfile({
          profilePicture: result.assets[0].uri
        });
        
        if (updateResult.success) {
          Alert.alert('Success', 'Profile picture updated successfully');
        } else {
          Alert.alert('Error', updateResult.error || 'Failed to update profile picture');
        }
      }
    } catch (error) {
      console.error('Error selecting image:', error);
      Alert.alert('Error', 'Failed to select profile picture');
    } finally {
      setIsLoading(false);
    }
  };

  // Profile menu items
  const menuItems = [
    {
      id: 'account',
      title: 'Account Information',
      icon: 'person-circle',
      screen: 'AccountInfo',
    },
    {
      id: 'preferences',
      title: 'Preferences',
      icon: 'options',
      screen: 'Preferences',
    },
    {
      id: 'notifications',
      title: 'Notifications',
      icon: 'notifications',
      screen: 'Notifications',
    },
    {
      id: 'privacy',
      title: 'Privacy & Security',
      icon: 'shield-checkmark',
      screen: 'PrivacySecurity',
    },
    {
      id: 'help',
      title: 'Help & Support',
      icon: 'help-circle',
      screen: 'HelpSupport',
    },
    {
      id: 'guide',
      title: 'User Guide',
      icon: 'book',
      screen: 'Guid',
    },
    {
      id: 'about',
      title: 'About',
      icon: 'information-circle',
      screen: 'About',
    },
  ];

  const renderMenuItem = (item) => (
    <TouchableOpacity
      key={item.id}
      style={styles.menuItem}
      onPress={() => navigation.navigate(item.screen)}
    >
      <View style={styles.menuItemLeft}>
        <Ionicons name={item.icon} size={24} color={Colors.primary} style={styles.menuItemIcon} />
        <Text style={styles.menuItemText}>{item.title}</Text>
      </View>
      <Ionicons name="chevron-forward" size={20} color={Colors.darkGray} />
    </TouchableOpacity>
  );

  return (
    <View style={styles.container}>
      <Header title="Profile" />

      <ScrollView style={styles.content} showsVerticalScrollIndicator={false}>
        <View style={styles.profileHeader}>
          <TouchableOpacity onPress={handleSelectProfilePicture} disabled={isLoading}>
            <Image 
              source={{ uri: user?.profilePicture || 'https://via.placeholder.com/100' }} 
              style={styles.avatar} 
            />
          </TouchableOpacity>
          <View style={styles.profileInfo}>
            <Text style={styles.userName}>{user?.name || 'User'}</Text>
            <Text style={styles.userEmail}>{user?.email || 'No email provided'}</Text>
            <Text style={styles.memberSince}>Member since {formatDate(user?.createdAt)}</Text>
          </View>
        </View>

        <View style={styles.menuContainer}>
          {menuItems.map(renderMenuItem)}
        </View>

        <TouchableOpacity style={styles.logoutButton} onPress={handleLogout}>
          <Ionicons name="log-out" size={20} color={Colors.white} style={styles.logoutIcon} />
          <Text style={styles.logoutText}>Log Out</Text>
        </TouchableOpacity>

        <Text style={styles.versionText}>Version 1.0.0</Text>
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
  profileHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    marginTop: 20,
    marginBottom: 30,
  },
  avatar: {
    width: 80,
    height: 80,
    borderRadius: 40,
    backgroundColor: Colors.gray,
  },
  profileInfo: {
    marginLeft: 20,
    flex: 1,
  },
  userName: {
    fontSize: 20,
    fontWeight: 'bold',
    color: Colors.secondary,
    marginBottom: 4,
  },
  userEmail: {
    fontSize: 14,
    color: Colors.darkGray,
    marginBottom: 4,
  },
  memberSince: {
    fontSize: 12,
    color: Colors.darkGray,
  },
  menuContainer: {
    backgroundColor: Colors.white,
    borderRadius: 12,
    overflow: 'hidden',
    marginBottom: 20,
    shadowColor: Colors.black,
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 2,
  },
  menuItem: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingVertical: 16,
    paddingHorizontal: 20,
    borderBottomWidth: 1,
    borderBottomColor: Colors.gray,
  },
  menuItemLeft: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  menuItemIcon: {
    marginRight: 16,
  },
  menuItemText: {
    fontSize: 16,
    color: Colors.secondary,
  },
  logoutButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: Colors.primary,
    borderRadius: 8,
    paddingVertical: 14,
    marginBottom: 20,
  },
  logoutIcon: {
    marginRight: 8,
  },
  logoutText: {
    fontSize: 16,
    fontWeight: '600',
    color: Colors.white,
  },
  versionText: {
    textAlign: 'center',
    fontSize: 12,
    color: Colors.darkGray,
    marginBottom: 30,
  },
});

export default ProfileScreen; 