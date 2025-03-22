import React, { useState, useRef, useEffect } from 'react';
import {
  View,
  Text,
  StyleSheet,
  TextInput,
  TouchableOpacity,
  FlatList,
  KeyboardAvoidingView,
  Platform,
  SafeAreaView,
  Image,
  Keyboard,
  TouchableWithoutFeedback
} from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import * as ImagePicker from 'expo-image-picker';
import Header from '../../components/Header';
import Colors from '../../constants/colors';

const LiveChatScreen = ({ navigation }) => {
  const [message, setMessage] = useState('');
  const [messages, setMessages] = useState([
    {
      id: '1',
      text: 'Hello! How can I help you today?',
      sender: 'support',
      timestamp: new Date(),
    },
  ]);
  const [image, setImage] = useState(null);
  const flatListRef = useRef(null);

  const pickImage = async () => {
    const { status } = await ImagePicker.requestMediaLibraryPermissionsAsync();
    if (status !== 'granted') {
      alert('Sorry, we need camera roll permissions to make this work!');
      return;
    }

    const result = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ImagePicker.MediaTypeOptions.Images,
      allowsEditing: true,
      aspect: [4, 3],
      quality: 1,
    });

    if (!result.canceled) {
      setImage(result.assets[0].uri);
    }
  };

  const sendMessage = () => {
    if (message.trim() || image) {
      const newMessage = {
        id: Date.now().toString(),
        text: message.trim(),
        image: image,
        sender: 'user',
        timestamp: new Date(),
      };
      setMessages([...messages, newMessage]);
      setMessage('');
      setImage(null);
      Keyboard.dismiss();

      setTimeout(() => {
        const supportResponse = {
          id: (Date.now() + 1).toString(),
          text: 'Thank you for your message. Our support team will respond shortly.',
          sender: 'support',
          timestamp: new Date(),
        };
        setMessages((prev) => [...prev, supportResponse]);
      }, 1000);
    }
  };

  const renderMessage = ({ item }) => (
    <View style={[styles.messageContainer, item.sender === 'user' ? styles.userMessage : styles.supportMessage]}>
      <View style={[styles.messageBubble, item.sender === 'user' ? styles.userBubble : styles.supportBubble]}>
        {item.image && (
          <Image source={{ uri: item.image }} style={styles.messageImage} resizeMode="cover" />
        )}
        {item.text ? (
          <Text style={[styles.messageText, item.sender === 'user' ? styles.userMessageText : styles.supportMessageText]}>
            {item.text}
          </Text>
        ) : null}
      </View>
      <Text style={styles.timestamp}>
        {item.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
      </Text>
    </View>
  );

  return (
    <KeyboardAvoidingView
      style={styles.container}
      behavior={Platform.OS === 'ios' ? 'padding' : 'height'}
      keyboardVerticalOffset={Platform.OS === 'ios' ? 100 : 30}
    >
      <TouchableWithoutFeedback onPress={Keyboard.dismiss}>
        <SafeAreaView style={styles.container}>
          <Header title="Live Chat Support" showBackButton={true} onBackPress={() => navigation.goBack()} />

          <View style={styles.chatContainer}>
            <FlatList
              ref={flatListRef}
              data={messages}
              renderItem={renderMessage}
              keyExtractor={(item) => item.id}
              style={styles.messagesList}
              contentContainerStyle={styles.messagesContainer}
              onContentSizeChange={() => flatListRef.current?.scrollToEnd()}
              keyboardShouldPersistTaps="handled"
            />
          </View>

          <View style={styles.inputContainer}>
            <TouchableOpacity style={styles.attachButton} onPress={pickImage}>
              <Ionicons name="image" size={24} color={Colors.primary} />
            </TouchableOpacity>

            <View style={styles.inputWrapper}>
              {image && (
                <View style={styles.imagePreview}>
                  <Image source={{ uri: image }} style={styles.previewImage} resizeMode="cover" />
                  <TouchableOpacity style={styles.removeImage} onPress={() => setImage(null)}>
                    <Ionicons name="close-circle" size={20} color={Colors.white} />
                  </TouchableOpacity>
                </View>
              )}
              <TextInput
                style={styles.input}
                value={message}
                onChangeText={setMessage}
                placeholder="Type your message..."
                placeholderTextColor={Colors.darkGray}
                multiline
              />
            </View>

            <TouchableOpacity
              style={[styles.sendButton, (!message.trim() && !image) && styles.sendButtonDisabled]}
              onPress={sendMessage}
              disabled={!message.trim() && !image}
            >
              <Ionicons name="send" size={24} color={message.trim() || image ? Colors.primary : Colors.darkGray} />
            </TouchableOpacity>
          </View>
        </SafeAreaView>
      </TouchableWithoutFeedback>
    </KeyboardAvoidingView>
  );
};

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: Colors.background },
  chatContainer: { flex: 1 },
  messagesList: { flex: 1 },
  messagesContainer: { padding: 10 ,flex: 1 , marginBottom:0 },
  messageContainer: { marginBottom: 16, maxWidth: '80%' },
  userMessage: { alignSelf: 'flex-end' },
  supportMessage: { alignSelf: 'flex-start' },
  messageBubble: { padding: 12, borderRadius: 16 },
  userBubble: { backgroundColor: Colors.primary },
  supportBubble: { backgroundColor: Colors.white, borderWidth: 1, borderColor: Colors.gray },
  messageText: { fontSize: 16 },
  userMessageText: { color: Colors.white },
  supportMessageText: { color: Colors.secondary },
  messageImage: { width: 200, height: 200, borderRadius: 8, marginBottom: 8 },
  timestamp: { fontSize: 12, color: Colors.darkGray, marginTop: 4, alignSelf: 'flex-end' },
  inputContainer: { flexDirection: 'row', padding: 12, backgroundColor: Colors.white, borderTopWidth: 1, borderTopColor: Colors.gray, alignItems: 'flex-end' },
  attachButton: { width: 40, height: 40, borderRadius: 20, backgroundColor: Colors.background, justifyContent: 'center', alignItems: 'center', marginRight: 8 },
  inputWrapper: { flex: 1, backgroundColor: Colors.background, borderRadius: 20, paddingHorizontal: 16, paddingVertical: 8, marginRight: 8, maxHeight: 120 },
  imagePreview: { position: 'relative', marginBottom: 8 },
  previewImage: { width: 100, height: 100, borderRadius: 8 },
  removeImage: { position: 'absolute', top: -8, right: -8, backgroundColor: Colors.primary, borderRadius: 12 },
  input: { fontSize: 16, color: Colors.secondary, maxHeight: 100 },
  sendButton: { width: 40, height: 40, borderRadius: 20, backgroundColor: Colors.background, justifyContent: 'center', alignItems: 'center' },
  sendButtonDisabled: { opacity: 0.5 },
});

export default LiveChatScreen;
