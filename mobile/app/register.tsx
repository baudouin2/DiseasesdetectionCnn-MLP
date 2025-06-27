import React, { useState } from 'react';
import {
  View,
  Text,
  StyleSheet,
  TextInput,
  TouchableOpacity,
  ActivityIndicator,
  KeyboardAvoidingView,
  Platform,
  ScrollView,
  TouchableWithoutFeedback,
  Keyboard,
} from 'react-native';
import { useRouter } from 'expo-router';
import { LinearGradient } from 'expo-linear-gradient';
import axios from 'axios';

const api = axios.create({
  baseURL: 'http://172.20.10.2:8000/',
  timeout: 180000,
});

export default function RegisterScreen() {
  const [username, setUsername] = useState('');
  const [email, setEmail] = useState('');
  const [password1, setPassword1] = useState('');
  const [password2, setPassword2] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const router = useRouter();

  const handleRegister = async () => {
    setLoading(true);
    setError('');
    try {
      if (!username || !email || !password1 || !password2) {
        setError("Tous les champs sont requis.");
        setLoading(false);
        return;
      }
      if (password1 !== password2) {
        setError("Les mots de passe ne correspondent pas.");
        setLoading(false);
        return;
      }
      await api.post('/api/auth/register/', {
        username,
        email,
        password1,
        password2,
      });
      router.replace('/login');
    } catch (e: any) {
      if (e.response?.data?.username) {
        setError(e.response.data.username);
      } else if (e.response?.data?.email) {
        setError(e.response.data.email);
      } else if (e.response?.data?.password1) {
        setError(e.response.data.password1);
      } else if (e.response?.data?.password2) {
        setError(e.response.data.password2);
      } else {
        setError("Erreur lors de l'inscription.");
      }
    } finally {
      setLoading(false);
    }
  };

  return (
    <KeyboardAvoidingView
      style={{ flex: 1 }}
      behavior={Platform.OS === 'ios' ? 'padding' : 'height'}
      keyboardVerticalOffset={80}
    >
      <LinearGradient
        colors={['#f4f8f3', '#e8f5e9']}
        style={StyleSheet.absoluteFill}
        start={{ x: 0, y: 0 }}
        end={{ x: 1, y: 1 }}
      />
      <TouchableWithoutFeedback onPress={Keyboard.dismiss} accessible={false}>
        <ScrollView contentContainerStyle={styles.scrollContainer} keyboardShouldPersistTaps="handled">
          <View style={styles.card}>
            <Text style={styles.title}>Inscription AGROTOM</Text>
            <Text style={styles.helper}>Nom d'utilisateur</Text>
            <TextInput
              style={styles.input}
              placeholder="Nom d'utilisateur"
              placeholderTextColor="#888"
              value={username}
              onChangeText={setUsername}
              autoCapitalize="none"
              autoCorrect={false}
            />
            <Text style={styles.helper}>Email</Text>
            <TextInput
              style={styles.input}
              placeholder="Email"
              placeholderTextColor="#888"
              value={email}
              onChangeText={setEmail}
              autoCapitalize="none"
              autoCorrect={false}
              keyboardType="email-address"
            />
            <Text style={styles.helper}>Mot de passe</Text>
            <TextInput
              style={styles.input}
              placeholder="Mot de passe"
              placeholderTextColor="#888"
              value={password1}
              onChangeText={setPassword1}
              secureTextEntry
            />
            <Text style={styles.helper}>Confirmer le mot de passe</Text>
            <TextInput
              style={styles.input}
              placeholder="Confirmer le mot de passe"
              placeholderTextColor="#888"
              value={password2}
              onChangeText={setPassword2}
              secureTextEntry
            />
            <TouchableOpacity
              style={[styles.button, loading && { opacity: 0.7 }]}
              onPress={handleRegister}
              disabled={loading}
              activeOpacity={0.85}
            >
              {loading ? <ActivityIndicator color="#fff" /> : <Text style={styles.buttonText}>S'inscrire</Text>}
            </TouchableOpacity>
            <TouchableOpacity onPress={() => router.replace('/login')}>
              <Text style={styles.link}>Déjà un compte ? Se connecter</Text>
            </TouchableOpacity>
            {error ? <Text style={styles.error}>{error}</Text> : null}
          </View>
        </ScrollView>
      </TouchableWithoutFeedback>
    </KeyboardAvoidingView>
  );
}

const styles = StyleSheet.create({
  flex: { flex: 1, backgroundColor: '#fff8e1' },
  scrollContainer: {
    flexGrow: 1,
    justifyContent: 'center',
    alignItems: 'center',
    padding: 24,
    minHeight: '100%',
  },
  card: {
    width: '100%',
    maxWidth: 400,
    alignItems: 'center',
    backgroundColor: '#fff',
    borderRadius: 16,
    padding: 24,
    shadowColor: '#388e3c',
    shadowOpacity: 0.12,
    shadowRadius: 16,
    shadowOffset: { width: 0, height: 4 },
    elevation: 8,
  },
  title: {
    fontSize: 26,
    fontWeight: 'bold',
    color: '#388e3c',
    marginBottom: 8,
    textAlign: 'center',
  },
  helper: {
    color: '#ff9800',
    fontSize: 18,
    marginBottom: 6,
    textAlign: 'left',
    width: '100%',
    fontWeight: 'bold',
    letterSpacing: 0.2,
  },
  input: {
    width: '100%',
    backgroundColor: '#fff',
    borderRadius: 8,
    padding: 12,
    marginBottom: 12,
    borderWidth: 1.5,
    borderColor: '#ff9800',
    fontSize: 16,
  },
  button: {
    backgroundColor: '#388e3c',
    borderRadius: 8,
    paddingVertical: 12,
    paddingHorizontal: 32,
    alignItems: 'center',
    marginTop: 8,
    width: '100%',
    elevation: 2,
  },
  buttonText: {
    color: '#fff',
    fontWeight: 'bold',
    fontSize: 17,
    letterSpacing: 0.5,
  },
  link: {
    color: '#1976d2',
    fontWeight: 'bold',
    marginTop: 18,
    fontSize: 16,
    textAlign: 'center',
    textDecorationLine: 'underline',
  },
  error: {
    color: '#e53935',
    marginTop: 12,
    fontWeight: 'bold',
    fontSize: 16,
    textAlign: 'center',
  },
});
