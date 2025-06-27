import React, { use, useState } from 'react';
import {
  View,
  Text,
  StyleSheet,
  Image,
  TextInput,
  TouchableOpacity,
  ActivityIndicator,
  KeyboardAvoidingView,
  Platform,
  ScrollView,
  TouchableWithoutFeedback,
  Keyboard,
  Alert,
} from 'react-native';
import { useRouter } from 'expo-router';
import { LinearGradient } from 'expo-linear-gradient';
import axios from 'axios';

// TODO: Replace with production backend URL or use env variable for deployment
const api = axios.create({
  baseURL: 'http://localhost:8000/',
  timeout: 180000,
});

export default function LoginScreen() {
  const [identifier, setIdentifier] = useState('');
  const [password, setPassword] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const router = useRouter();

  // TODO: Use secure storage for token/session if needed for protected routes

  const handleLogin = async () => {
    setLoading(true);
    setError('');
    try {
      const response = await api.post('/api/auth/login/', {
        username: identifier,
        password,
      });
      const user = response.data?.user;
      if (user && user.id) {
        // TODO: Store user/token in secure storage or context for session persistence
        if (user.is_admin) {
          router.replace('/admin');
        } else {
          router.replace('/user');
        }
      } else {
        setError("Identifiants invalides.");
      }
    } catch (e) {
      if (e.response?.status === 401) {
        setError("Identifiants invalides.");
      } else if (e.code === 'ECONNABORTED') {
        setError("Le serveur met trop de temps à répondre. Veuillez réessayer.");
      } else {
        setError("Erreur réseau ou serveur.");
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
            <Image source={{ uri: 'https://cdn-icons-png.flaticon.com/512/2909/2909769.png' }} style={styles.logo} />
            <Text style={styles.title}>Connexion AGROTOM</Text>
            <Text style={styles.helper}>Nom d'utilisateur ou email</Text>
            <TextInput
              style={styles.input}
              placeholder="Entrez votre nom d'utilisateur ou email"
              placeholderTextColor="#888"
              value={identifier}
              onChangeText={setIdentifier}
              autoCapitalize="none"
              autoCorrect={false}
              returnKeyType="next"
              accessibilityLabel="Nom d'utilisateur ou email"
              textContentType="username"
              importantForAutofill="yes"
            />
            <Text style={styles.helper}>Mot de passe</Text>
            <TextInput
              style={styles.input}
              placeholder="Entrez votre mot de passe"
              placeholderTextColor="#888"
              value={password}
              onChangeText={setPassword}
              secureTextEntry
              returnKeyType="done"
              accessibilityLabel="Mot de passe"
              textContentType="password"
              importantForAutofill="yes"
            />
            <TouchableOpacity
              style={[styles.button, loading && { opacity: 0.7 }]}
              onPress={handleLogin}
              disabled={loading || !identifier || !password}
              activeOpacity={0.85}
            >
              {loading ? <ActivityIndicator color="#fff" /> : <Text style={styles.buttonText}>Se connecter</Text>}
            </TouchableOpacity>
            <TouchableOpacity onPress={() => router.push('/register')}>
              <Text style={styles.link}>Pas de compte ? S'inscrire</Text>
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
  logo: {
    width: 90,
    height: 90,
    marginBottom: 18,
    borderRadius: 20,
    backgroundColor: '#e8f5e9',
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