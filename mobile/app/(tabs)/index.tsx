import React, { useEffect, useState } from 'react';
import { View, Text, Button, StyleSheet, ScrollView, ActivityIndicator, TouchableOpacity, useColorScheme } from 'react-native';
import { useRouter } from 'expo-router';
import axios from 'axios';
import { MaterialCommunityIcons } from '@expo/vector-icons';

// Instance axios similaire à celle du frontend web
// Remplacez 'localhost' par l'adresse IP locale de votre PC (trouvée avec ipconfig).
// Exemple : baseURL: 'http://172.20.10.2:8000/',
const api = axios.create({
  baseURL: 'http://172.20.10.2:8000/', // <-- IP locale du PC, pas 'localhost'
  timeout: 180000,
});

export default function HomeScreen() {
  const router = useRouter();
  const [apiLoading, setApiLoading] = useState(true);
  const [apiError, setApiError] = useState<string | null>(null);
  const colorScheme = useColorScheme();
  const [theme, setTheme] = useState<'light' | 'dark'>(colorScheme === 'dark' ? 'dark' : 'light');

  // Vérification santé backend (comme le web)
  useEffect(() => {
    setApiLoading(true);
    api.get('/api/health/')
      .then((res) => {
        if (res.data?.status === 'ok') {
          setApiError(null);
        } else {
          setApiError("Le backend ne répond pas correctement.");
        }
      })
      .catch(() => {
        setApiError("Impossible de contacter le backend. Vérifiez que le serveur est bien lancé.");
      })
      .finally(() => setApiLoading(false));
  }, []);

  // Redirection automatique après authentification réussie
  useEffect(() => {
    // Si vous souhaitez TOUJOURS afficher la page d'accueil "Bienvenue sur ...",
    // il NE FAUT PAS faire de router.replace('/user') ici ni dans la logique d'authentification.
    // Il suffit de laisser l'utilisateur sur cette page après login.
    // Donc, supprimez toute redirection automatique vers /user après authentification.
    // ...existing code...
  }, [router]);

  // Déconnexion (à adapter selon votre logique d'authentification)
  const handleLogout = () => {
    // Ex: supprimer le token, réinitialiser l'état utilisateur, etc.
    // Puis rediriger vers la page de login
    router.replace('/login');
  };

  // Changement de thème
  const handleToggleTheme = () => {
    setTheme(t => (t === 'light' ? 'dark' : 'light'));
  };

  // Retour en arrière (générique)
  const handleGoBack = () => {
    if (router.canGoBack?.()) router.back();
    else router.push('/diagnostic');
  };

  // Styles dynamiques selon le thème
  const themedStyles = theme === 'dark'
    ? {
        backgroundColor: '#222',
        color: '#fff',
        cardBg: '#2d2d2d',
        borderColor: '#444',
      }
    : {
        backgroundColor: '#f6f7f2',
        color: '#256029',
        cardBg: '#fff',
        borderColor: '#ccc',
      };

  if (apiLoading) {
    return (
      <View style={[styles.loaderContainer, { backgroundColor: themedStyles.backgroundColor }]}>
        {/* Bouton retour même sur écran de chargement */}
        <TouchableOpacity style={styles.backBtn} onPress={handleGoBack}>
          <MaterialCommunityIcons name="arrow-left" size={22} color="#497174" />
          <Text style={styles.backBtnText}>Retour</Text>
        </TouchableOpacity>
        <ActivityIndicator size="large" color="#256029" />
        <Text style={[styles.loaderText, { color: themedStyles.color }]}>
          Analyse en cours, veuillez patienter (cela peut prendre jusqu'à 3 minutes)...
        </Text>
      </View>
    );
  }

  if (apiError) {
    return (
      <View style={[styles.loaderContainer, { backgroundColor: themedStyles.backgroundColor }]}>
        {/* Bouton retour même sur écran d'erreur */}
        <TouchableOpacity style={styles.backBtn} onPress={handleGoBack}>
          <MaterialCommunityIcons name="arrow-left" size={22} color="#497174" />
          <Text style={styles.backBtnText}>Retour</Text>
        </TouchableOpacity>
        <Text style={[styles.errorText, { color: '#c0392b' }]}>{apiError}</Text>
        <Button title="Réessayer" color="#c0392b" onPress={() => {
          setApiError(null);
          setApiLoading(true);
          axios.get('http://172.20.10.2:8000/api/health/')
            .then((res) => {
              if (res.data?.status === 'ok') {
                setApiError(null);
              } else {
                setApiError("Le backend ne répond pas correctement.");
              }
            })
            .catch(() => {
              setApiError("Impossible de contacter le backend. Vérifiez que le serveur est bien lancé.");
            })
            .finally(() => setApiLoading(false));
        }} />
      </View>
    );
  }

  return (
    <ScrollView contentContainerStyle={[styles.container, { backgroundColor: themedStyles.backgroundColor }]}>
      {/* Bouton retour toujours visible en haut */}
      <View style={styles.topBar}>
        <TouchableOpacity style={styles.backBtn} onPress={handleGoBack}>
          <MaterialCommunityIcons name="arrow-left" size={22} color="#497174" />
          <Text style={styles.backBtnText}>Retour</Text>
        </TouchableOpacity>
        <TouchableOpacity style={styles.topBarBtn} onPress={handleToggleTheme}>
          <MaterialCommunityIcons name="theme-light-dark" size={22} color={theme === 'dark' ? "#fff" : "#497174"} />
          <Text style={[styles.topBarBtnText, { color: theme === 'dark' ? "#fff" : "#497174" }]}>Thème</Text>
        </TouchableOpacity>
        <TouchableOpacity style={styles.topBarBtn} onPress={handleLogout}>
          <MaterialCommunityIcons name="logout" size={22} color="#c0392b" />
          <Text style={[styles.topBarBtnText, { color: "#c0392b" }]}>Déconnexion</Text>
        </TouchableOpacity>
      </View>
      <Text style={[styles.title, { color: themedStyles.color }]}>Bienvenue sur Tomato Disease Detection</Text>
      <Text style={[styles.subtitle, { color: themedStyles.color }]}>
        Application mobile pour diagnostiquer les maladies de la tomate.
      </Text>
      <View style={styles.buttonContainer}>
        <Button
          title="Faire un diagnostic"
          color="#256029"
          onPress={() => router.push('/diagnostic')}
        />
      </View>
      {/* ...aucun autre bouton... */}
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: {
    flexGrow: 1,
    padding: 24,
    alignItems: 'center',
    justifyContent: 'center',
  },
  topBar: {
    flexDirection: 'row',
    width: '100%',
    justifyContent: 'flex-end',
    marginBottom: 10,
    gap: 10,
    alignItems: 'center',
  },
  backBtn: {
    flexDirection: 'row',
    alignItems: 'center',
    marginRight: 'auto',
    paddingVertical: 6,
    paddingHorizontal: 10,
    borderRadius: 16,
    backgroundColor: '#e8f5e9',
  },
  backBtnText: {
    marginLeft: 6,
    fontWeight: 'bold',
    fontSize: 15,
    color: '#497174',
  },
  topBarBtn: {
    flexDirection: 'row',
    alignItems: 'center',
    marginLeft: 12,
    paddingVertical: 6,
    paddingHorizontal: 10,
    borderRadius: 16,
    backgroundColor: '#e8f5e9',
  },
  topBarBtnText: {
    marginLeft: 6,
    fontWeight: 'bold',
    fontSize: 15,
  },
  title: {
    fontSize: 26,
    fontWeight: 'bold',
    marginBottom: 18,
    textAlign: 'center',
  },
  subtitle: {
    fontSize: 16,
    marginBottom: 32,
    textAlign: 'center',
  },
  buttonContainer: {
    width: '100%',
    marginBottom: 18,
  },
  loaderContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    padding: 24,
  },
  loaderText: {
    marginTop: 18,
    fontSize: 16,
    fontWeight: '600',
    textAlign: 'center',
  },
  errorText: {
    color: '#c0392b',
    fontWeight: 'bold',
    fontSize: 16,
    marginBottom: 18,
    textAlign: 'center',
  },
});

// IMPORTANT : Pour que le mobile communique avec le backend :
// 1. Côté mobile : utilisez l'IP locale du PC dans toutes les URLs d'API (jamais 'localhost').
// 2. Côté backend Django :
//    - Installez django-cors-headers : pip install django-cors-headers
//    - Ajoutez 'corsheaders' dans INSTALLED_APPS et 'corsheaders.middleware.CorsMiddleware' en haut de MIDDLEWARE.
//    - Ajoutez dans settings.py :
//        CORS_ALLOW_ALL_ORIGINS = True
//        ALLOWED_HOSTS = ['*']
//    - Lancez le serveur avec : python manage.py runserver 0.0.0.0:8000
// 3. Réseau :
//    - Le PC et le mobile doivent être sur le même réseau Wi-Fi.
//    - Le pare-feu Windows doit autoriser les connexions entrantes sur le port 8000.
// 4. Vérifiez que le backend affiche bien : "Starting development server at http://0.0.0.0:8000/"
// Le message "Bad Request: /api/auth/login/" avec un code 400 signifie que le backend reçoit bien la requête,
// mais que les données envoyées ne sont pas correctes ou incomplètes.
// Pour corriger ce problème :
// 1. Vérifiez que le corps de la requête POST correspond exactement à ce que le backend attend (noms des champs, format).
// 2. Exemple : si le backend Django attend {"username": "...", "password": "..."} et non {"identifier": "...", "password": "..."},
//    il faut adapter la requête côté mobile :

// --- AUTH LOGIC ---
// Remplacez dans votre code d'authentification mobile :
// Remplacez 'votreNomUtilisateur' par la variable contenant le nom d'utilisateur réel
const username = "votreNomUtilisateur"; // à remplacer par la vraie valeur ou variable
const password = "votreMotDePasse"; // à remplacer par la vraie valeur ou variable

// Exemple d'appels API adaptés aux attentes classiques d'un backend Django REST

// Authentification (login)
async function login(username: string, password: string) {
  return api.post("api/auth/login/", {
    username, // <-- champ attendu par Django REST Auth ou SimpleJWT
    password,
  });
}

// Inscription (register)
async function register(username: string, email: string, password1: string, password2: string) {
  return api.post("/api/auth/registration/", {
    username,
    email,
    password1,
    password2,
  });
}

// Upload diagnostic (image)
async function uploadDiagnostic(imageUri: string) {
  const formData = new FormData();
  formData.append("image", {
    uri: imageUri,
    name: "photo.jpg",
    type: "image/jpeg",
  } as any);
  return api.post("/api/diagnostics/upload/", formData, {
    headers: { "Content-Type": "multipart/form-data" },
  });
}

// Récupérer l'historique des diagnostics
async function getHistory(username: string) {
  return api.get(`/api/diagnostics/history/?username=${encodeURIComponent(username)}`);
}

// Récupérer les prévisions météo
async function getForecast() {
  return api.get("/api/meteo/forecast/");
}

// Vérification santé backend
async function checkHealth() {
  return api.get("/api/health/");
}




































































