import axios from 'axios';
import AsyncStorage from '@react-native-async-storage/async-storage';

// Configuration de l'URL de l'API (priorité à la variable d'environnement Expo)
const API_URL =
  process.env.EXPO_PUBLIC_API_URL ||
  process.env.API_URL ||
  'http://192.168.1.100:8000/api';

// Création d'une instance Axios avec timeout adapté
const api = axios.create({
  baseURL: API_URL,
  timeout: 180000, // 3 minutes pour les traitements longs
});

// Intercepteur de réponse pour gestion centralisée des erreurs
api.interceptors.response.use(
  response => response,
  error => {
    // Log ou traitement d'erreur global ici si besoin
    return Promise.reject(error);
  }
);

// Clé de stockage local pour le cache météo
const LOCAL_STORAGE_METEO_KEY = 'agrotom_meteo_cache';

// Mise en cache des données météo avec limitation du nombre d'entrées
export const cacheMeteoData = async (city, data, maxEntries = 20) => {
  try {
    let cache = {};
    const raw = await AsyncStorage.getItem(LOCAL_STORAGE_METEO_KEY);
    if (raw) cache = JSON.parse(raw);
    cache[city] = { ...data, cachedAt: Date.now() };
    const entries = Object.entries(cache);
    if (entries.length > maxEntries) {
      entries.sort((a, b) => (a[1].cachedAt || 0) - (b[1].cachedAt || 0));
      const toKeep = entries.slice(-maxEntries);
      cache = Object.fromEntries(toKeep);
    }
    await AsyncStorage.setItem(LOCAL_STORAGE_METEO_KEY, JSON.stringify(cache));
  } catch (e) {
    // Optionnel : log pour debug
  }
};

// Récupération des données météo en cache pour une ville donnée
export const getCachedMeteoData = async (city) => {
  try {
    const raw = await AsyncStorage.getItem(LOCAL_STORAGE_METEO_KEY);
    if (raw) {
      const cache = JSON.parse(raw);
      if (cache && cache[city]) return { ...cache[city], offline_mode: true };
    }
  } catch (e) {}
  return null;
};

// Récupération des prévisions météo (online/offline)
export const fetchForecast = async (city) => {
  try {
    const res = await api.get('/meteo/forecast/', { params: { city } });
    await cacheMeteoData(city, { forecast: res.data.forecast || [] });
    return { forecast: res.data.forecast || [], offline_mode: false };
  } catch {
    const cached = await getCachedMeteoData(city);
    return { forecast: cached?.forecast || [], offline_mode: true };
  }
};

// Récupération des statistiques météo (online/offline)
export const fetchMeteoStats = async (city) => {
  try {
    const res = await api.get('/meteo/stats/', { params: { city } });
    await cacheMeteoData(city, res.data);
    return { ...res.data, offline_mode: false };
  } catch {
    const cached = await getCachedMeteoData(city);
    return cached ? { ...cached, offline_mode: true } : { offline_mode: true };
  }
};

// Envoi d'un diagnostic (requiert le réseau)
export const uploadDiagnostic = async (formData, userId) =>
  api.post('/diagnostics/upload/', formData, {
    headers: userId ? { 'X-User-Id': userId } : {},
  }).then(res => res.data);

// Récupération de l'historique des diagnostics (requiert le réseau)
// userId = identifiant utilisateur (optionnel)
export const getDiagnosticHistory = async (userId) =>
  api.get('/diagnostics/history', {
    headers: userId ? { 'X-User-Id': userId } : {},
  }).then(res => res.data);

// Export de l'instance Axios pour usage avancé
export default api;
