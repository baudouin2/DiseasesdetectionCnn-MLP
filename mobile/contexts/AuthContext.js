import React, { createContext, useContext, useState, useEffect } from 'react';
import api from '../utils/api';
import * as SecureStore from 'expo-secure-store';

const AuthContext = createContext();

export const AuthProvider = ({ children }) => {
  const [user, setUser] = useState(null);

  // Chargement sécurisé de l'utilisateur au démarrage
  useEffect(() => {
    const loadUser = async () => {
      try {
        const userStr = await SecureStore.getItemAsync('user');
        if (userStr) setUser(JSON.parse(userStr));
      } catch (e) {
        setUser(null);
      }
    };
    loadUser();
  }, []);

  // Gestion stricte des headers custom pour chaque utilisateur (même logique que frontend)
  useEffect(() => {
    if (user && user.id && user.id !== 'None') {
      api.defaults.headers.common['X-User-Id'] = user.id;
      api.defaults.headers.common['X-User-Username'] = user.username || '';
      api.defaults.headers.common['X-User-Email'] = user.email || '';
      api.defaults.headers.common['X-User-Is-Admin'] = user.is_admin ? 'true' : 'false';
      api.defaults.headers.common['X-User-Is-Farmer'] = user.is_farmer ? 'true' : 'false';
    } else {
      delete api.defaults.headers.common['X-User-Id'];
      delete api.defaults.headers.common['X-User-Username'];
      delete api.defaults.headers.common['X-User-Email'];
      delete api.defaults.headers.common['X-User-Is-Admin'];
      delete api.defaults.headers.common['X-User-Is-Farmer'];
    }
  }, [user]);

  // Ajout: Intercepteur axios pour injecter automatiquement l'utilisateur dans chaque requête API
  useEffect(() => {
    // Ejecter l'ancien intercepteur si déjà présent
    let interceptorId;
    if (user && user.username) {
      interceptorId = api.interceptors.request.use((config) => {
        // Supprime tout header X-User-Id potentiellement présent
        delete config.headers['X-User-Id'];
        // Utilise uniquement le username en query param pour les requêtes GET
        if (config.method === 'get' && user.username) {
          if (!config.params) config.params = {};
          config.params.username = user.username;
        }
        // Pour les autres requêtes (POST, etc.), n'ajoute pas X-User-Id
        return config;
      });
    }
    return () => {
      if (interceptorId !== undefined) {
        api.interceptors.request.eject(interceptorId);
      }
    };
  }, [user]);

  // Connexion stricte avec gestion sécurisée du token et de l'utilisateur
  const login = async (credentials) => {
    try {
      const { data } = await api.post('/auth/login/', credentials);
      if (data && data.token && data.user) {
        setUser(data.user);
        await SecureStore.setItemAsync('token', data.token);
        await SecureStore.setItemAsync('user', JSON.stringify(data.user));
      } else {
        throw new Error('Échec de connexion');
      }
    } catch (error) {
      throw error;
    }
  };

  // Déconnexion stricte et nettoyage du stockage sécurisé
  const logout = async () => {
    setUser(null);
    await SecureStore.deleteItemAsync('token');
    await SecureStore.deleteItemAsync('user');
  };

  // Rafraîchissement professionnel de l'utilisateur depuis le backend
  const refreshUser = async () => {
    try {
      const { data } = await api.get('/auth/me/');
      if (data && data.id) {
        setUser(data);
        await SecureStore.setItemAsync('user', JSON.stringify(data));
      }
    } catch (e) {
      // Optionnel : gestion d'erreur ou notification utilisateur
    }
  };

  // Diagnostic upload (équivalent web)
  const uploadDiagnostic = async (formData) => {
    try {
      const response = await api.post('/diagnostics/upload/', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
        timeout: 180000, // 3 minutes
      });
      return response.data;
    } catch (error) {
      throw error;
    }
  };

  // Historique des diagnostics (sans headers custom, même logique que frontend)
  const getDiagnosticsHistory = async (username) => {
    try {
      // On passe les headers à vide pour ne pas envoyer les headers custom
      const response = await api.get(`/diagnostics/history/?username=${encodeURIComponent(username)}`, {
        headers: {},
      });
      return response.data;
    } catch (error) {
      throw error;
    }
  };

  // Prévisions météo
  const getWeatherForecast = async () => {
    try {
      const response = await api.get('/meteo/forecast/');
      return response.data;
    } catch (error) {
      throw error;
    }
  };

  // Dashboard admin (exemple)
  const getAdminDashboard = async () => {
    try {
      const response = await api.get('/admin/dashboard/');
      return response.data;
    } catch (error) {
      throw error;
    }
  };

  // Health check (pour loader global)
  const checkApiHealth = async () => {
    try {
      const response = await api.get('/api/health/');
      return response.data;
    } catch (error) {
      throw error;
    }
  };

  return (
    <AuthContext.Provider
      value={{
        user,
        login,
        logout,
        refreshUser,
        uploadDiagnostic,
        getDiagnosticsHistory,
        getWeatherForecast,
        getAdminDashboard,
        checkApiHealth,
      }}
    >
      {children}
    </AuthContext.Provider>
  );
};

export const useAuth = () => useContext(AuthContext);
