import React, { useEffect } from 'react';
import { NavigationContainer } from '@react-navigation/native';
import { createStackNavigator } from '@react-navigation/stack';
import { AuthProvider } from './src/contexts/AuthContext';
import Navigation from './src/navigation';
import { Image, View, Text, StyleSheet, StatusBar, Alert } from 'react-native';
import api from './src/services/api';

// Adaptez les chemins d'import selon votre structure réelle
import LoginScreen from './src/screens/LoginScreen';
import RegisterScreen from './src/screens/RegisterScreen';
import HomeScreen from './src/screens/HomeScreen';
import DiagnosticScreen from './src/screens/DiagnosticScreen';
import HistoryScreen from './src/screens/HistoryScreen';
import ForecastScreen from './src/screens/ForecastScreen';
import AdminDashboardScreen from './src/screens/AdminDashboardScreen';
import AdminCollectMeteoScreen from './src/screens/AdminCollectMeteoScreen';

const Stack = createStackNavigator();

function BrandingHeader() {
  return (
    <View style={styles.brandingContainer}>
      <Image
        source={{ uri: 'https://cdn-icons-png.flaticon.com/512/415/415733.png' }}
        style={styles.logo}
        resizeMode="contain"
      />
      <Text style={styles.title}>AGROTOM</Text>
      <Text style={styles.subtitle}>
        Diagnostic intelligent des maladies de la tomate par CNN+MLP
      </Text>
    </View>
  );
}

export default function App() {
  useEffect(() => {
    // Correction : ne bloquez pas l'affichage sur un loader, affichez l'alerte mais laissez la navigation continuer
    api.get('/api/health/')
      .catch(() => {
        Alert.alert(
          "Erreur de connexion",
          "Impossible de contacter le backend. Vérifiez que le serveur est bien lancé."
        );
      });
  }, []);

  return (
    <AuthProvider>
      <View style={styles.background}>
        <StatusBar barStyle="dark-content" backgroundColor="#f7fafc" />
        <BrandingHeader />
        <View style={styles.content}>
          <NavigationContainer>
            <Stack.Navigator initialRouteName="Login" screenOptions={{ headerShown: false }}>
              <Stack.Screen name="Login" component={LoginScreen} />
              <Stack.Screen name="Register" component={RegisterScreen} />
              <Stack.Screen name="Home" component={HomeScreen} />
              <Stack.Screen name="Diagnostic" component={DiagnosticScreen} />
              <Stack.Screen name="History" component={HistoryScreen} />
              <Stack.Screen name="Forecast" component={ForecastScreen} />
              <Stack.Screen name="AdminDashboard" component={AdminDashboardScreen} />
              {/* Ajoutez les écrans manquants si vous souhaitez utiliser ces routes */}
              {/* 
              <Stack.Screen name="AdminUsers" component={AdminUsersScreen} />
              <Stack.Screen name="DiseaseInfo" component={DiseaseInfoScreen} />
              <Stack.Screen name="Notifications" component={NotificationsScreen} />
              <Stack.Screen name="Settings" component={SettingsScreen} />
              <Stack.Screen name="Help" component={HelpScreen} />
              */}
            </Stack.Navigator>
          </NavigationContainer>
        </View>
      </View>
    </AuthProvider>
  );
}

const styles = StyleSheet.create({
  background: {
    flex: 1,
    backgroundColor: '#f7fafc',
    justifyContent: 'flex-start',
  },
  brandingContainer: {
    alignItems: 'center',
    marginTop: 48,
    marginBottom: 16,
  },
  logo: {
    width: 80,
    height: 80,
    marginBottom: 8,
  },
  title: {
    fontWeight: 'bold',
    fontSize: 28,
    color: '#388e3c',
    letterSpacing: 1,
  },
  subtitle: {
    fontSize: 16,
    color: '#666',
    marginTop: 4,
    textAlign: 'center',
    paddingHorizontal: 16,
  },
  content: {
    flex: 1,
  },
});
