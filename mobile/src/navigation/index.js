import React from 'react';
import { NavigationContainer } from '@react-navigation/native';
import { createStackNavigator } from '@react-navigation/stack';
import LoginScreen from '../screens/LoginScreen';
import RegisterScreen from '../screens/RegisterScreen';
import UploadScreen from '../screens/UploadScreen';
import HistoryScreen from '../screens/HistoryScreen';
import ForecastScreen from '../screens/ForecastScreen';
import AdminDashboardScreen from '../screens/AdminDashboardScreen';
import AdminReviewsScreen from '../screens/AdminReviewsScreen';
import AdminHistoryScreen from '../screens/AdminHistoryScreen';
import AdminForecastScreen from '../screens/AdminForecastScreen';

// TODO: Import context/provider for authentication and user roles if not already done
// import { AuthProvider, useAuth } from '../contexts/AuthContext';

const Stack = createStackNavigator();

export default function Navigation() {
  // TODO: Replace with real authentication/role logic
  // const { user } = useAuth();
  // Example: const isAdmin = user?.is_admin;

  return (
    <NavigationContainer>
      <Stack.Navigator
        initialRouteName="Login"
        screenOptions={{
          headerShown: false,
          cardStyle: { backgroundColor: '#f4f8f3' }, // UI cohérente avec le frontend web
        }}
      >
        {/* Authentification */}
        <Stack.Screen name="Login" component={LoginScreen} />
        <Stack.Screen name="Register" component={RegisterScreen} />

        {/* Utilisateur standard */}
        <Stack.Screen name="Upload" component={UploadScreen} />
        <Stack.Screen name="History" component={HistoryScreen} />
        <Stack.Screen name="Forecast" component={ForecastScreen} />

        {/* Administration */}
        <Stack.Screen name="AdminDashboard" component={AdminDashboardScreen} />
        <Stack.Screen name="AdminReviews" component={AdminReviewsScreen} />
        <Stack.Screen name="AdminHistory" component={AdminHistoryScreen} />
        <Stack.Screen name="AdminForecast" component={AdminForecastScreen} />

        {/* TODO: Ajouter d'autres écrans si besoin pour la gestion des utilisateurs, notifications, etc. */}
      </Stack.Navigator>
    </NavigationContainer>
  );
}
