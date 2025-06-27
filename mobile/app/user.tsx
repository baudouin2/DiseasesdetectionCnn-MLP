import React from 'react';
import { View, Text, StyleSheet, Button, ScrollView } from 'react-native';
import { useRouter } from 'expo-router';

export default function UserScreen() {
  const router = useRouter();

  return (
    <ScrollView contentContainerStyle={styles.container}>
      <Text style={styles.title}>Espace Utilisateur</Text>
      <Text style={styles.subtitle}>
        Bienvenue dans votre espace utilisateur AGROTOM.
      </Text>
      <View style={styles.buttonContainer}>
        <Button
          title="Faire un diagnostic"
          color="#256029"
          onPress={() => router.push('/diagnostic')}
        />
      </View>
      
      
      {/* Ajoutez d'autres boutons/fonctionnalités utilisateur si besoin */}
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: {
    flexGrow: 1,
    padding: 24,
    backgroundColor: '#f6f7f2',
    alignItems: 'center',
    justifyContent: 'center',
  },
  title: {
    fontSize: 26,
    fontWeight: 'bold',
    color: '#256029',
    marginBottom: 18,
    textAlign: 'center',
  },
  subtitle: {
    fontSize: 16,
    color: '#223322',
    marginBottom: 32,
    textAlign: 'center',
  },
  buttonContainer: {
    width: '100%',
    marginBottom: 18,
  },
});
