import React, { useState, useEffect } from 'react';
import {
  View,
  Text,
  StyleSheet,
  Image,
  TouchableOpacity,
  ActivityIndicator,
  TextInput,
  ScrollView,
  Alert,
  KeyboardAvoidingView,
  Platform,
  TouchableWithoutFeedback,
  Keyboard,
  Modal,
  Pressable,
  FlatList,
  SafeAreaView,
  useColorScheme,
} from 'react-native';
import { Picker } from '@react-native-picker/picker';
import * as ImagePicker from 'expo-image-picker';
import { useRouter } from 'expo-router';
import axios from 'axios';
import { Ionicons, MaterialCommunityIcons } from '@expo/vector-icons';
import AsyncStorage from '@react-native-async-storage/async-storage';

const api = axios.create({
  baseURL: 'http://172.20.10.2:8000/',
  timeout: 180000,
});

const SOL_TYPES = [
  'Argileux', 'Sableux', 'Sablonneux', 'Ferrallitique', 'Ferrugineux',
  'Limon-argileux', 'Sablo-argileux', 'Alluvial', 'Latéritique'
];
const IRRIGATION_TYPES = ['aspersion', 'ruissellement', 'manuel', 'goutte_a_goutte'];

type AgroKeys = 'sol' | 'irrigation' | 'freq_traitement' | 'densite_plantation' | 'azote' | 'phosphore' | 'potassium' | 'compost' | 'engrais_chimique' | 'localite';
type AgroState = Record<AgroKeys, string>;

export default function DiagnosticScreen() {
  const router = useRouter();
  const [image, setImage] = useState<any>(null);
  const [imagePreview, setImagePreview] = useState<string | null>(null);
  const [result, setResult] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [agro, setAgro] = useState<AgroState>({
    sol: '',
    irrigation: '',
    freq_traitement: '',
    densite_plantation: '',
    azote: '',
    phosphore: '',
    potassium: '',
    compost: '',
    engrais_chimique: '',
    localite: '',
  });
  const [localites, setLocalites] = useState<string[]>([]);
  const [meteo, setMeteo] = useState<any>(null);
  const [localiteModalVisible, setLocaliteModalVisible] = useState(false);
  const [meteoModalVisible, setMeteoModalVisible] = useState(false);
  const [solModalVisible, setSolModalVisible] = useState(false);
  const [irrigationModalVisible, setIrrigationModalVisible] = useState(false);
  const colorScheme = useColorScheme();
  const [theme, setTheme] = useState<'light' | 'dark'>(colorScheme === 'dark' ? 'dark' : 'light');

  useEffect(() => {
    api.get('/api/meteo/localites/')
      .then(res => {
        let locs: string[] = [];
        if (Array.isArray(res.data)) {
          if (typeof res.data[0] === 'object' && res.data[0]?.nom) {
            locs = res.data.map((l: any) => l.nom);
          } else if (typeof res.data[0] === 'string') {
            locs = res.data;
          }
        }
        setLocalites(locs.sort((a, b) => a.localeCompare(b, 'fr')));
      })
      .catch(() => setLocalites([]));
  }, []);

  useEffect(() => {
    if (agro.localite) {
      api.get(`/api/meteo/stats/?city=${encodeURIComponent(agro.localite)}`)
        .then(res => setMeteo(res.data))
        .catch((err) => setMeteo({ error: err?.response?.data?.error || "Erreur serveur" }));
    }
  }, [agro.localite]);

  useEffect(() => {
    return () => {
      setImage(null);
      setImagePreview(null);
      setResult(null);
    };
  }, []);

  const pickImage = async () => {
    let res = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ['images'],
      allowsEditing: true,
      quality: 0.7,
      base64: true,
    });
    if (!res.canceled) {
      setImage(res);
      setImagePreview(res.assets?.[0]?.uri);
    }
  };

  const handleUpload = async () => {
    if (!image) return;
    setLoading(true);
    setResult(null);
    try {
      const username = await AsyncStorage.getItem('username');
      const formData = new FormData();
      formData.append('image', {
        uri: image.assets?.[0]?.uri || image.uri,
        name: 'photo.jpg',
        type: 'image/jpeg'
      } as any);
      Object.entries(agro).forEach(([key, value]) => formData.append(key, value !== undefined ? value : ''));
      if (username) {
        formData.append('username', username);
      }
      const { data } = await api.post('/api/diagnostics/upload/', formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });
      setResult(data);
    } catch (e) {
      Alert.alert('Erreur', "Erreur lors de l'analyse");
    } finally {
      setLoading(false);
    }
  };

  const handleLogout = () => {
    Alert.alert('Déconnexion', 'Vous avez été déconnecté.');
    router.replace('/login');
  };

  const handleGoBack = () => {
    if (router.canGoBack?.()) router.back();
    else router.push('/diagnostic');
  };

  const handleToggleTheme = () => {
    setTheme(t => (t === 'light' ? 'dark' : 'light'));
  };

  const themedStyles = theme === 'dark'
    ? {
        backgroundColor: '#181a20',
        color: '#f5f5f5',
        cardBg: '#23262f',
        inputBg: '#23262f',
        borderColor: '#444',
        navBg: '#23262f',
        navBtnBg: '#23262f',
        navBtnText: '#f5f5f5',
        navBtnActiveBg: '#388e3c',
        navBtnActiveText: '#fff',
        buttonBg: '#388e3c',
        buttonText: '#fff',
        label: '#b5b5b5',
        resultText: '#b5e48c',
      }
    : {
        backgroundColor: '#f6f7f2',
        color: '#256029',
        cardBg: '#fff',
        inputBg: '#f8faf8',
        borderColor: '#ccc',
        navBg: '#e8f5e9',
        navBtnBg: '#fff',
        navBtnText: '#497174',
        navBtnActiveBg: '#497174',
        navBtnActiveText: '#fff',
        buttonBg: '#388e3c',
        buttonText: '#fff',
        label: '#256029',
        resultText: '#388e3c',
      };

  return (
    <SafeAreaView style={{ flex: 1, backgroundColor: themedStyles.backgroundColor }}>
      {/* Navigation Bar */}
      <View style={[styles.navBar, { backgroundColor: themedStyles.navBg, borderBottomColor: themedStyles.borderColor }]}>
        <ScrollView horizontal showsHorizontalScrollIndicator={false} contentContainerStyle={styles.navBarContent}>
          <NavButton icon="arrow-left" label="Retour" onPress={handleGoBack} theme={theme} themedStyles={themedStyles} />
          <NavButton icon="theme-light-dark" label="Thème" onPress={handleToggleTheme} theme={theme} themedStyles={themedStyles} />
          <NavButton icon="logout" label="Déconnexion" onPress={handleLogout} theme={theme} themedStyles={themedStyles} />
          <NavButton icon="leaf" label="Diagnostic" onPress={() => router.push('/diagnostic')} active theme={theme} themedStyles={themedStyles} />
        </ScrollView>
      </View>
      <KeyboardAvoidingView
        style={{ flex: 1 }}
        behavior={Platform.OS === 'ios' ? 'padding' : undefined}
        keyboardVerticalOffset={80}
      >
        <TouchableWithoutFeedback onPress={Keyboard.dismiss} accessible={false}>
          <ScrollView
            contentContainerStyle={[styles.container, { backgroundColor: themedStyles.backgroundColor }]}
            keyboardShouldPersistTaps="handled"
          >
            <Text style={[styles.title, { color: themedStyles.color }]}>Diagnostic feuille de tomate</Text>
            {/* 1. Choix de la localité */}
            <Card themedStyles={themedStyles}>
              <Text style={[styles.label, { color: themedStyles.label }]}>Localité</Text>
              <Pressable
                style={[styles.input, { justifyContent: 'center', minHeight: 44 }]}
                onPress={() => setLocaliteModalVisible(true)}
              >
                <Text style={{ color: agro.localite ? '#222' : '#888' }}>
                  {agro.localite || 'Choisir votre localité'}
                </Text>
                <Ionicons name="chevron-down" size={18} color="#888" style={{ position: 'absolute', right: 10, top: 12 }} />
              </Pressable>
              <Modal
                visible={localiteModalVisible}
                animationType="slide"
                transparent={true}
                onRequestClose={() => setLocaliteModalVisible(false)}
              >
                <View style={styles.modalOverlay}>
                  <View style={styles.modalContent}>
                    <Text style={styles.modalTitle}>Sélectionner une localité</Text>
                    <FlatList
                      data={localites}
                      keyExtractor={item => item}
                      renderItem={({ item }) => (
                        <Pressable
                          style={styles.modalItem}
                          onPress={() => {
                            setAgro({ ...agro, localite: item });
                            setLocaliteModalVisible(false);
                          }}
                        >
                          <Text style={styles.modalItemText}>{item}</Text>
                        </Pressable>
                      )}
                      initialNumToRender={20}
                      maxToRenderPerBatch={30}
                      keyboardShouldPersistTaps="handled"
                    />
                    <Pressable style={styles.modalCloseBtn} onPress={() => setLocaliteModalVisible(false)}>
                      <Text style={styles.modalCloseText}>Annuler</Text>
                    </Pressable>
                  </View>
                </View>
              </Modal>
            </Card>
            {/* Affichage du bouton pour voir la météo après sélection de la localité */}
            {agro.localite ? (
              <TouchableOpacity
                style={styles.meteoBtn}
                onPress={() => setMeteoModalVisible(true)}
              >
                <MaterialCommunityIcons name="weather-partly-cloudy" size={20} color="#fff" />
                <Text style={styles.meteoBtnText}>Voir la météo de ces derniers mois</Text>
              </TouchableOpacity>
            ) : null}
            {/* Modal météo */}
            <Modal
              visible={meteoModalVisible}
              animationType="slide"
              transparent={true}
              onRequestClose={() => setMeteoModalVisible(false)}
            >
              <View style={styles.modalOverlay}>
                <View style={styles.modalContent}>
                  <Text style={styles.modalTitle}>Météo 3 derniers mois ({agro.localite})</Text>
                  {meteo && (
                    meteo.error ? (
                      <Text style={styles.meteoText}>
                        {meteo.error}
                        {"\n"}{JSON.stringify(meteo, null, 2)}
                      </Text>
                    ) : Array.isArray(meteo.months) && meteo.months.length > 0 && meteo.stats && Object.values(meteo.stats).some(val => typeof val === 'object' && val !== null && 'avg' in val && (val as any).avg !== null) ? (
                      <ScrollView style={{ maxHeight: 300, width: '100%' }}>
                        <Text style={styles.meteoText}>
                          Température moyenne : {meteo.stats.temperature?.avg !== null ? `${meteo.stats.temperature.avg.toFixed(2)} °C` : 'N/A'}
                        </Text>
                        <Text style={styles.meteoText}>
                          Humidité moyenne : {meteo.stats.humidity?.avg !== null ? `${meteo.stats.humidity.avg.toFixed(2)} %` : 'N/A'}
                        </Text>
                        <Text style={styles.meteoText}>
                          Pression moyenne : {meteo.stats.pressure?.avg !== null ? `${meteo.stats.pressure.avg.toFixed(2)} hPa` : 'N/A'}
                        </Text>
                        <Text style={styles.meteoText}>
                          Vent moyen : {meteo.stats.wind_speed?.avg !== null ? `${meteo.stats.wind_speed.avg.toFixed(2)} m/s` : 'N/A'}
                        </Text>
                        <Text style={styles.meteoText}>
                          Précipitations moyennes : {meteo.stats.precipitation?.avg !== null ? `${meteo.stats.precipitation.avg.toFixed(2)} mm` : 'N/A'}
                        </Text>
                        <Text style={[styles.meteoText, { marginTop: 10, fontWeight: 'bold' }]}>Détail par mois :</Text>
                        {Array.isArray(meteo.months) && meteo.months.length > 0 ? (
                          meteo.months.map((m: any) => (
                            <View key={m.month} style={{ marginBottom: 6 }}>
                              <Text style={[styles.meteoText, { fontWeight: 'bold' }]}>{m.month}</Text>
                              <Text style={styles.meteoText}>
                                Température : {m.temperature_avg !== null ? `${m.temperature_avg} °C` : 'N/A'}
                              </Text>
                              <Text style={styles.meteoText}>
                                Humidité : {m.humidity_avg !== null ? `${m.humidity_avg} %` : 'N/A'}
                              </Text>
                              <Text style={styles.meteoText}>
                                Pression : {m.pressure_avg !== null ? `${m.pressure_avg} hPa` : 'N/A'}
                              </Text>
                              <Text style={styles.meteoText}>
                                Vent : {m.wind_speed_avg !== null ? `${m.wind_speed_avg} m/s` : 'N/A'}
                              </Text>
                              <Text style={styles.meteoText}>
                                Précipitations : {m.precipitation_avg !== null ? `${m.precipitation_avg} mm` : 'N/A'}
                              </Text>
                            </View>
                          ))
                        ) : (
                          <Text style={styles.meteoText}>Aucun détail mensuel.</Text>
                        )}
                      </ScrollView>
                    ) : (
                      <Text style={styles.meteoText}>
                        {meteo && meteo.stats && Object.values(meteo.stats).some(val => val && typeof val === 'object' && 'avg' in val && typeof (val as any).avg === 'number')
                          ? [
                              meteo.stats.temperature?.avg !== undefined
                                ? `Température moyenne : ${meteo.stats.temperature.avg !== null ? `${meteo.stats.temperature.avg.toFixed(2)} °C` : 'N/A'}\n`
                                : '',
                              meteo.stats.humidity?.avg !== undefined
                                ? `Humidité moyenne : ${meteo.stats.humidity.avg !== null ? `${meteo.stats.humidity.avg.toFixed(2)} %` : 'N/A'}\n`
                                : '',
                              meteo.stats.pressure?.avg !== undefined
                                ? `Pression moyenne : ${meteo.stats.pressure.avg !== null ? `${meteo.stats.pressure.avg.toFixed(2)} hPa` : 'N/A'}\n`
                                : '',
                              meteo.stats.wind_speed?.avg !== undefined
                                ? `Vent moyen : ${meteo.stats.wind_speed.avg !== null ? `${meteo.stats.wind_speed.avg.toFixed(2)} m/s` : 'N/A'}\n`
                                : '',
                              meteo.stats.precipitation?.avg !== undefined
                                ? `Précipitations moyennes : ${meteo.stats.precipitation.avg !== null ? `${meteo.stats.precipitation.avg.toFixed(2)} mm` : 'N/A'}\n`
                                : '',
                            ].join('')
                          : (meteo && meteo.error
                            ? meteo.error
                            : "Aucune donnée météo disponible.")
                        }
                      </Text>
                    )
                  )}
                  <Pressable style={styles.modalCloseBtn} onPress={() => setMeteoModalVisible(false)}>
                    <Text style={styles.modalCloseText}>Fermer</Text>
                  </Pressable>
                </View>
              </View>
            </Modal>
            {/* 2. Type de sol */}
            <Card themedStyles={themedStyles}>
              <Text style={[styles.label, { color: themedStyles.label }]}>Type de sol</Text>
              <Pressable
                style={[styles.input, { justifyContent: 'center', minHeight: 44 }]}
                onPress={() => setSolModalVisible(true)}
              >
                <Text style={{ color: agro.sol ? '#222' : '#888' }}>
                  {agro.sol || 'Sur quel type de sol sont les cultures ?'}
                </Text>
                <Ionicons name="chevron-down" size={18} color="#888" style={{ position: 'absolute', right: 10, top: 12 }} />
              </Pressable>
              <Modal
                visible={solModalVisible}
                animationType="slide"
                transparent={true}
                onRequestClose={() => setSolModalVisible(false)}
              >
                <View style={styles.modalOverlay}>
                  <View style={styles.modalContent}>
                    <Text style={styles.modalTitle}>Sélectionner le type de sol</Text>
                    <FlatList
                      data={SOL_TYPES}
                      keyExtractor={item => item}
                      renderItem={({ item }) => (
                        <Pressable
                          style={styles.modalItem}
                          onPress={() => {
                            setAgro({ ...agro, sol: item });
                            setSolModalVisible(false);
                          }}
                        >
                          <Text style={styles.modalItemText}>{item}</Text>
                        </Pressable>
                      )}
                      initialNumToRender={10}
                      maxToRenderPerBatch={20}
                      keyboardShouldPersistTaps="handled"
                    />
                    <Pressable style={styles.modalCloseBtn} onPress={() => setSolModalVisible(false)}>
                      <Text style={styles.modalCloseText}>Annuler</Text>
                    </Pressable>
                  </View>
                </View>
              </Modal>
            </Card>
            {/* 3. Type d'irrigation */}
            <Card themedStyles={themedStyles}>
              <Text style={[styles.label, { color: themedStyles.label }]}>Type d'irrigation</Text>
              <Pressable
                style={[styles.input, { justifyContent: 'center', minHeight: 44 }]}
                onPress={() => setIrrigationModalVisible(true)}
              >
                <Text style={{ color: agro.irrigation ? '#222' : '#888' }}>
                  {agro.irrigation || "le type d'irrigation effectuee"}
                </Text>
                <Ionicons name="chevron-down" size={18} color="#888" style={{ position: 'absolute', right: 10, top: 12 }} />
              </Pressable>
              <Modal
                visible={irrigationModalVisible}
                animationType="slide"
                transparent={true}
                onRequestClose={() => setIrrigationModalVisible(false)}
              >
                <View style={styles.modalOverlay}>
                  <View style={styles.modalContent}>
                    <Text style={styles.modalTitle}>Sélectionner le type d'irrigation</Text>
                    <FlatList
                      data={IRRIGATION_TYPES}
                      keyExtractor={item => item}
                      renderItem={({ item }) => (
                        <Pressable
                          style={styles.modalItem}
                          onPress={() => {
                            setAgro({ ...agro, irrigation: item });
                            setIrrigationModalVisible(false);
                          }}
                        >
                          <Text style={styles.modalItemText}>{item}</Text>
                        </Pressable>
                      )}
                      initialNumToRender={10}
                      maxToRenderPerBatch={20}
                      keyboardShouldPersistTaps="handled"
                    />
                    <Pressable style={styles.modalCloseBtn} onPress={() => setIrrigationModalVisible(false)}>
                      <Text style={styles.modalCloseText}>Annuler</Text>
                    </Pressable>
                  </View>
                </View>
              </Modal>
            </Card>
            {/* 4. Image */}
            <Card themedStyles={themedStyles}>
              <Text style={styles.label}>Image de la feuille</Text>
              <TouchableOpacity style={styles.button} onPress={pickImage} disabled={loading}>
                <Ionicons name="image-outline" size={20} color="#fff" style={{ marginRight: 6 }} />
                <Text style={styles.buttonText}>Choisir une image</Text>
              </TouchableOpacity>
              {imagePreview && !result && (
                <Image source={{ uri: imagePreview }} style={styles.image} />
              )}
            </Card>
            {/* 5. Pratiques agricoles */}
            <Card themedStyles={themedStyles}>
              <Text style={[styles.label, { color: themedStyles.label }]}>Fréquence de traitement (par mois)</Text>
              <TextInput
                style={[
                  styles.input,
                  {
                    backgroundColor: themedStyles.inputBg,
                    borderColor: themedStyles.borderColor,
                    color: themedStyles.color,
                  },
                ]}
                value={agro.freq_traitement}
                onChangeText={v => setAgro({ ...agro, freq_traitement: v })}
                keyboardType="numeric"
                placeholder="Ex: 4"
              />
              <Text style={[styles.label, { color: themedStyles.label }]}>Densité de plantation (plants/m²)</Text>
              <TextInput
                style={[
                  styles.input,
                  {
                    backgroundColor: themedStyles.inputBg,
                    borderColor: themedStyles.borderColor,
                    color: themedStyles.color,
                  },
                ]}
                value={agro.densite_plantation}
                onChangeText={v => setAgro({ ...agro, densite_plantation: v })}
                keyboardType="numeric"
                placeholder="Ex: 1.98"
              />
              {(['azote', 'phosphore', 'potassium', 'compost', 'engrais_chimique'] as AgroKeys[]).map(key => (
                <View key={key} style={styles.formGroup}>
                  <Text style={[styles.label, { color: themedStyles.label }]}>{key.charAt(0).toUpperCase() + key.slice(1).replace('_', ' ')} (kg/m²)</Text>
                  <TextInput
                    style={[
                      styles.input,
                      {
                        backgroundColor: themedStyles.inputBg,
                        borderColor: themedStyles.borderColor,
                        color: themedStyles.color,
                      },
                    ]}
                    value={agro[key]}
                    onChangeText={v => setAgro({ ...agro, [key]: v })}
                    keyboardType="numeric"
                  />
                </View>
              ))}
            </Card>
            {/* 6. Lancer le diagnostic */}
            <TouchableOpacity
              style={[
                styles.button,
                { backgroundColor: themedStyles.buttonBg },
                (!image || loading) && { opacity: 0.7 },
              ]}
              onPress={handleUpload}
              disabled={!image || loading}
            >
              <MaterialCommunityIcons name="magnify-scan" size={20} color={themedStyles.buttonText} style={{ marginRight: 6 }} />
              <Text style={[styles.buttonText, { color: themedStyles.buttonText }]}>Analyser</Text>
            </TouchableOpacity>
            {loading && <ActivityIndicator size="large" color="#388e3c" style={{ marginTop: 12 }} />}
            {result && (
              <Card themedStyles={themedStyles}>
                <Text style={[styles.resultTitle, { color: themedStyles.resultText }]}>Résultat du diagnostic</Text>
                <Text style={[styles.resultText, { color: themedStyles.resultText }]}>
                  Maladies détectées : {Array.isArray(result.maladies_detectees) && result.maladies_detectees.length > 0
                    ? result.maladies_detectees.join(', ')
                    : 'Aucune maladie détectée'}
                </Text>
                <Text style={styles.resultText}>
                  Niveau d'affection : {result.predicted_state || '--'}
                </Text>
                <Text style={styles.resultText}>
                  Confiance : {result.confidence ? Math.round(result.confidence * 100) : '--'}%
                </Text>
          
                {result.annotated_image_b64 && (
                  <Image
                    source={{ uri: `data:image/jpeg;base64,${result.annotated_image_b64}` }}
                    style={styles.image}
                    resizeMode="contain"
                  />
                )}
              </Card>
            )}
          </ScrollView>
        </TouchableWithoutFeedback>
      </KeyboardAvoidingView>
    </SafeAreaView>
  );
}

// --- UI Components ---
function Card({ children, themedStyles }: { children: React.ReactNode, themedStyles: any }) {
  return (
    <View style={[styles.card, { backgroundColor: themedStyles.cardBg, borderColor: themedStyles.borderColor }]}>
      {children}
    </View>
  );
}

function NavButton({ icon, label, onPress, active, theme, themedStyles }: { icon: string, label: string, onPress: () => void, active?: boolean, theme: string, themedStyles: any }) {
  let IconComp = icon === "weather-partly-cloudy" || icon === "account-cog" || icon === "history" || icon === "magnify-scan" || icon === "theme-light-dark" || icon === "logout"
    ? MaterialCommunityIcons
    : Ionicons;
  return (
    <TouchableOpacity
      style={[
        styles.navBtn,
        { backgroundColor: active ? themedStyles.navBtnActiveBg : themedStyles.navBtnBg, borderColor: themedStyles.borderColor },
        active && styles.navBtnActive,
      ]}
      onPress={onPress}
    >
      <IconComp name={icon as any} size={22} color={active ? themedStyles.navBtnActiveText : themedStyles.navBtnText} />
      <Text style={[styles.navBtnText, { color: active ? themedStyles.navBtnActiveText : themedStyles.navBtnText }]}>{label}</Text>
    </TouchableOpacity>
  );
}

const styles = StyleSheet.create({
  container: {
    flexGrow: 1,
    padding: 18,
    backgroundColor: '#f6f7f2',
    alignItems: 'center',
    justifyContent: 'flex-start',
    paddingBottom: 40,
  },
  navBar: {
    backgroundColor: '#e8f5e9',
    paddingVertical: 8,
    borderBottomWidth: 1,
    borderBottomColor: '#d0e0d0',
    elevation: 2,
  },
  navBarContent: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: 8,
  },
  navBtn: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#fff',
    marginHorizontal: 6,
    paddingVertical: 8,
    paddingHorizontal: 16,
    borderRadius: 22,
    borderWidth: 1,
    borderColor: '#497174',
  },
  navBtnActive: {
    backgroundColor: '#497174',
    borderColor: '#497174',
  },
  navBtnText: {
    marginLeft: 7,
    fontWeight: 'bold',
    color: '#497174',
    fontSize: 15,
  },
  card: {
    width: '100%',
    backgroundColor: '#fff',
    borderRadius: 14,
    padding: 16,
    marginBottom: 14,
    shadowColor: '#497174',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.08,
    shadowRadius: 6,
    elevation: 2,
  },
  title: {
    fontSize: 26,
    fontWeight: 'bold',
    color: '#256029',
    marginBottom: 18,
    textAlign: 'center',
    marginTop: 12,
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
    flexDirection: 'row',
    justifyContent: 'center',
  },
  buttonText: {
    color: '#fff',
    fontWeight: 'bold',
    fontSize: 17,
    letterSpacing: 0.5,
  },
  image: {
    width: 220,
    height: 220,
    alignSelf: 'center',
    marginVertical: 16,
    borderRadius: 18,
    borderWidth: 2,
    borderColor: '#e8f5e9',
  },
  formGroup: {
    width: '100%',
    marginBottom: 10,
  },
  label: {
    fontWeight: 'bold',
    color: '#256029',
    marginBottom: 2,
    marginTop: 8,
  },
  input: {
    width: '100%',
    backgroundColor: '#f8faf8',
    borderRadius: 8,
    padding: 8,
    borderWidth: 1,
    borderColor: '#ccc',
    fontSize: 15,
    marginBottom: 2,
  },
  meteoBtn: {
    backgroundColor: '#497174',
    borderRadius: 8,
    paddingVertical: 10,
    paddingHorizontal: 18,
    alignItems: 'center',
    marginBottom: 10,
    marginTop: 4,
    width: '100%',
    flexDirection: 'row',
    justifyContent: 'center',
  },
  meteoBtnText: {
    color: '#fff',
    fontWeight: 'bold',
    fontSize: 15,
    marginLeft: 8,
  },
  meteoText: {
    fontSize: 13,
    color: '#497174',
    fontFamily: Platform.OS === 'ios' ? 'Courier' : 'monospace',
  },
  resultTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#388e3c',
    marginBottom: 8,
    textAlign: 'center',
  },
  resultText: {
    fontSize: 16,
    color: '#388e3c',
    fontWeight: '600',
    marginBottom: 2,
  },
  modalOverlay: {
    flex: 1,
    backgroundColor: 'rgba(0,0,0,0.35)',
    justifyContent: 'center',
    alignItems: 'center',
  },
  modalContent: {
    backgroundColor: '#fff',
    borderRadius: 14,
    padding: 18,
    width: '90%',
    maxHeight: '80%',
    alignItems: 'center',
  },
  modalTitle: {
    fontWeight: 'bold',
    fontSize: 20,
    marginBottom: 12,
    color: '#256029',
    textAlign: 'center',
  },
  modalItem: {
    paddingVertical: 12,
    paddingHorizontal: 8,
    borderBottomWidth: 1,
    borderBottomColor: '#eee',
    width: '100%',
  },
  modalItemText: {
    fontSize: 16,
    color: '#256029',
    textAlign: 'left',
  },
  modalCloseBtn: {
    marginTop: 12,
    padding: 10,
    backgroundColor: '#eee',
    borderRadius: 8,
    alignSelf: 'center',
  },
  modalCloseText: {
    color: '#c0392b',
    fontWeight: 'bold',
    fontSize: 16,
  },
});
