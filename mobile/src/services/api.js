import axios from 'axios';

const api = axios.create({
  baseURL: 'http://localhost:8000', // Adaptez selon votre config réseau/mobile
  timeout: 180000,
});

export default api;
