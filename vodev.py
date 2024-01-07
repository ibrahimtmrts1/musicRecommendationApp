# Gerekli kütüphaneleri yükleyelim
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Örnek bir müzik veri seti oluşturalım
data = {
    'Şarkı': ['Song1', 'Song2', 'Song3', 'Song4', 'Song5', 'Song6', 'Song7'],
    'Tempo': [120, 100, 130, 110, 115, 105, 125],
    'Enerji': [70, 60, 80, 75, 72, 68, 78],
    'Danslık': [0.8, 0.7, 0.9, 0.85, 0.88, 0.75, 0.92],
    'Romantiklik': [0.2, 0.1, 0.5, 0.3, 0.4, 0.15, 0.6],
    'Popülerlik': [0.7, 0.5, 0.8, 0.6, 0.75, 0.9, 0.85]
}

music_data = pd.DataFrame(data)

# Veriyi inceleyelim
print("Veri Seti:")
print(music_data)

# Veriyi normalleştirelim
scaler = StandardScaler()
X_scaled = scaler.fit_transform(music_data[['Tempo', 'Enerji', 'Danslık', 'Romantiklik', 'Popülerlik']])

# K-Means kümeleme modelini oluşturalım
kmeans = KMeans(n_clusters=2)
music_data['Cluster'] = kmeans.fit_predict(X_scaled)

# Kullanıcı tercihlerini ekleyelim
user_preferences = pd.DataFrame({
    'Tempo': [115],
    'Enerji': [70],
    'Danslık': [0.85],
    'Romantiklik': [0.4],
    'Popülerlik': [0.7]
})

# Kullanıcı tercihlerini de normalleştirelim
user_preferences_scaled = scaler.transform(user_preferences)

# Kullanıcının tercihlerine en yakın küme etiketini alalım
user_cluster = kmeans.predict(user_preferences_scaled)[0]

# Kullanıcının tercih ettiği kümedeki şarkıları seçelim
recommended_songs = music_data[music_data['Cluster'] == user_cluster]

# Öneri şarkıları
print("\nÖneri Şarkıları:")
print(recommended_songs[['Şarkı', 'Cluster']])
