from flask import Flask, render_template, request, flash
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

#Flask uygulamasını başlatalım
app = Flask(__name__)
app.secret_key = 'super_secret_key'  

#Gerekli müzik veri setini ekleyelim
data = {
    'Şarkı': [
        'Ben Yoruldum Hayat', 'Taş', 'Yıldızlar', 'Bir Derdim Var', 'Gidersen',
        'Aşk', 'Kış Güneşi', 'Rüzgar', 'Duman', 'Gece', 'Uzaklar', 'Yol',
        'Zaman', 'Sevda', 'Hayal', 'Veda', 'Sonsuz', 'Düş', 'Yıldız', 'Hüzün'
    ],
    'Sanatçı': [
        'Mümin Sarıkaya', 'Cem Adrian', 'Mor ve Ötesi', 'mor ve ötesi', 'Jehan Barbur',
        'Teoman', 'Tarkan', 'Duman', 'Sezen Aksu', 'Sıla', 'MFÖ', 'Kenan Doğulu',
        'Ajda Pekkan', 'Manga', 'Duman', 'Tarkan', 'Sezen Aksu', 'Mor ve Ötesi', 'Teoman', 'Levent Yüksel'
    ],
    'Tempo': [
        85, 95, 130, 120, 100, 105, 110, 125, 128, 132, 90, 118,
        115, 123, 108, 113, 109, 107, 111, 96
    ],
    'Enerji': [
        45, 55, 80, 75, 65, 70, 60, 85, 90, 55, 50, 68,
        72, 77, 63, 59, 66, 81, 76, 58
    ],
    'Danslık': [
        0.4, 0.5, 0.9, 0.8, 0.6, 0.7, 0.5, 0.8, 0.7, 0.6, 0.4, 0.7,
        0.8, 0.5, 0.6, 0.4, 0.7, 0.9, 0.8, 0.5
    ],
    'Romantiklik': [
        0.8, 0.3, 0.1, 0.5, 0.7, 0.9, 0.6, 0.2, 0.4, 0.5, 0.7, 0.4,
        0.6, 0.5, 0.3, 0.7, 0.4, 0.2, 0.6, 0.5
    ],
    'Popülerlik': [
        0.75, 0.8, 0.85, 0.9, 0.65, 0.7, 0.8, 0.78, 0.82, 0.79, 0.6, 0.83,
        0.68, 0.86, 0.64, 0.69, 0.73, 0.87, 0.71, 0.66
    ]
}

music_data = pd.DataFrame(data)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(music_data[['Tempo', 'Enerji', 'Danslık', 'Romantiklik', 'Popülerlik']])

#Kümeleme işlemi uygulayalım
default_cluster_count = 5
kmeans = KMeans(n_clusters=default_cluster_count)
music_data['Cluster'] = kmeans.fit_predict(X_scaled)

#Yönlendirme yapalım
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            user_preferences = {
                'Tempo': float(request.form['tempo']),
                'Enerji': float(request.form['enerji']),
                'Danslık': float(request.form['danslık']),
                'Romantiklik': float(request.form['romantiklik']),
                'Popülerlik': float(request.form['popülerlik']),
            }
            cluster_count = int(request.form.get('cluster_count', default_cluster_count))
            user_preferences_scaled = scaler.transform([list(user_preferences.values())])
            kmeans = KMeans(n_clusters=cluster_count)
            music_data['Cluster'] = kmeans.fit_predict(X_scaled)
            user_cluster = kmeans.predict(user_preferences_scaled)[0]

            recommended_songs = music_data[music_data['Cluster'] == user_cluster]

            return render_template('recommendations.html', songs=recommended_songs.to_dict(orient='records'))

        except ValueError as e:
            flash("Lütfen geçerli değerler girin!")
            return redirect(url_for('index'))

    return render_template('index.html')
#Önerilen şarkıları gösterecek sayfaya yönlendirme yapalım
@app.route('/recommendations')
def recommendations():
    return render_template('recommendations.html')

if __name__ == '__main__':
    app.run(debug=True)
