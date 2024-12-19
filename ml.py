from flask import Flask, request, render_template_string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
import joblib
import pandas as pd
import os
import matplotlib
matplotlib.use('Agg')  # Backend non-interaktif
import matplotlib.pyplot as plt

# Direktori dataset dan model
DATASET_PATH = r"D:\Dashboard\myenv\proyek_fismod\cities_air_quality_water_pollution.18-10-2021 (1).csv"
MODEL_TEXT_CLASSIFIER_PATH = r"D:\Dashboard\myenv\proyek_fismod\model_text_classifier.pkl"
MODEL_AIR_PATH = r"D:\Dashboard\myenv\proyek_fismod\model_prediksi_kualitas_udara.pkl"
MODEL_WATER_PATH = r"D:\Dashboard\myenv\proyek_fismod\model_prediksi_kualitas_air.pkl"
SCALER_PATH = r"D:\Dashboard\myenv\proyek_fismod\skala_prediksi_kualitas_air.pkl"


# Muat data
data = pd.read_csv(DATASET_PATH)
data.columns = data.columns.str.strip().str.replace('"', '')

def label_daerah(row):
    if row['AirQuality'] > 50 and row['WaterPollution'] < 60:
        return 'Layak'
    else:
        return 'Tidak Layak'

data['Label'] = data.apply(label_daerah, axis=1)
data['Deskripsi'] = data['City'] + ' memiliki kualitas udara ' + data['AirQuality'].astype(str) + ' dan polusi air ' + data['WaterPollution'].astype(str)
X = data['Deskripsi']
y = data['Label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Membuat pipeline: TF-IDF + Random Forest Classifier
model = make_pipeline(TfidfVectorizer(), RandomForestClassifier())
model.fit(X_train, y_train)

# Menyimpan model
joblib.dump(model, MODEL_TEXT_CLASSIFIER_PATH)
# Fungsi membuat grafik distribusi
def buat_grafik_distribusi():
    plt.figure(figsize=(8, 5))
    plt.hist(data['WaterPollution'], bins=20, alpha=0.7, label='Polusi Air', color='tab:blue')
    plt.hist(data['AirQuality'], bins=20, alpha=0.7, label='Kualitas Udara', color='tab:orange')
    plt.xlabel('Nilai Indeks')
    plt.ylabel('Frekuensi')
    plt.title('Distribusi Kualitas Udara dan Polusi Air')
    plt.legend()
    plt.savefig('static/distribusi_kualitas_air_udara.png')
    plt.close()

# Buat grafik jika belum ada
if not os.path.exists('static'):
    os.makedirs('static')
buat_grafik_distribusi()

# Flask
app = Flask(__name__)

@app.route('/')
def beranda():
    rata_kualitas_udara = round(data['AirQuality'].mean(), 2)
    rata_polusi_air = round(data['WaterPollution'].mean(), 2)

    return render_template_string('''
    <!DOCTYPE html>
    <html lang="id">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <title>Prediksi Keberlanjutan Sumber Air dan Udara</title>
        <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/@picocss/pico@1/css/pico.min.css">
        <style>
            body {
                background-color: #f8f9fa;
                color: #2c3e50;
                font-family: Arial, sans-serif;
            }
            h1, h2, h3 {
                color: #34495e;
            }
            nav strong {
                color: #2c3e50;
                font-weight: bold;
            }
            .container {
                padding: 20px;
            }
            img {
                width: 100%;
                max-width: 900px;
                border-radius: 8px;
                margin: 10px 0;
            }
            .result-card {
                background-color: #ffffff;
                padding: 20px;
                border-radius: 8px;
                box-shadow: 0px 4px 12px rgba(0, 0, 0, 0.1);
                margin-bottom: 20px;
            }
            .result-card h3 {
                color: #3498db;
            }
            .result-card ul {
                list-style-type: none;
                padding: 0;
            }
            .result-card ul li {
                font-size: 1.2em;
            }
            footer small a {
                color: #3498db;
                text-decoration: none;
            }
            html {
                scroll-behavior: smooth;
            }
        </style>
    </head>
    <body>
        <!-- Navigasi -->
        <nav class="container-fluid">
            <ul>
                <li style="font-size: 36px; font-family: Arial, sans-serif; 
                            background-color: #f0f0f0; 
                            padding: 20px; 
                            border-radius: 15px;
                            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);">
                    <strong>U-AIR</strong>
                </li>
            </ul>
            <ul>
                <li><a href="#informasi">Informasi</a></li>
                <li><a href="#grafik">Grafik</a></li>
                <li><a href="#masukkan" role="button">Masukkan Data</a></li>
            </ul>
        </nav>

        <!-- Main Content -->
        <main class="container">
            <section class="result-card">
                <hgroup>
                    <h2>Prediksi Keberlanjutan Air & Udara</h2>
                    <h3>Data Polusi dan Kualitas Udara</h3>
                </hgroup>
                <p>Web ini dikembangkan untuk membantu Anda memahami kondisi keberlanjutan sumber daya air serta kualitas udara di berbagai wilayah. Dengan memanfaatkan dataset polusi air dan kualitas udara, aplikasi ini mampu memproses data tersebut lalu menyajikan visualisasi, informasi statistik, serta melakukan prediksi terhadap tingkat keberlanjutan lingkungan berdasarkan masukan yang Anda berikan.</p>
                

                <h3>Distribusi Data</h3>
                <figure id="grafik">
                    <img src="{{ url_for('static', filename='distribusi_kualitas_air_udara.png') }}" alt="Grafik Kualitas dan Polusi Udara">
                    <figcaption>Visualisasi distribusi kualitas udara dan polusi air berdasarkan data yang ada.</figcaption>
                </figure>

                <h3 id="informasi">Informasi Rata-rata</h3>
                <p>
                    Berikut adalah nilai rata-rata dari kualitas udara dan polusi air yang diambil dari data yang tersedia. 
                    Nilai ini memberikan gambaran umum mengenai kondisi lingkungan secara keseluruhan dalam dataset:
                </p>
                <p>
                    - Rata-rata Kualitas Udara: <strong>{{ rata_kualitas_udara }}</strong><br>
                    - Rata-rata Polusi Air: <strong>{{ rata_polusi_air }}</strong>
                </p>
                <p>
                    Nilai-nilai ini membantu memantau kondisi lingkungan dan dapat digunakan untuk mengidentifikasi wilayah yang memerlukan perhatian khusus.
                </p>
            </section>
        </main>
        <!-- Form Input Deskripsi untuk Prediksi Kondisi Daerah -->
        <section id="masukkan_kondisi" aria-label="Masukkan Deskripsi Kondisi Daerah">
            <div class="container">
                <hgroup>
                    <h2>Masukkan Deskripsi Kondisi Daerah</h2>
                    <h3>Berikan Deskripsi Singkat</h3>
                </hgroup>
                <form action="/prediksi_kondisi" method="post" class="grid">
                    <textarea name="deskripsi" placeholder="Masukkan deskripsi kondisi daerah" required></textarea>
                    <button type="submit">Prediksi</button>
                </form>
            </div>
        </section>

        <!-- Form Input Data -->
        <section id="masukkan" aria-label="Masukkan Data">
            <div class="container">
                <hgroup>
                    <h2>Masukkan Data Anda</h2>
                    <h3>Kontribusi untuk Pemantauan Lingkungan</h3>
                </hgroup>
                <form action="/prediksi" method="post" class="grid">
                    <input type="number" name="kualitas_udara" placeholder="Nilai Kualitas Udara" required>
                    <input type="number" name="polusi_air" placeholder="Nilai Polusi Air" required>
                    <button type="submit">Prediksi</button>
                </form>
            </div>
        </section>

        <!-- Footer -->
        <footer class="container">
            <small>
                <a href="#">Kebijakan Privasi</a> • <a href="#">Tentang Kami</a>
            </small>
        </footer>
    </body>
    </html>
    ''', kualitas_udara="65.4", polusi_air="50.2", rata_kualitas_udara="62.85", rata_polusi_air="44.64")
@app.route('/prediksi_kondisi', methods=['POST'])
@app.route('/prediksi_kondisi', methods=['POST'])
def prediksi_kondisi():
    deskripsi = request.form['deskripsi']
    
    # Memuat model yang sudah disimpan
    model = joblib.load(MODEL_TEXT_CLASSIFIER_PATH)
    
    # Menggunakan model untuk memprediksi
    prediksi = model.predict([deskripsi])[0]
    
    return render_template_string('''
    <h1>Hasil Prediksi</h1>
    <p>Deskripsi: <strong>{{ deskripsi }}</strong></p>
    <p>Kondisi Daerah: <strong>{{ prediksi }}</strong></p>
    <a href="/">Kembali</a>
    ''', deskripsi=deskripsi, prediksi=prediksi)


@app.route('/prediksi', methods=['POST'])
def prediksi():
    kualitas_udara = float(request.form['kualitas_udara'])
    polusi_air = float(request.form['polusi_air'])

    # Cari kota dengan nilai indeks mirip
    data_terkait = data[
        (data['AirQuality'].between(kualitas_udara - 5, kualitas_udara + 5)) &
        (data['WaterPollution'].between(polusi_air - 5, polusi_air + 5))
    ]

    return render_template_string('''
    <h1>Hasil Prediksi</h1>
    <p>Kualitas Udara: <strong>{{ kualitas_udara }}</strong></p>
    <p>Polusi Air: <strong>{{ polusi_air }}</strong></p>

    <h2>Kota dengan Indeks Mirip</h2>
    {% if kota_terkait %}
        <table border="1" cellpadding="5">
            <tr>
                <th>Kota</th>
                <th>Kualitas Udara</th>
                <th>Polusi Air</th>
            </tr>
            {% for row in kota_terkait %}
            <tr>
                <td>{{ row['City'] }}</td>
                <td>{{ row['AirQuality'] }}</td>
                <td>{{ row['WaterPollution'] }}</td>
            </tr>
            {% endfor %}
        </table>
    {% else %}
        <p>Tidak ada kota dengan nilai indeks yang serupa.</p>
    {% endif %}
    <a href="/">Kembali</a>
    ''', kualitas_udara=kualitas_udara, polusi_air=polusi_air, 
         kota_terkait=data_terkait.to_dict(orient='records'))

if __name__ == '__main__':
    app.run(debug=True)
