from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)
model = joblib.load('svr_pipeline.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/panduan')
def panduan():
    return render_template('panduan.html')

@app.route('/tentang')
def tentang():
    return render_template('tentang.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Ambil data input dari form
        luas_bangunan = float(request.form['luas_bangunan'])
        luas_tanah = float(request.form['luas_tanah'])
        kt = int(request.form['kamar_tidur'])
        km = int(request.form['kamar_mandi'])
        lantai = int(request.form['jumlah_lantai'])
        total_rooms = int(request.form['total_rooms'])
        harga_per_m2 = float(request.form['harga_per_m2'])

        # Fitur tambahan
        bangunan_log = np.log1p(luas_bangunan)

        # Urutan fitur sesuai training
        features = np.array([[kt, km, lantai, luas_bangunan, luas_tanah, total_rooms, bangunan_log, harga_per_m2]])

        # Prediksi
        prediction_log = model.predict(features)[0]
        prediction = np.expm1(prediction_log)  # balikan log1p

        # Konversi ke rupiah (jika 1 juta satuan dasar)
        prediction_full = prediction * 1_000_000
        pred_rupiah = f"{prediction_full:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")

        return render_template(
            'index.html',
            prediction=pred_rupiah,
            luas_bangunan=luas_bangunan,
            luas_tanah=luas_tanah,
            kamar_tidur=kt,
            kamar_mandi=km,
            jumlah_lantai=lantai,
            total_rooms=total_rooms,
            harga_per_m2=harga_per_m2
        )

    except Exception as e:
        return f"Terjadi kesalahan: {e}"

if __name__ == '__main__':
    app.run(debug=True)
