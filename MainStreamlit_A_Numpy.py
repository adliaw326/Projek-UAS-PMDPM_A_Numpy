import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

# Load model yang sudah dilatih
model = load_model(r'd:\Matakuliah Semester 5\Pembelajaran Mesin dan Pembelajaran Mendalam\UAS\Googlenet_A_Numpy_RAFI\gugelnet.h5')

# Nama kelas yang sesuai dengan 3 jenis kacang
class_names = ['Kacang Tanah', 'Kacang Mete', 'Kacang Almond']

def classify_image(image_path):
    try:
        # Memuat dan menyiapkan gambar
        input_image = tf.keras.utils.load_img(image_path, target_size=(180, 180))
        input_image_array = tf.keras.utils.img_to_array(input_image)
        input_image_exp_dim = tf.expand_dims(input_image_array, 0)

        # Prediksi menggunakan model
        predictions = model.predict(input_image_exp_dim)
        result = tf.nn.softmax(predictions[0])

        # Mendapatkan kelas dengan probabilitas tertinggi
        class_idx = np.argmax(result)
        confidence_scores = result.numpy()

        return class_names[class_idx], confidence_scores
    except Exception as e:
        return "Error", str(e)

def custom_progress_bar(confidence, color1, color2, color3):
    percentage1 = confidence[0] * 100
    percentage2 = confidence[1] * 100
    percentage3 = confidence[2] * 100
    progress_html = f"""
    <div style="border: 1px solid #ddd; border-radius: 5px; overflow: hidden; width: 100%; font-size: 14px;">
        <div style="width: {percentage1:.2f}%; background: {color1}; color: white; text-align: center; height: 24px; float: left;">
            {percentage1:.2f}% Kacang Tanah
        </div>
        <div style="width: {percentage2:.2f}%; background: {color2}; color: white; text-align: center; height: 24px; float: left;">
            {percentage2:.2f}% Kacang Mete
        </div>
        <div style="width: {percentage3:.2f}%; background: {color3}; color: white; text-align: center; height: 24px; float: left;">
            {percentage3:.2f}% Kacang Almond
        </div>
    </div>
    """
    st.sidebar.markdown(progress_html, unsafe_allow_html=True)

st.title("Prediksi Klasifikasi Jenis Kacang - Numpy")

uploaded_files = st.file_uploader("Unggah Gambar (Beberapa diperbolehkan)", type=["jpg", "png", "jpeg"], accept_multiple_files=True)

if st.sidebar.button("Prediksi"):
    if uploaded_files:
        st.sidebar.write("### Hasil Prediksi")
        for uploaded_file in uploaded_files:
            with open(uploaded_file.name, "wb") as f:
                f.write(uploaded_file.getbuffer())

            label, confidence = classify_image(uploaded_file.name)
            
            if label != "Error":
                primary_color = "#007BFF"
                secondary_color = "#FF4136"
                tertiary_color = "#4CAF50"
                label_color = primary_color if label == "Kacang Tanah" else secondary_color if label == "Kacang Mete" else tertiary_color
                
                st.sidebar.write(f"**Nama File:** {uploaded_file.name}")
                st.sidebar.markdown(f"<h4 style='color: {label_color};'>Prediksi: {label}</h4>", unsafe_allow_html=True)
                
                st.sidebar.write("**Confidence:**")
                for i, class_name in enumerate(class_names):
                    st.sidebar.write(f"- {class_name}: {confidence[i] * 100:.2f}%")
                
                custom_progress_bar(confidence, primary_color, secondary_color, tertiary_color)
                
                st.sidebar.write("---")
            else:
                st.sidebar.error(f"Kesalahan saat memproses gambar {uploaded_file.name}: {confidence}")
    else:
        st.sidebar.error("Silakan unggah setidaknya satu gambar untuk diprediksi.")

if uploaded_files:
    st.write("### Preview Gambar")
    for uploaded_file in uploaded_files:
        image = Image.open(uploaded_file)
        st.image(image, caption=f"{uploaded_file.name}", use_column_width=True)
