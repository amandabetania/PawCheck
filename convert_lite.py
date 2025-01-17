import tensorflow as tf

# Memuat model Keras
model = tf.keras.models.load_model('dog_disease_detection_model_same_lokal.h5')

# Konversi ke format TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Menyimpan model TFLite ke file
with open('dog_disease_detection_model(same).tflite', 'wb') as f:
    f.write(tflite_model)

print("Model berhasil dikonversi ke TensorFlow Lite!")