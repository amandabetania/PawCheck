from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import os

# Initialize Flask app
app = Flask(__name__)

# Load the pre-trained model
model = load_model(r'D:\Documents\KULIAH\Semester 5\Studi Independen\Project Capstone\Project 2\dog_disease_detection_model.h5')

# Define class labels
class_labels = ['Blepharitis', 'Conjunctivitis', 'Entropion', 'Eyelid Lump', 'Nuclear Sclerosis', 'Pigmented Keratitis']

# Function to preprocess the image
def preprocess_image(image_path):
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found at path: {image_path}")
    # Resize to match model input
    resized_image = cv2.resize(image, (224, 224))
    # Normalize pixel values
    normalized_image = resized_image.astype('float32') / 255.0
    # Expand dimensions for model (1, 224, 224, 3)
    normalized_image = np.expand_dims(normalized_image, axis=0)
    return image, normalized_image

# Function to predict diseases from the image
def predict_image(model, image_path):
    # Preprocess the image
    original_image, processed_image = preprocess_image(image_path)

    # Get predictions
    predictions = model.predict(processed_image)[0]
    predictions = predictions.astype(float)

    # Sort predictions based on confidence
    top_indices = predictions.argsort()[-3:][::-1]  # Sorting indices by confidence (desc)
    top_labels = [class_labels[i] for i in top_indices]  # Map indices to class labels
    top_scores = [predictions[i] * 100 for i in top_indices]  # Convert to percentage

    # Find description and treatment based on the top prediction
    result_data = {}
    description, treatment, note = "", [], ""

    # Check if the top prediction confidence is below a threshold
    if top_scores[0] < 65:
        predicted = ("Disease not detected")
        description = (
            "The accuracy of the prediction is below 65%, so we are unable to display the detected disease. For now, we can only detect 6 diseases. If your dog shows unusual symptoms, please consult a veterinarian directly for a proper diagnosis."
        )
        note = (
            "Always consult with a veterinarian before administering any treatment to your dog. Proper diagnosis and treatment are crucial to ensure effective recovery and prevent further complications."
        )
        
        result_data = {
            "top_predictions": [
                {"label": top_labels[i], "score": top_scores[i]} for i in range(3)
            ],
            "predicted": predicted,
            "description": description,
            "note": note
        }
    else:
        if top_labels[0] == "Blepharitis":
            description = (
                "Blepharitis is an inflammation of the eyelids that can affect one or both eyelids of a dog. Common symptoms include redness, swelling, itching, and discharge from the eye area. Blepharitis can cause discomfort and may affect the dog's quality of life if left untreated."
            )
            treatment = [
                'Warm compress to reduce the risk of recurrence.',
                'Trim the hair around the eyes to reduce fluid buildup.',
                'Use baby shampoo to remove dirt that may clog the meibomian gland openings.',
            ]
        elif top_labels[0] == "Conjunctivitis":
            description = (
                "Conjunctivitis is an inflammation of the conjunctiva, which is the thin tissue lining the front of the eyeball and the inside of the eyelids. In dogs, conjunctivitis can cause redness, swelling, discharge from the eyes, and discomfort."
            )
            treatment = [
                'Use an eye cleaning solution to clean the area around the eyes from dirt and discharge.',
                'Avoid known allergens and use antihistamines if necessary.',
                'If conjunctivitis is caused by an underlying medical condition, treatment for that condition is also required.'
            ]
        elif top_labels[0] == "Entropion":
            description = (
                "Entropion is a condition where the eyelid folds inward, causing irritation to the eye. It can lead to inflammation, discomfort, and potentially damage to the cornea if untreated."
            )
            treatment = [
                'Accurate diagnosis is crucial to determine the severity of entropion.',
                'Before surgery, eye drops may be prescribed to reduce irritation.',
                'Surgery is usually the most effective treatment for entropion.'
            ]
        elif top_labels[0] == "Eyelid Lump":
            description = (
                "An eyelid lump is an abnormal growth in the dog's eyelid area. The lump could be benign or malignant, and depending on the type, treatment can vary."
            )
            treatment = [
                'A thorough examination to determine whether the lump is benign or malignant is essential.',
                'If caused by an infection, antibiotics or anti-inflammatory medications may be prescribed.',
                'Surgery may be required to remove the lump, especially if it’s cancerous or affecting vision.'
            ]
        elif top_labels[0] == "Nuclear Sclerosis":
            description = (
                "Nuclear Sclerosis is a normal aging process in dogs where the lens of the eye becomes cloudy. It doesn't impair vision as much as cataracts but should be monitored."
            )
            treatment = [
                'Monitor the dog\'s eye condition regularly for changes in vision.',
                'If cataracts develop, surgery may be needed to remove them.'
            ]
        elif top_labels[0] == "Pigmented Keratitis":
            description = (
                "Pigmented Keratitis is a condition where pigment accumulates on the cornea, potentially leading to vision issues. It often occurs alongside other conditions like dry eye."
            )
            treatment = [
                'Use artificial tears to keep the cornea moist.',
                'Steroids or NSAIDs may help reduce inflammation.',
                'Address underlying conditions like dry eye to prevent further damage.'
            ]
    
        note = ("Always consult with a veterinarian before administering any treatment to your dog. Proper diagnosis and treatment are crucial to ensure effective recovery and prevent further complications.")
        
        result_data = {
            "top_predictions": [
                {"label": top_labels[i], "score": top_scores[i]} for i in range(3)
            ],
            "predicted_class": top_labels[0],
            "description": description,
            "treatment": treatment,
            "note": note,
            "confidence": top_scores[0]
        }
    return result_data

# API Flask route for prediction
# Define the uploads folder
UPLOAD_FOLDER = './uploads'

# Ensure the uploads folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request."}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file."}), 400

    try:
        # Safely save the uploaded file with the original filename
        image_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(image_path)

        # Run the prediction
        result_data = predict_image(model, image_path)
        
        # Return the result as a JSON response
        return jsonify(result_data)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run Flask app
if __name__ == '__main__':
    app.run(debug=True)
