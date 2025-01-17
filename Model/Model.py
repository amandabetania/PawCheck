import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import regularizers, models, layers
from tensorflow.keras.optimizers import Adam

# Menggunakan ImageDataGenerator untuk preprocessing
train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=20, horizontal_flip=True)
valid_datagen = ImageDataGenerator(rescale=1./255, horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)

# Pastikan label Anda berupa angka, gunakan 'class_mode' yang sesuai
train_dataset = train_datagen.flow_from_directory(
    r'D:\Documents\KULIAH\Semester 5\Studi Independen\Project Capstone\Project\Data\Data_Preprocessed\train', 
    target_size=(224, 224),
    batch_size=32,
    class_mode='sparse'
)

valid_dataset = valid_datagen.flow_from_directory(
    r'D:\Documents\KULIAH\Semester 5\Studi Independen\Project Capstone\Project\Data\Data_Preprocessed\valid',
    target_size=(224, 224),
    batch_size=32,
    class_mode='sparse'
)

test_dataset = test_datagen.flow_from_directory(
    r'D:\Documents\KULIAH\Semester 5\Studi Independen\Project Capstone\Project\Data\Data_Preprocessed\test',
    target_size=(224, 224),
    batch_size=32,
    class_mode='sparse'
)

def create_model():
    model = models.Sequential([
        layers.Input(shape=(224, 224, 3)),

        layers.Conv2D(16, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.001)),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.4),

        layers.Conv2D(32, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.001)),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.4),

        layers.Conv2D(64, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.001)),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.4),

        layers.Conv2D(128, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.001)),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.4),

        layers.Flatten(),
        layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
        layers.Dropout(0.5),
        layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
        layers.Dense(9, activation='softmax')
    ])
    return model

model = create_model()


total_params = model.count_params()
print(f"Total parameters in the model: {total_params}")

optimizer = Adam(learning_rate=0.0001)

model.compile(optimizer=optimizer,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

class EarlyStoppingCallback(tf.keras.callbacks.Callback):

    def on_epoch_end(self, epoch, logs=None):
        # Check if the accuracy is greater or equal to 0.95 and validation accuracy is greater or equal to 0.95
        train_accuracy = logs.get('accuracy')  # Get training accuracy from logs
        val_accuracy = logs.get('val_accuracy')  # Get validation accuracy from logs

        if train_accuracy is not None and val_accuracy is not None:
            if train_accuracy >= 0.95 and val_accuracy >= 0.95:
                self.model.stop_training = True  # Stop the training

                print("\nReached 95% train accuracy and 95% validation accuracy, so cancelling training!")


# Train the model and save the training history
history = model.fit(
	train_dataset,
	epochs=150,
	validation_data=valid_dataset,
	callbacks = [EarlyStoppingCallback()]
)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(test_dataset)
print(f'Test accuracy: {test_accuracy}')

model.save('dog_disease_detection_model.h5')