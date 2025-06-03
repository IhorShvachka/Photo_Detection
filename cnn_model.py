import cv2
import numpy as np
import os
from tensorflow.keras import layers, models, optimizers, initializers
from tensorflow.keras.preprocessing.image import ImageDataGenerator

class CNNModel:
    def __init__(self, input_shape=(64, 64, 3)):
        self.input_shape = input_shape
        self.model = self.build_model(input_shape)

    def build_model(self, input_shape):
        initializer = initializers.GlorotUniform()

        model = models.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu',
                          kernel_initializer=initializer,
                          input_shape=input_shape),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),

            layers.Conv2D(64, (3, 3), activation='relu',
                          kernel_initializer=initializer),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),

            layers.Conv2D(128, (3, 3), activation='relu',
                          kernel_initializer=initializer),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),

            layers.Conv2D(256, (3, 3), activation='relu',
                          kernel_initializer=initializer),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),

            layers.Flatten(),
            layers.Dense(256, activation='relu',
                         kernel_initializer=initializer),
            layers.Dropout(0.5),
            layers.Dense(1, activation='sigmoid',
                         kernel_initializer=initializer)
        ])

        model.compile(
            optimizer=optimizers.Adam(learning_rate=0.0001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        return model

    def train(self, X_train, y_train, X_val, y_val, epochs=30, batch_size=32, callbacks=None):
        if not self.validate_data(X_train, y_train, X_val, y_val):
            raise ValueError("Invalid training data detected")

        datagen = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest')

        history = self.model.fit(
            datagen.flow(X_train, y_train, batch_size=batch_size),
            epochs=epochs,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=0
        )
        return history

    def validate_data(self, X_train, y_train, X_val, y_val):
        checks = [
            (not np.isnan(X_train).any(), "NaN values in X_train"),
            (not np.isinf(X_train).any(), "Inf values in X_train"),
            (X_train.min() >= 0, "Negative values in X_train"),
            (X_train.max() <= 1, "Values > 1 in X_train"),
            (len(np.unique(y_train)) == 2, "Invalid labels in y_train")
        ]

        for condition, message in checks:
            if not condition:
                print(f"Validation Error: {message}")
                return False
        return True

    def predict(self, image):
        try:
            img = cv2.resize(image, (self.input_shape[0], self.input_shape[1]))
            img = img / 255.0
            if len(img.shape) == 3 and img.shape[2] == 3:
                img = np.expand_dims(img, axis=0)
                prediction = self.model.predict(img, verbose=0)
                return prediction[0][0]
            else:
                raise ValueError("Invalid image dimensions")
        except Exception as e:
            print(f"Prediction error: {str(e)}")
            return 0.5

    def save_model(self, filepath='saved_model.h5'):
        try:
            self.model.save(filepath)
            print(f"Model successfully saved to {filepath}")
            return True
        except Exception as e:
            print(f"Error saving model: {str(e)}")
            return False

    def load_model(self, filepath='saved_model.h5'):
        if os.path.exists(filepath):
            try:
                self.model = models.load_model(filepath)
                print(f"Model successfully loaded from {filepath}")
                return True
            except Exception as e:
                print(f"Error loading model: {str(e)}")
                return False
        return False