import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import os
import cv2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import joblib

class GestureModelTrainer:
    def __init__(self, data_dir='dataset', img_size=(64, 64)):
        self.data_dir = data_dir
        self.img_size = img_size
        self.gestures = ['stone', 'paper', 'scissors']
        self.model = None
        self.label_encoder = LabelEncoder()
        
    def load_data(self):
        """Load and preprocess the collected gesture data"""
        print("Loading training data...")
        
        images = []
        labels = []
        
        for gesture in self.gestures:
            gesture_path = os.path.join(self.data_dir, gesture)
            if not os.path.exists(gesture_path):
                print(f"Warning: Directory {gesture_path} not found!")
                continue
                
            image_files = [f for f in os.listdir(gesture_path) if f.endswith('.jpg')]
            print(f"Found {len(image_files)} images for {gesture}")
            
            for img_file in image_files:
                img_path = os.path.join(gesture_path, img_file)
                
                # Load and preprocess image
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    img = cv2.resize(img, self.img_size)  # ✅ Ensure correct size
                    img = img.astype('float32') / 255.0
                    images.append(img)
                    labels.append(gesture)
        
        # Convert to numpy arrays
        X = np.array(images)
        y = np.array(labels)
        
        # Reshape images to include channel dimension
        X = X.reshape(X.shape[0], self.img_size[0], self.img_size[1], 1)
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)   # raw labels for stratify
        y_categorical = keras.utils.to_categorical(y_encoded, num_classes=3)
        
        print(f"Loaded {len(X)} total images")
        print(f"Data shape: {X.shape}")
        print(f"Labels shape: {y_categorical.shape}")
        
        return X, y_categorical, y_encoded
    
    def create_model(self):
        """Create CNN model for gesture recognition"""
        model = keras.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 1)),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            layers.Flatten(),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(3, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train_model(self, epochs=50, batch_size=32):
        """Train the gesture recognition model"""
        # Load data
        X, y_onehot, y_encoded = self.load_data()
        
        if len(X) == 0:
            raise RuntimeError("❌ No training data found! Please run data_collection.py first.")
        
        # Split data (use raw encoded labels for stratify)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_onehot, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        print(f"Training set size: {X_train.shape[0]}")
        print(f"Test set size: {X_test.shape[0]}")
        
        # Create model
        self.model = self.create_model()
        print("\nModel architecture:")
        self.model.summary()
        
        # Data augmentation
        datagen = keras.preprocessing.image.ImageDataGenerator(
            rotation_range=10,
            width_shift_range=0.1,
            height_shift_range=0.1,
            zoom_range=0.1,
            horizontal_flip=True,
            fill_mode='nearest'
        )
        
        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5),
            keras.callbacks.ModelCheckpoint('best_model.h5', save_best_only=True)
        ]
        
        # Train model
        print("\nStarting training...")
        history = self.model.fit(
            datagen.flow(X_train, y_train, batch_size=batch_size),
            steps_per_epoch=len(X_train) // batch_size,
            epochs=epochs,
            validation_data=(X_test, y_test),
            callbacks=callbacks,
            verbose=1
        )
        
        # Evaluate
        test_loss, test_accuracy = self.model.evaluate(X_test, y_test, verbose=0)
        print(f"\n✅ Test Accuracy: {test_accuracy:.4f}")
        print(f"✅ Test Loss: {test_loss:.4f}")
        
        # Save model + label encoder
        self.model.save('gesture_model.h5')
        joblib.dump(self.label_encoder, 'label_encoder.pkl')
        
        print("Model saved as 'gesture_model.h5'")
        print("Label encoder saved as 'label_encoder.pkl'")
        
        # Plot training history
        self.plot_training_history(history)
        
        return history
    
    def plot_training_history(self, history):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        ax1.plot(history.history['accuracy'], label='Train Accuracy')
        ax1.plot(history.history['val_accuracy'], label='Val Accuracy')
        ax1.set_title('Accuracy')
        ax1.legend()
        
        ax2.plot(history.history['loss'], label='Train Loss')
        ax2.plot(history.history['val_loss'], label='Val Loss')
        ax2.set_title('Loss')
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig('training_history.png')
        plt.show()

if __name__ == "__main__":
    trainer = GestureModelTrainer()
    
    print("=== Training Gesture Recognition Model ===")
    print("Make sure you have collected training data first!")
    
    if os.path.exists('dataset'):
        history = trainer.train_model(epochs=50, batch_size=16)
        print("\n🎉 Training completed successfully!")
    else:
        print("❌ Error: 'dataset' directory not found!")
        print("Please run data_collection.py first to collect training data.")
