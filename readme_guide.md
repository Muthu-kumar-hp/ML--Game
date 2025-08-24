# 🎮 Rock Paper Scissors AI Game

A complete **Computer Vision + Machine Learning** implementation of Rock Paper Scissors using **OpenCV**, **TensorFlow/Keras**, and **Streamlit**.

## 🌟 Features

- **Real-time Hand Gesture Recognition** using webcam
- **CNN Model** trained to classify Stone ✊, Paper ✋, and Scissors ✌️
- **Smart Computer AI** that learns from your patterns
- **Live Score Tracking** and game statistics  
- **Two Game Modes**: OpenCV desktop app and Streamlit web interface
- **Bounding Box Detection** with confidence scores
- **Model Persistence** - save and load trained models

## 📋 Requirements

- Python 3.8+
- Webcam/Camera
- GPU recommended (but not required) for training

## 🚀 Quick Start

### Option 1: Automated Setup

```bash
# Clone/download all files to a directory
# Install requirements
python setup_and_run.py --install

# Run complete setup (data collection → training → game)
python setup_and_run.py --mode setup
```

### Option 2: Step-by-Step Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Collect training data
python data_collection.py

# 3. Train the model
python model_training.py

# 4. Play the game!
python rock_paper_scissors_game.py

# OR use Streamlit interface
streamlit run streamlit_app.py
```

## 📁 Project Structure

```
rock-paper-scissors-ai/
│
├── data_collection.py          # Collect gesture training data
├── model_training.py           # Train CNN model
├── rock_paper_scissors_game.py # Main OpenCV game
├── streamlit_app.py           # Streamlit web interface
├── setup_and_run.py           # Automated setup script
├── requirements.txt           # Python dependencies
│
├── gesture_data/              # Training data (created automatically)
│   ├── stone/
│   ├── paper/
│   └── scissors/
│
├── gesture_model.h5           # Trained model (created after training)
├── label_encoder.pkl          # Label encoder (created after training)
└── training_history.png       # Training plots
```

## 🎯 How to Use

### 1. Data Collection Phase

Run `data_collection.py`:
- Position your hand in the **green rectangle**
- Follow the on-screen instructions for each gesture
- **Stone**: Make a fist ✊
- **Paper**: Open palm facing camera ✋
- **Scissors**: Peace sign with index and middle finger ✌️
- Press **SPACE** to start/pause collection
- Press **ESC** to move to the next gesture
- Collect ~300 images per gesture for best results

### 2. Training Phase  

Run `model_training.py`:
- Uses a **Convolutional Neural Network** (CNN)
- Includes **data augmentation** and **regularization**
- Training typically takes 5-15 minutes
- Model saved as `gesture_model.h5`
- Achieves 90%+ accuracy with good data

### 3. Playing the Game

**OpenCV Version** (`rock_paper_scissors_game.py`):
- Real-time webcam feed with gesture prediction
- Press **SPACE** to start a round
- 3-second countdown, then make your gesture
- Computer uses strategy based on your history
- Press **R** to reset scores, **ESC** to quit

**Streamlit Version** (`streamlit_app.py`):
- Web-based interface
- Live camera feed with gesture detection
- Click buttons or use detected gestures
- Real-time score tracking
- More user-friendly interface

## 🧠 Technical Details

### Model Architecture

```python
CNN Model:
├── Conv2D(32) + BatchNorm + MaxPool + Dropout
├── Conv2D(64) + BatchNorm + MaxPool + Dropout  
├── Conv2D(128) + BatchNorm + MaxPool + Dropout
├── Flatten
├── Dense(512) + BatchNorm + Dropout
├── Dense(256) + BatchNorm + Dropout
└── Dense(3, softmax) # stone, paper, scissors
```

### Computer AI Strategy

- **Pattern Recognition**: Analyzes your last 5-10 moves
- **Counter Strategy**: Plays the move that beats your most frequent choice
- **Randomness**: 30% random moves to stay unpredictable
- Gets smarter as you play more rounds!

### Image Preprocessing Pipeline

1. **ROI Extraction**: Focus on hand region (300x300px)
2. **Grayscale Conversion**: Reduce complexity
3. **Gaussian Blur**: Noise reduction
4. **Resize**: Scale to 64x64 for model input
5. **Normalization**: Pixel values 0-1
6. **Data Augmentation**: Rotation, shift, zoom during training

## 🎮 Game Controls

### OpenCV Game
- **SPACE**: Start new round
- **R**: Reset scores  
- **ESC**: Quit game

### Streamlit Game
- **Start Camera**: Begin webcam feed
- **Stone/Paper/Scissors Buttons**: Manual play
- **Play Detected Gesture**: Use AI prediction
- **Reset Game**: Clear all scores

## 🔧 Troubleshooting

### Camera Issues
```bash
# Test camera access
python -c "import cv2; cap = cv2.VideoCapture(0); print('Camera OK' if cap.read()[0] else 'Camera Error'); cap.release()"
```

### Model Not Found
```bash
# Check files exist
ls -la *.h5 *.pkl

# Retrain if needed
python model_training.py
```

### Low Accuracy
- Collect more training data (500+ per gesture)
- Ensure good lighting conditions
- Make gestures clearly and consistently
- Avoid background clutter

### Performance Issues
- Close other applications using camera
- Reduce video resolution in code if needed
- Use GPU for training (install tensorflow-gpu)

## 📊 Performance Metrics

With proper training data:
- **Model Accuracy**: 90-95%
- **Real-time FPS**: 15-30 FPS
- **Prediction Latency**: <50ms
- **Training Time**: 5-15 minutes (CPU)

## 🎯 Advanced Usage

### Custom Model Training
```python
# Modify model architecture in model_training.py
model = keras.Sequential([
    # Add your custom layers here
    layers.Conv2D(64, (5, 5), activation='relu'),
    # ... more layers
])
```

### Adjust Detection Sensitivity
```python
# In main game, change confidence threshold
if confidence > 0.6:  # Lower = more sensitive
    # Process gesture
```

### Add New Gestures
1. Modify `gestures` list in data collection
2. Collect data for new gesture
3. Retrain model
4. Update game logic

## 🤝 Contributing

Feel free to improve the project:
- Add more gestures (rock, lizard, spock?)
- Improve model architecture
- Better UI/UX design
- Mobile app version
- Multiplayer support

## 📜 License

Open source - feel free to use and modify!

## 🎉 Have Fun!

Enjoy playing against the AI and watch it learn your patterns! The more you play, the smarter it gets. Can you outsmart the machine? 🤖✨