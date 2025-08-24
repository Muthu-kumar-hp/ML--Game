# streamlit_app.py
import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
import joblib
import random
import time
from collections import deque, Counter
import threading
from PIL import Image

class StreamlitRPSGame:
    def __init__(self):
        self.init_session_state()
        self.load_model()
    
    def init_session_state(self):
        """Initialize Streamlit session state variables"""
        if 'player_score' not in st.session_state:
            st.session_state.player_score = 0
        if 'computer_score' not in st.session_state:
            st.session_state.computer_score = 0
        if 'round_number' not in st.session_state:
            st.session_state.round_number = 0
        if 'player_history' not in st.session_state:
            st.session_state.player_history = deque(maxlen=10)
        if 'last_result' not in st.session_state:
            st.session_state.last_result = ""
        if 'game_active' not in st.session_state:
            st.session_state.game_active = False
        if 'webcam_active' not in st.session_state:
            st.session_state.webcam_active = False
        if 'current_gesture' not in st.session_state:
            st.session_state.current_gesture = "unknown"
        if 'gesture_confidence' not in st.session_state:
            st.session_state.gesture_confidence = 0.0
    
    def load_model(self):
        """Load the trained model and label encoder"""
        try:
            self.model = tf.keras.models.load_model('gesture_model.h5')
            self.label_encoder = joblib.load('label_encoder.pkl')
            self.model_loaded = True
        except Exception as e:
            st.error(f"Could not load model: {e}")
            self.model_loaded = False
    
    def preprocess_frame(self, frame):
        """Preprocess frame for model prediction"""
        try:
            # Convert BGR to RGB for display
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Define hand region (center of frame)
            h, w = frame.shape[:2]
            x, y, roi_w, roi_h = w//4, h//4, w//2, h//2
            
            # Extract and preprocess hand region
            hand_region = frame[y:y+roi_h, x:x+roi_w]
            gray = cv2.cvtColor(hand_region, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (64, 64))
            gray = cv2.GaussianBlur(gray, (5, 5), 0)
            gray = gray.astype('float32') / 255.0
            
            # Reshape for model
            processed = gray.reshape(1, 64, 64, 1)
            
            # Draw rectangle on display frame
            cv2.rectangle(frame_rgb, (x, y), (x+roi_w, y+roi_h), (0, 255, 0), 2)
            
            return processed, frame_rgb, (x, y, roi_w, roi_h)
        except:
            return None, frame, None
    
    def predict_gesture(self, processed_frame):
        """Predict gesture from processed frame"""
        if not self.model_loaded or processed_frame is None:
            return "unknown", 0.0
        
        try:
            prediction = self.model.predict(processed_frame, verbose=0)
            predicted_class = np.argmax(prediction)
            confidence = np.max(prediction)
            gesture_name = self.label_encoder.inverse_transform([predicted_class])[0]
            return gesture_name, confidence
        except:
            return "unknown", 0.0
    
    def get_computer_move(self):
        """Generate computer move with strategy"""
        gestures = ['stone', 'paper', 'scissors']
        
        if len(st.session_state.player_history) < 3:
            return random.choice(gestures)
        
        # Simple strategy: counter most frequent recent move
        recent_moves = list(st.session_state.player_history)[-5:]
        most_common = Counter(recent_moves).most_common(1)[0][0]
        
        counters = {'stone': 'paper', 'paper': 'scissors', 'scissors': 'stone'}
        predicted_move = counters.get(most_common, random.choice(gestures))
        
        # Add randomness (70% strategy, 30% random)
        return predicted_move if random.random() > 0.3 else random.choice(gestures)
    
    def determine_winner(self, player_move, computer_move):
        """Determine winner of the round"""
        if player_move == computer_move:
            return "tie"
        
        win_conditions = {
            ('stone', 'scissors'): 'player',
            ('scissors', 'paper'): 'player',
            ('paper', 'stone'): 'player'
        }
        
        return win_conditions.get((player_move, computer_move), 'computer')
    
    def play_round(self, player_move):
        """Play a single round"""
        computer_move = self.get_computer_move()
        winner = self.determine_winner(player_move, computer_move)
        
        # Update history
        st.session_state.player_history.append(player_move)
        st.session_state.round_number += 1
        
        # Update scores
        if winner == 'player':
            st.session_state.player_score += 1
            result = f"🎉 You Win! You: {player_move.upper()} vs Computer: {computer_move.upper()}"
        elif winner == 'computer':
            st.session_state.computer_score += 1
            result = f"🤖 Computer Wins! You: {player_move.upper()} vs Computer: {computer_move.upper()}"
        else:
            result = f"🤝 Tie! Both played: {player_move.upper()}"
        
        st.session_state.last_result = result
        return result
    
    def reset_game(self):
        """Reset all game statistics"""
        st.session_state.player_score = 0
        st.session_state.computer_score = 0
        st.session_state.round_number = 0
        st.session_state.player_history.clear()
        st.session_state.last_result = "Game Reset!"
    
    def run_streamlit_app(self):
        """Main Streamlit application"""
        st.set_page_config(
            page_title="Rock Paper Scissors AI",
            page_icon="✂️",
            layout="wide"
        )
        
        st.title("🎮 Rock Paper Scissors with AI")
        st.markdown("---")
        
        # Sidebar for controls
        st.sidebar.title("🎯 Game Controls")
        
        # Model status
        if self.model_loaded:
            st.sidebar.success("✅ AI Model Loaded")
        else:
            st.sidebar.error("❌ AI Model Not Found")
            st.sidebar.info("Please train the model first by running:")
            st.sidebar.code("python model_training.py")
        
        # Game statistics
        st.sidebar.markdown("### 📊 Score Board")
        col1, col2 = st.sidebar.columns(2)
        with col1:
            st.metric("👤 You", st.session_state.player_score)
        with col2:
            st.metric("🤖 Computer", st.session_state.computer_score)
        
        st.sidebar.metric("🏁 Round", st.session_state.round_number)
        
        # Controls
        st.sidebar.markdown("### 🎮 Controls")
        if st.sidebar.button("🔄 Reset Game", use_container_width=True):
            self.reset_game()
            st.rerun()
        
        # Main content area
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("### 📹 Camera Feed")
            
            if not self.model_loaded:
                st.error("Please train the AI model first!")
                st.info("1. Run `python data_collection.py` to collect gesture data")
                st.info("2. Run `python model_training.py` to train the model")
                st.info("3. Refresh this page")
                return
            
            # Camera controls
            camera_col1, camera_col2 = st.columns(2)
            with camera_col1:
                start_camera = st.button("📸 Start Camera", use_container_width=True)
            with camera_col2:
                stop_camera = st.button("⏹️ Stop Camera", use_container_width=True)
            
            if start_camera:
                st.session_state.webcam_active = True
            if stop_camera:
                st.session_state.webcam_active = False
            
            # Camera feed placeholder
            camera_placeholder = st.empty()
            
            # Webcam processing
            if st.session_state.webcam_active:
                try:
                    cap = cv2.VideoCapture(0)
                    
                    if not cap.isOpened():
                        st.error("Could not open camera!")
                        st.session_state.webcam_active = False
                    else:
                        while st.session_state.webcam_active:
                            ret, frame = cap.read()
                            if not ret:
                                break
                            
                            frame = cv2.flip(frame, 1)  # Mirror effect
                            processed_frame, display_frame, roi = self.preprocess_frame(frame)
                            
                            if processed_frame is not None:
                                gesture, confidence = self.predict_gesture(processed_frame)
                                st.session_state.current_gesture = gesture
                                st.session_state.gesture_confidence = confidence
                                
                                # Add prediction text to frame
                                if roi:
                                    x, y, w, h = roi
                                    color = (0, 255, 0) if confidence > 0.7 else (255, 255, 0)
                                    cv2.putText(display_frame, f"{gesture.upper()}: {confidence:.2f}", 
                                              (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                            
                            # Convert to PIL for Streamlit
                            pil_image = Image.fromarray(display_frame)
                            camera_placeholder.image(pil_image, channels="RGB", use_column_width=True)
                            
                            time.sleep(0.1)  # Control frame rate
                        
                        cap.release()
                
                except Exception as e:
                    st.error(f"Camera error: {e}")
                    st.session_state.webcam_active = False
        
        with col2:
            st.markdown("### 🎯 Current Detection")
            
            # Current gesture display
            gesture_icons = {
                'stone': '✊',
                'paper': '✋', 
                'scissors': '✌️',
                'unknown': '❓'
            }
            
            current_icon = gesture_icons.get(st.session_state.current_gesture, '❓')
            st.markdown(f"## {current_icon} {st.session_state.current_gesture.title()}")
            
            confidence_color = "green" if st.session_state.gesture_confidence > 0.7 else "orange" if st.session_state.gesture_confidence > 0.4 else "red"
            st.markdown(f"**Confidence:** :{confidence_color}[{st.session_state.gesture_confidence:.1%}]")
            
            st.markdown("### 🎲 Play Your Move")
            
            # Manual play buttons
            play_col1, play_col2, play_col3 = st.columns(3)
            
            with play_col1:
                if st.button("✊\nStone", use_container_width=True, key="stone_btn"):
                    result = self.play_round("stone")
                    st.success(result)
                    st.rerun()
            
            with play_col2:
                if st.button("✋\nPaper", use_container_width=True, key="paper_btn"):
                    result = self.play_round("paper")
                    st.success(result)
                    st.rerun()
            
            with play_col3:
                if st.button("✌️\nScissors", use_container_width=True, key="scissors_btn"):
                    result = self.play_round("scissors")
                    st.success(result)
                    st.rerun()
            
            # Auto-play with detected gesture
            st.markdown("---")
            if (st.session_state.current_gesture != "unknown" and 
                st.session_state.gesture_confidence > 0.7):
                
                if st.button("🚀 Play Detected Gesture", use_container_width=True):
                    result = self.play_round(st.session_state.current_gesture)
                    st.success(result)
                    st.rerun()
            else:
                st.info("Make a clear gesture to enable auto-play")
            
            # Last result
            if st.session_state.last_result:
                st.markdown("### 🏆 Last Result")
                st.info(st.session_state.last_result)
            
            # Game rules
            st.markdown("### 📋 Rules")
            st.markdown("""
            - ✊ **Stone** beats ✌️ Scissors
            - ✋ **Paper** beats ✊ Stone  
            - ✌️ **Scissors** beats ✋ Paper
            - Same moves = Tie
            
            **Tips:**
            - Position your hand in the green rectangle
            - Make clear gestures for better detection
            - The AI learns from your patterns!
            """)

# Main execution
if __name__ == "__main__":
    game = StreamlitRPSGame()
    game.run_streamlit_app()