# rock_paper_scissors_game.py
import cv2
import numpy as np
import tensorflow as tf
import joblib
import random
import time
from collections import deque, Counter

class RockPaperScissorsGame:
    def __init__(self, model_path='gesture_model.h5', encoder_path='label_encoder.pkl'):
        # Load trained model and label encoder
        try:
            self.model = tf.keras.models.load_model(model_path)
            self.label_encoder = joblib.load(encoder_path)
            print("Model and encoder loaded successfully!")
        except:
            print("Error: Could not load model or encoder!")
            print("Make sure you have trained the model first (run model_training.py)")
            exit()
        
        # Game settings
        self.gestures = ['stone', 'paper', 'scissors']
        self.img_size = (64, 64)
        self.cap = cv2.VideoCapture(0)
        
        # Game state
        self.player_score = 0
        self.computer_score = 0
        self.round_number = 0
        self.game_active = False
        self.countdown = 0
        self.result_message = ""
        self.last_computer_move = None
        self.last_player_move = None
        
        # For computer strategy (predicting player's next move)
        self.player_history = deque(maxlen=10)  # Keep last 10 moves
        self.prediction_confidence = 0.0
        
        # UI settings
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.8
        self.font_thickness = 2
        
    def preprocess_frame(self, frame, x, y, w, h):
        """Preprocess frame for model prediction"""
        # Extract hand region
        hand_region = frame[y:y+h, x:x+w]
        
        # Convert to grayscale
        gray = cv2.cvtColor(hand_region, cv2.COLOR_BGR2GRAY)
        
        # Resize to model input size
        gray = cv2.resize(gray, self.img_size)
        
        # Apply preprocessing (same as training)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Normalize
        gray = gray.astype('float32') / 255.0
        
        # Reshape for model
        return gray.reshape(1, self.img_size[0], self.img_size[1], 1)
    
    def predict_gesture(self, frame, x, y, w, h):
        """Predict gesture from hand region"""
        try:
            # Preprocess frame
            processed_frame = self.preprocess_frame(frame, x, y, w, h)
            
            # Make prediction
            prediction = self.model.predict(processed_frame, verbose=0)
            predicted_class = np.argmax(prediction)
            confidence = np.max(prediction)
            
            # Get gesture name
            gesture_name = self.label_encoder.inverse_transform([predicted_class])[0]
            
            return gesture_name, confidence
        
        except Exception as e:
            print(f"Prediction error: {e}")
            return "unknown", 0.0
    
    def get_computer_move(self):
        """Get computer move using strategy or random"""
        if len(self.player_history) < 3:
            # Not enough history, play random
            return random.choice(self.gestures)
        
        # Simple pattern prediction: most common move in recent history
        recent_moves = list(self.player_history)[-5:]  # Last 5 moves
        most_common = Counter(recent_moves).most_common(1)[0][0]
        
        # Counter the predicted player move
        counters = {
            'stone': 'paper',
            'paper': 'scissors', 
            'scissors': 'stone'
        }
        
        predicted_move = counters.get(most_common, random.choice(self.gestures))
        
        # Add some randomness (70% strategy, 30% random)
        if random.random() < 0.3:
            return random.choice(self.gestures)
        else:
            return predicted_move
    
    def determine_winner(self, player_move, computer_move):
        """Determine the winner of a round"""
        if player_move == computer_move:
            return "tie"
        
        win_conditions = {
            ('stone', 'scissors'): 'player',
            ('scissors', 'paper'): 'player',
            ('paper', 'stone'): 'player',
            ('scissors', 'stone'): 'computer',
            ('paper', 'scissors'): 'computer',
            ('stone', 'paper'): 'computer'
        }
        
        return win_conditions.get((player_move, computer_move), 'tie')
    
    def start_new_round(self):
        """Start a new round"""
        self.game_active = True
        self.countdown = 3
        self.round_number += 1
        self.result_message = ""
    
    def draw_ui(self, frame):
        """Draw game UI on frame"""
        height, width = frame.shape[:2]
        
        # Draw scoreboard
        score_text = f"Player: {self.player_score}  Computer: {self.computer_score}  Round: {self.round_number}"
        cv2.putText(frame, score_text, (10, 30), self.font, self.font_scale, (255, 255, 255), self.font_thickness)
        
        # Draw instructions
        if not self.game_active:
            cv2.putText(frame, "Press SPACE to start round", (10, height - 100), 
                       self.font, self.font_scale, (0, 255, 0), self.font_thickness)
            cv2.putText(frame, "Press ESC to quit", (10, height - 70), 
                       self.font, self.font_scale, (0, 255, 0), self.font_thickness)
            cv2.putText(frame, "Press R to reset scores", (10, height - 40), 
                       self.font, self.font_scale, (0, 255, 0), self.font_thickness)
        
        # Draw countdown
        if self.game_active and self.countdown > 0:
            countdown_text = f"Get Ready: {self.countdown}"
            text_size = cv2.getTextSize(countdown_text, self.font, 2, 3)[0]
            x = (width - text_size[0]) // 2
            y = (height + text_size[1]) // 2
            cv2.putText(frame, countdown_text, (x, y), self.font, 2, (0, 0, 255), 3)
        
        # Draw result
        if self.result_message:
            text_size = cv2.getTextSize(self.result_message, self.font, 1.5, 3)[0]
            x = (width - text_size[0]) // 2
            y = height - 150
            cv2.putText(frame, self.result_message, (x, y), self.font, 1.5, (255, 255, 0), 3)
        
        # Draw last moves
        if self.last_player_move and self.last_computer_move:
            moves_text = f"You: {self.last_player_move.upper()}  Computer: {self.last_computer_move.upper()}"
            cv2.putText(frame, moves_text, (10, 70), self.font, self.font_scale, (255, 255, 255), self.font_thickness)
    
    def draw_gesture_prediction(self, frame, gesture, confidence, x, y, w, h):
        """Draw gesture prediction on frame"""
        # Draw bounding box
        color = (0, 255, 0) if confidence > 0.7 else (0, 255, 255)
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        
        # Draw prediction text
        pred_text = f"{gesture.upper()}: {confidence:.2f}"
        cv2.putText(frame, pred_text, (x, y-10), self.font, self.font_scale, color, self.font_thickness)
    
    def run_game(self):
        """Main game loop"""
        print("=== Rock Paper Scissors Game ===")
        print("Instructions:")
        print("- Position your hand in the green rectangle")
        print("- Press SPACE to start a round")
        print("- Make your gesture when countdown reaches 0")
        print("- Press R to reset scores")
        print("- Press ESC to quit")
        
        frame_count = 0
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # Flip frame horizontally
            frame = cv2.flip(frame, 1)
            height, width = frame.shape[:2]
            
            # Define hand region
            x, y, w, h = 100, 100, 300, 300
            
            # Predict gesture
            gesture, confidence = self.predict_gesture(frame, x, y, w, h)
            
            # Draw prediction
            self.draw_gesture_prediction(frame, gesture, confidence, x, y, w, h)
            
            # Handle countdown
            if self.game_active and self.countdown > 0:
                if frame_count % 30 == 0:  # Update every ~1 second (30 FPS)
                    self.countdown -= 1
                    if self.countdown == 0:
                        # Round starts now - capture moves
                        if confidence > 0.6:  # Only if confident prediction
                            player_move = gesture
                            computer_move = self.get_computer_move()
                            
                            # Add to history
                            self.player_history.append(player_move)
                            
                            # Determine winner
                            winner = self.determine_winner(player_move, computer_move)
                            
                            # Update scores
                            if winner == "player":
                                self.player_score += 1
                                self.result_message = "You Win!"
                            elif winner == "computer":
                                self.computer_score += 1
                                self.result_message = "Computer Wins!"
                            else:
                                self.result_message = "It's a Tie!"
                            
                            # Store last moves
                            self.last_player_move = player_move
                            self.last_computer_move = computer_move
                        else:
                            self.result_message = "Gesture not clear, try again!"
                        
                        self.game_active = False
            
            # Draw UI
            self.draw_ui(frame)
            
            # Show frame
            cv2.imshow('Rock Paper Scissors', frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord(' '):  # Space to start round
                if not self.game_active:
                    self.start_new_round()
            elif key == ord('r') or key == ord('R'):  # Reset scores
                self.player_score = 0
                self.computer_score = 0
                self.round_number = 0
                self.result_message = "Scores Reset!"
                self.player_history.clear()
            elif key == 27:  # ESC to quit
                break
            
            frame_count += 1
        
        # Cleanup
        self.cap.release()
        cv2.destroyAllWindows()
        
        # Final scores
        print(f"\nFinal Scores:")
        print(f"Player: {self.player_score}")
        print(f"Computer: {self.computer_score}")
        
        if self.player_score > self.computer_score:
            print("You won overall! 🎉")
        elif self.computer_score > self.player_score:
            print("Computer won overall! 🤖")
        else:
            print("It's a tie overall! 🤝")

if __name__ == "__main__":
    game = RockPaperScissorsGame()
    game.run_game()