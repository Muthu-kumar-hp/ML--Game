y# setup_and_run.py
"""
Rock Paper Scissors AI Game Setup Script
This script helps you set up and run the complete game.
"""

import os
import sys
import subprocess
import argparse

def install_requirements():
    """Install required packages"""
    print("Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Requirements installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing requirements: {e}")
        return False

def check_model_exists():
    """Check if trained model exists"""
    return os.path.exists('gesture_model.h5') and os.path.exists('label_encoder.pkl')

def check_data_exists():
    """Check if training data exists"""
    data_dir = 'gesture_data'
    if not os.path.exists(data_dir):
        return False
    
    gestures = ['stone', 'paper', 'scissors']
    for gesture in gestures:
        gesture_path = os.path.join(data_dir, gesture)
        if not os.path.exists(gesture_path):
            return False
        
        # Check if directory has images
        images = [f for f in os.listdir(gesture_path) if f.endswith('.jpg')]
        if len(images) < 50:  # Minimum images per gesture
            return False
    
    return True

def run_data_collection():
    """Run data collection script"""
    print("Starting data collection...")
    try:
        subprocess.run([sys.executable, "data_collection.py"])
        return True
    except Exception as e:
        print(f"❌ Error in data collection: {e}")
        return False

def run_model_training():
    """Run model training script"""
    print("Starting model training...")
    try:
        subprocess.run([sys.executable, "model_training.py"])
        return True
    except Exception as e:
        print(f"❌ Error in model training: {e}")
        return False

def run_game(mode='opencv'):
    """Run the game"""
    if mode == 'streamlit':
        print("Starting Streamlit app...")
        subprocess.run([sys.executable, "-m", "streamlit", "run", "streamlit_app.py"])
    else:
        print("Starting OpenCV game...")
        subprocess.run([sys.executable, "rock_paper_scissors_game.py"])

def main():
    parser = argparse.ArgumentParser(description="Rock Paper Scissors AI Game Setup")
    parser.add_argument('--mode', choices=['setup', 'collect', 'train', 'play', 'streamlit'], 
                       default='setup', help='Mode to run')
    parser.add_argument('--install', action='store_true', help='Install requirements')
    
    args = parser.parse_args()
    
    print("🎮 Rock Paper Scissors AI Game Setup")
    print("=" * 40)
    
    # Install requirements if requested
    if args.install:
        if not install_requirements():
            return
    
    if args.mode == 'setup':
        # Full setup process
        print("\n📦 Checking setup...")
        
        # Check if model exists
        if check_model_exists():
            print("✅ Trained model found!")
            choice = input("Do you want to retrain the model? (y/n): ").lower()
            if choice != 'y':
                print("Skipping to game...")
                run_game()
                return
        
        # Check if data exists
        if not check_data_exists():
            print("📸 No training data found. Starting data collection...")
            print("\nInstructions for data collection:")
            print("1. Position your hand in the green rectangle")
            print("2. Follow instructions for each gesture")
            print("3. Press SPACE to start/pause collection")
            print("4. Press ESC to move to next gesture")
            
            input("Press Enter to start data collection...")
            if not run_data_collection():
                return
        else:
            print("✅ Training data found!")
        
        # Train model
        print("\n🤖 Starting model training...")
        if not run_model_training():
            return
        
        # Run game
        print("\n🎮 Starting game...")
        run_game()
    
    elif args.mode == 'collect':
        run_data_collection()
    
    elif args.mode == 'train':
        if not check_data_exists():
            print("❌ No training data found. Run data collection first.")
            return
        run_model_training()
    
    elif args.mode == 'play':
        if not check_model_exists():
            print("❌ No trained model found. Run setup first.")
            return
        run_game()
    
    elif args.mode == 'streamlit':
        if not check_model_exists():
            print("❌ No trained model found. Run setup first.")
            return
        run_game(mode='streamlit')

if __name__ == "__main__":
    main()