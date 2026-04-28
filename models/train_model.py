import numpy as np
import os
import pickle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def train_asl_model():
    # Create models directory
    os.makedirs("models", exist_ok=True)
    
    # Check if training data exists
    data_dir = "data/training_data"
    if not os.path.exists(data_dir) or len(os.listdir(data_dir)) == 0:
        print("No training data found!")
        return
    
    # Load data
    print("Loading training data...")
    X = []
    y = []
    
    for filename in os.listdir(data_dir):
        if filename.endswith(".npy"):
            letter = filename.split("_")[0]
            features = np.load(os.path.join(data_dir, filename))
            X.append(features)
            y.append(letter)
    
    X = np.array(X)
    y = np.array(y)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    print("Training model...")
    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model accuracy: {accuracy:.2f}")
    
    # Save model
    model_file = "models/asl_model.pkl"
    with open(model_file, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved to {model_file}")

if __name__ == "__main__":
    print("Training ASL Recognition Model")
    train_asl_model()
