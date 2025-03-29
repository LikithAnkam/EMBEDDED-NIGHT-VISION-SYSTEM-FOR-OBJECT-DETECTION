import cv2
import numpy as np
import os
import joblib
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from skimage.feature import haar_like_feature
from skimage.transform import integral_image
from sklearn.model_selection import train_test_split

def load_images_from_folder(folder):
    images = []
    labels = []
    for label, subfolder in enumerate(["negatives", "positives"]):  
        path = os.path.join(folder, subfolder)
        for filename in os.listdir(path):
            img = cv2.imread(os.path.join(path, filename), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img = cv2.resize(img, (24, 24))  # Resize for consistency
                images.append(img)
                labels.append(label)  # 0 = No pedestrian, 1 = Pedestrian
    return np.array(images), np.array(labels)

def extract_haar_features(images):
    features = []
    for img in images:
        ii = integral_image(img)  # Convert to integral image
        haar_features = haar_like_feature(ii, 0, 0, ii.shape[0], ii.shape[1], feature_type='type-2-x')
        features.append(haar_features)
    return np.array(features)

def train_adaboost():
    # Load dataset
    dataset_path = "testImages"
    images, labels = load_images_from_folder(dataset_path)

    # Extract Haar-like features
    features = extract_haar_features(images)

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

    # Train AdaBoost with Decision Tree stumps
    model = AdaBoostClassifier(
        base_estimator=DecisionTreeClassifier(max_depth=1),  
        n_estimators=50,  
        learning_rate=1.0  
    )

    model.fit(X_train, y_train)

    # Save trained model
    joblib.dump(model, "adaboost_pedestrian.pkl")
    print("Training completed. Model saved as 'adaboost_pedestrian.pkl'.")


