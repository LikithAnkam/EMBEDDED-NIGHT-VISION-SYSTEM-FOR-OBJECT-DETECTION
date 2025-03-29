import joblib
import cv2
import numpy as np
from skimage.feature import haar_like_feature
from skimage.transform import integral_image

def test_adaboost(image_path):
    model = joblib.load("adaboost_pedestrian.pkl")
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (24, 24))

    # Extract Haar-like features
    ii = integral_image(img)
    features = haar_like_feature(ii, 0, 0, ii.shape[0], ii.shape[1], feature_type='type-2-x')
    features = np.array(features).reshape(1, -1)

    # Predict pedestrian presence
    prediction = model.predict(features)
    print("Pedestrian Detected" if prediction == 1 else "No Pedestrian")

# test_adaboost("temp.png")
