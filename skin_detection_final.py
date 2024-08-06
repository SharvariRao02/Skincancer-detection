import cv2
import numpy as np
from sklearn.svm import SVC 
from sklearn.metrics import accuracy_score

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    return image, gray, blurred

def segment_image(blurred):
    _, thresholded = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thresholded = cv2.bitwise_not(thresholded)
    contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def extract_features(image, contours):
    features = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        roi = image[y:y+h, x:x+w]
        mean_val = cv2.mean(roi)
        features.append(mean_val[:3])
    return np.array(features)

# Load and preprocess image
image_path = r"D:\cg\cancer_image.jpg"  
image, gray, blurred = preprocess_image(image_path)

# Segment image
contours = segment_image(blurred)

# Extract features
features = extract_features(image, contours)

# Example training data (replace with actual data)
X_train = np.random.rand(100, 3)
y_train = np.random.randint(0, 2, 100)

# Train classifier
classifier = SVC(kernel='linear')
classifier.fit(X_train, y_train)

# Predict
y_pred = classifier.predict(features)
prediction = y_pred[0]
if prediction == 1:
    print("The lesion is predicted to be malignant(cancerous).")
else:
    print("The lesion is predicted to be benign(noncancerous).")
