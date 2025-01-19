# Hand Gesture Recognition using CNN

This project involves building and training a Convolutional Neural Network (CNN) to recognize hand gestures from images. The dataset used contains 20,000 images across 10 classes of gestures.

## Project Overview
The goal of this project is to develop a model capable of accurately identifying hand gestures using image data. The trained model can be used in applications such as sign language recognition or gesture-based controls.

---

## Files and Directories
- `hand_gesture_model.keras`: The trained Keras model for gesture recognition.
- `predict_gesture.py`: Script to predict gestures for a given image.
- Dataset:
  - Located at: `C:\Users\apput\Downloads\archive (1)\leapGestRecog\leapGestRecog`
  - Structure:
    ```
    leapGestRecog/
    ├── 00/
    │   ├── 01_palm/
    │   └── frame_00_01_0001.png
    │   └── ...
    ├── 01/
    └── ...
    ```

---

## Setup Instructions
### Prerequisites
1. Python 3.x
2. Libraries:
   - `numpy`
   - `tensorflow`
   - `Pillow`
   - `sklearn`
   - `matplotlib`

### Steps to Run
1. Clone or download the repository.
2. Install the required libraries using pip:
   ```bash
   pip install numpy tensorflow Pillow scikit-learn matplotlib
   ```
3. Place the dataset in the appropriate directory as specified.
4. Train the model (if required) or use the pre-trained model.

---

## Model Training
The model is a CNN with the following architecture:
1. **Convolutional Layers**:
   - 2 Conv2D layers followed by MaxPooling2D and Dropout layers.
2. **Dense Layers**:
   - A fully connected Dense layer followed by a Dropout layer.
   - Softmax layer for classification.

### Training Script
- The training script loads the dataset, preprocesses the images, splits the data into training and testing sets, and trains the CNN.

### Model Performance
- **Accuracy:** Achieved nearly 100% accuracy on both training and validation sets.

---

## Predicting Gestures
Use the `predict_gesture.py` script to predict the gesture for a given image.

### Example Usage
```python
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image

# Load the trained model
model = load_model("hand_gesture_model.keras")

# Define the function to predict gestures
def predict_gesture(image_path, model, label_encoder, image_size=(64, 64)):
    img = Image.open(image_path).convert('L').resize(image_size)
    img_array = np.array(img) / 255.0
    img_array = img_array.reshape(1, image_size[0], image_size[1], 1)
    prediction = model.predict(img_array)
    class_index = np.argmax(prediction)
    class_label = label_encoder.inverse_transform([class_index])[0]
    return class_label

# Example Image Path
image_path = r"C:\Users\apput\Downloads\archive (1)\leapGestRecog\leapGestRecog\00\01_palm\frame_00_01_0001.png"

# Predict Gesture
print(predict_gesture(image_path, model, label_encoder))
```

---

## Results
- The model was able to accurately classify gestures across all 10 classes.
- Example classes:
  - `01_palm`
  - `02_l`
  - `03_fist`
  - `04_fist_moved`
  - `05_thumb`

---

## Future Improvements
1. **Augment Dataset**:
   - Increase variability in hand sizes, backgrounds, and lighting conditions.
2. **Optimize Model**:
   - Experiment with different architectures and hyperparameters.
3. **Real-Time Prediction**:
   - Integrate the model with a live video feed for real-time gesture recognition.

---

## Acknowledgments
- Dataset Source: [LeapGestRecog Dataset](https://www.kaggle.com)
- TensorFlow Documentation

## License
This project is for educational purposes and adheres to the dataset license conditions.

