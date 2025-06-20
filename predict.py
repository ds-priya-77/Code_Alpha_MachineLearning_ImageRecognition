
from keras.utils import load_img, img_to_array
from keras.models import load_model
import numpy as np
import tensorflow as tf

# Load the trained model
model = load_model('my_model.h5')

# Load your test image
img = load_img('cat.54.jpg', target_size=(180, 180))  # make sure file exists
img_array = img_to_array(img)
img_array = tf.expand_dims(img_array, 0)  # shape: (1, 180, 180, 3)

# Make prediction
prediction = model.predict(img_array)

# For sigmoid output, threshold at 0.5
if prediction[0][0] > 0.5:
    print(f"Predicted: Dog 🐶 ({prediction[0][0]*100:.2f}% confidence)")
else:
    print(f"Predicted: Cat 🐱 ({(1 - prediction[0][0])*100:.2f}% confidence)")
