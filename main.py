import tensorflow as tf
import matplotlib.pyplot as plt

# Step 1: Load the dataset
train_dataset = tf.keras.utils.image_dataset_from_directory(
    'dataset/train',
    image_size=(180, 180),
    batch_size=32
)

test_dataset = tf.keras.utils.image_dataset_from_directory(
    'dataset/test',
    image_size=(180, 180),
    batch_size=32
)

# Step 2: Build the CNN model
model = tf.keras.Sequential([
    tf.keras.layers.Rescaling(1./255, input_shape=(180, 180, 3)),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')  # Binary classification
])

# Step 3: Compile the model
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Step 4: Train the model
history = model.fit(
    train_dataset,
    validation_data=test_dataset,
    epochs=5
)





#extera for predict
model.save('my_model.h5')






# Step 5: Evaluate the model
test_loss, test_accuracy = model.evaluate(test_dataset)
print(f"Test accuracy: {test_accuracy:.2f}")

# Step 6: Plot accuracy and loss
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.show()
