import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


mnist = keras.datasets.mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()


X_train = X_train.astype("float32") / 255.0
X_test = X_test.astype("float32") / 255.0


X_train = X_train.reshape(-1, 28, 28, 1)  
X_test = X_test.reshape(-1, 28, 28, 1)


model = keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)), 
    layers.MaxPooling2D((2, 2)),  
    layers.Conv2D(64, (3, 3), activation='relu'),  
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),  
    layers.Dense(128, activation='relu'),  
    layers.Dense(10, activation='softmax')  
])


model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


model.fit(X_train, y_train, epochs=5, batch_size=32, validation_data=(X_test, y_test))


test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Độ chính xác của CNN: {test_acc:.4f}")

predictions = model.predict(X_test)
fig, axes = plt.subplots(2, 5, figsize=(10, 5))
for i, ax in enumerate(axes.flat):
    ax.imshow(X_test[i].reshape(28, 28), cmap='gray')
    predicted_label = np.argmax(predictions[i])  
    ax.set_title(f"Dự đoán: {predicted_label}")
    ax.axis("off")
plt.show()

model.save("mnist_cnn_model.keras")

