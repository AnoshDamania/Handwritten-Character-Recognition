from tensorflow import keras as kr
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import time

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

data = pd.read_csv("C:/ML/A_Z Handwritten Data.csv").astype('float32')
data_array = np.array(data, dtype=np.uint8)

y = data_array[:, 0]

X = data_array[:, 1:].reshape(372450, 28, 28)

X = X.reshape(372450, 28, 28, 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)

train_y_c = to_categorical(y_train, num_classes=26, dtype='int')

test_y_c = to_categorical(y_test, num_classes=26, dtype='int')

s_time = time.time()
model = kr.models.Sequential()
model.add(kr.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(kr.layers.MaxPooling2D((2, 2)))
model.add(kr.layers.Conv2D(64, (3, 3), activation='relu'))
model.add(kr.layers.MaxPooling2D((2, 2)))
model.add(kr.layers.Conv2D(64, (3, 3), activation='relu'))

model.add(kr.layers.Flatten())
model.add(kr.layers.Dense(64, activation="relu"))
model.add(kr.layers.Dense(128, activation="relu"))
model.add(kr.layers.Dense(26, activation="softmax"))

model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()

history = model.fit(X_train, train_y_c, epochs=3, validation_data=(X_test, test_y_c))

print("\n\n")
test_loss, test_acc = model.evaluate(X_test, test_y_c, verbose=1)

print("The loss is : ", test_loss)
print("The accuracy is : ", test_acc)

time_taken = time.time() - s_time
print("\n\n", time_taken)
