
import keras
from keras.models import Sequential ,load_model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.utils import to_categorical
from get_train_test import get_train_test

import matplotlib.pyplot as plt


X_train, X_test, y_train, y_test = get_train_test()

X_train = X_train.reshape(X_train.shape[0], 20, 130, 1)
X_test = X_test.reshape(X_test.shape[0], 20, 130, 1)
y_train_hot = to_categorical(y_train,18)
print("y_train_hot = ")
print(y_train_hot)
y_test_hot = to_categorical(y_test ,18)
model = Sequential()


model.add(Conv2D(32, kernel_size=(2, 2), activation='relu', input_shape=(20, 130, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(18, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

#model.predict(X_train, batch_size=100, verbose=1)
print("----0" , X_train.shape)
model = load_model('my_model.h5')
history = model.fit(X_train, y_train_hot, batch_size=10, epochs=500, verbose=1, validation_data=(X_test, y_test_hot))
print("y_train = ")
print(y_train)

print("y_test = ")
print(y_test)
model.save("./my_model.h5")

plt.figure(figsize=(12,4))
plt.plot(history.history['acc'])
plt.title("model acc")
plt.ylabel("acc")
plt.xlabel("epoch")
plt.legend(["train","test"] ,loc="upper left")
plt.savefig("acc_pic.png")
