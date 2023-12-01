import numpy as np
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers.core import Dense, Flatten
from keras.layers.convolutional import Conv2D
from keras.optimizers import Adam
from keras.layers.pooling import MaxPooling2D
from keras.utils import to_categorical

np.random.seed(42)
(X_train, Y_train), (X_test, Y_test) = cifar10.load_data()

# Centering the data
X_train_mean = np.mean(X_train, axis = 0)
X_train_cent = X_train - X_train_mean       #Normalization
X_train_std = np.std(X_train, axis = 0)
X_train_norm = X_train_cent / X_train_std

X_test_norm = (X_test - X_train_mean) / X_train_std # getting test set ready

import matplotlib.pyplot as plt
class_names = ['airplane','automobile','bird','cat','deer',
'dog','frog','horse','ship','truck']
fig = plt.figure(figsize=(8,3))
for i in range(len(class_names)):
    ax = fig.add_subplot(2, 5, 1 + i, xticks=[], yticks=[])
    idx = np.where(Y_train[:]==i)[0]
    features_idx = X_train[idx,::]
    img_num = np.random.randint(features_idx.shape[0])
    im = features_idx[img_num,::]
    ax.set_title(class_names[i])
    #im = np.transpose(features_idx[img_num,::], (1, 2, 0))
    plt.imshow(im)
#plt.show()



# Elastic Net Regularization (L1 + L2)# Regularizer layer import
from keras.layers import Dropout
from keras.constraints import MaxNorm
from keras.regularizers import l2
from keras.regularizers import l1_l2
from keras.regularizers import l1
from keras.layers import BatchNormalization, Activation #Inizializting the model
model = Sequential() # Defining a convolutional layer
model.add(Conv2D(128, kernel_size=(3, 3), input_shape=(32, 32,
3)))
model.add(BatchNormalization())
model.add(Dropout(0.25))
model.add(Activation('relu'))# Defining a second convolutional layer
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.25))
model.add(Activation('relu'))# Defining a thirdd convolutional layer
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.25))
model.add(Activation('relu'))# We include our classifier
model.add(Flatten())
model.add(Dense(1024, activation='relu'))
#kernel_costraint=MaxNorm(max_value=3)))
#kernel_regularizer=l2(0.01)))
#kernel_regularizer=l1_l2(0.01, 0.01)))
#kernel_regularizer=l1(0.01)))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))# Compiling the
model
model.compile(loss='categorical_crossentropy',
    optimizer=Adam(lr=0.0001, decay=1e-6),
    metrics=['accuracy'])# Training the model
model.fit(X_train_norm, to_categorical(Y_train),
    batch_size=128,
    shuffle=True,
    epochs=17,
    validation_data=(X_test_norm, to_categorical(Y_test)))
scores = model.evaluate(X_test_norm,
to_categorical(Y_test))
print('Loss: %.3f' % scores[0])
print('Accuracy: %.3f' % scores[1])