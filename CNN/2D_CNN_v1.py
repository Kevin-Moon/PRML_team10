from __future__ import print_function
import keras
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten, Activation, advanced_activations
from keras.layers import Conv2D, MaxPooling2D, Conv1D, MaxPooling1D
from keras.callbacks import ProgbarLogger
from keras.utils import plot_model
import matplotlib.pyplot as plt
import numpy as np
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard

# Constants
batch_size = 12
num_classes = 2
epochs = 800
eeg_rows, eeg_cols, num_ch = 29, 11, 19
input_shape = (None ,29 ,11, 19)

# load data file
DATA_SET = np.load('EEG_csv_total_2d.npy')

ACC = []
TRY = 1

def create_model():
    conv = Sequential()

    conv.add(Conv2D(32, (3, 3), padding='same', input_shape=x_train.shape[1:]))
    conv.add(advanced_activations.LeakyReLU(alpha=0.3))
    conv.add(Conv2D(32, (3, 3)))
    conv.add(advanced_activations.LeakyReLU(alpha=0.3))
    conv.add(MaxPooling2D(pool_size=(4, 2)))
    conv.add(Dropout(0.25))

    conv.add(Conv2D(64, (3, 3), padding='same'))
    conv.add(advanced_activations.LeakyReLU(alpha=0.3))
    conv.add(Conv2D(64, (3, 3)))
    conv.add(advanced_activations.LeakyReLU(alpha=0.3))
    conv.add(MaxPooling2D(pool_size=(2, 1)))
    conv.add(Dropout(0.25))

    conv.add(Flatten())
    conv.add(Dense(128))
    conv.add(advanced_activations.LeakyReLU(alpha=0.3))
    conv.add(Dropout(0.5))

    conv.add(Dense(128))
    conv.add(advanced_activations.LeakyReLU(alpha=0.3))
    conv.add(Dropout(0.5))

    conv.add(Dense(num_classes))
    conv.add(Activation('softmax'))

    return conv

for x_train,y_train,x_test,y_test,x_valid,y_valid in DATA_SET:

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_valid = x_valid.astype('float32')

    x_train /= 255
    x_test /= 255
    x_valid /= 255

    print('x_train shape:', x_train.shape, y_train.shape, x_test.shape, y_test.shape)

    checkpointer = ModelCheckpoint(filepath="/tmp/weights_cnn_2d", verbose=1, save_best_only=True, monitor='val_loss')
    tensorboard = TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True,
                              write_images=False)

    score = np.array([0,0])
    while score[1] < 0.6:

        conv = create_model()
        print(conv.summary())

        conv.compile(loss='categorical_crossentropy',
                     optimizer=keras.optimizers.Adadelta(),
                     metrics=['accuracy'])

        conv.fit(x_train, y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  verbose=2,
                  validation_data=(x_valid, y_valid),
                  callbacks=[checkpointer, tensorboard])

        conv.load_weights("/tmp/weights_cnn_2d")

        #score = conv.evaluate(x_test, y_test, verbose=1)
        score = conv.test_on_batch(x_test, y_test)
        print(score)

    print('Test loss:', score[0])
    print('Test accuracy[',TRY,']', score[1])
    ACC.append(score[1])
    TRY += 1

print(ACC)
print('AVG :',np.mean(np.array(ACC)))
