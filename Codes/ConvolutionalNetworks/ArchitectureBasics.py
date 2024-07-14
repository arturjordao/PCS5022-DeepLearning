from keras import backend
from keras import layers
from keras import models
from keras.layers import *
import numpy as np
from tensorflow.data import Dataset
from sklearn.metrics._classification import accuracy_score
from sklearn.metrics import top_k_accuracy_score

if __name__ == '__main__':
    np.random.seed(12227)

    input_shape = (64, 64, 3)
    num_classes = 10

    n_samples = 1000
    X = np.random.rand(n_samples, 64, 64, 3)
    y = np.eye(num_classes)[np.random.randint(0, num_classes, n_samples)]

    inputs = Input(input_shape)
    H = inputs
    filters = 16
    for s in range(3):#Stages
        for _ in range(4):#Layers per stage
            H = Conv2D(filters, (3, 3), padding='same', strides=1)(H)
            H = BatchNormalization(axis=3)(H)
            H = Activation('relu')(H)

        if s==0:
            H = Conv2D(filters, (3, 3), padding='same', strides=2)(H)
        if s==1:
            H = AveragePooling2D(pool_size=(2, 2))(H)
        if s==2:
            H = MaxPool2D(pool_size=(2, 2))(H)

        filters = filters * 2

    H = GlobalAveragePooling2D()(H)
    output = Dense(num_classes, activation='softmax', name='probs')(H)
    model = models.Model(inputs, output)

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()

    data = Dataset.from_tensor_slices((X, y)).shuffle(4 * 128).batch(128)
    model.fit(data, batch_size=128, verbose=2, epochs=1)

    y_pred = model.predict(X)

    for k in range(3):
        top_k = top_k_accuracy_score(np.argmax(y, axis=1), y_pred, k=1)
        print('Top[{}] Accuracy [{:.4f}]'.format(k, top_k))
