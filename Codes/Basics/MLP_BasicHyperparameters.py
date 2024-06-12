import numpy as np
from sklearn.datasets import make_classification
from sklearn import preprocessing
import keras

if __name__ == "__main__":
    np.random.seed(12227)

    n = 1000
    m = 300
    n_classes = 5
    X = np.random.rand(n, m)
    y = np.random.randint(0, n_classes, size=n)
    y = np.eye(n_classes)[y]

    inputs = keras.layers.Input(shape=X.shape[-1])

    initializer = keras.initializers.GlorotNormal(seed=12227)
    opt = keras.optimizers.SGD(learning_rate=0.01)

    H = keras.layers.Dense(units=8, activation='relu', kernel_initializer=initializer)(inputs)
    H = keras.layers.Dense(units=16, activation='relu', kernel_initializer=initializer)(H)
    H = keras.layers.Dense(units=32, activation='relu', kernel_initializer=initializer)(H)
    H = keras.layers.Dense(units=64, activation='relu', kernel_initializer=initializer)(H)
    outputs = keras.layers.Dense(units=n_classes, activation='softmax', kernel_initializer=initializer)(H)
    model = keras.models.Model(inputs=inputs, outputs=outputs)

    model.compile(loss='categorical_crossentropy', optimizer=opt)
    model.fit(X, y, epochs=10)
    
