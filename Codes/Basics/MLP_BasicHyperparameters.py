import numpy as np
from sklearn.datasets import make_classification
from sklearn import preprocessing
from tensorflow import keras
from sklearn.metrics import accuracy_score

def lr_schedule(epoch, lr):
    if epoch < 10:
        return 0.01
    elif epoch < 25:
        return 0.001
    else:
        return 0.0001

if __name__ == "__main__":
    np.random.seed(12227)

    n = 10000
    m = 300
    n_classes = 4
    #Opt1
    X = np.random.rand(n, m)
    y = np.random.randint(0, n_classes, size=n)

    #Opt2
    X, y = make_classification(n_samples=n, n_features=m,
                               n_classes=n_classes, n_clusters_per_class=1)

    y = np.eye(n_classes)[y]

    inputs = keras.layers.Input(shape=X.shape[-1])

    initializer = keras.initializers.GlorotNormal(seed=12227)
    opt = keras.optimizers.SGD(learning_rate=0.1)

    H = keras.layers.Dense(units=8, activation='relu', kernel_initializer=initializer)(inputs)
    H = keras.layers.Dense(units=16, activation='relu', kernel_initializer=initializer)(H)
    H = keras.layers.Dense(units=32, activation='relu', kernel_initializer=initializer)(H)
    H = keras.layers.Dense(units=64, activation='relu', kernel_initializer=initializer)(H)
    outputs = keras.layers.Dense(units=n_classes, activation='softmax', kernel_initializer=initializer)(H)
    model = keras.models.Model(inputs=inputs, outputs=outputs)

    #Opt1
    scheduler = keras.callbacks.LearningRateScheduler(lr_schedule)
    model.compile(loss='categorical_crossentropy', optimizer=opt)
    model.fit(X, y, epochs=50, callbacks=[scheduler])
    y_pred = model.predict(X)
    acc = accuracy_score(np.argmax(y, axis=1), np.argmax(y_pred, axis=1))
    print(acc)

    #Opt2
    model = keras.models.Model(inputs=inputs, outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer=opt)
    T = 50
    for t in range(T):
        lr = 0.01 if t < 10 else (0.001 if t < 25 else 0.0001)
        keras.backend.set_value(model.optimizer.learning_rate, lr)
        model.fit(X, y, epochs=t, initial_epoch=t - 1)

    y_pred = model.predict(X)
    acc = accuracy_score(np.argmax(y, axis=1), np.argmax(y_pred, axis=1))
    print(acc)
