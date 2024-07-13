import numpy as np
from sklearn.datasets import make_classification
from sklearn import preprocessing
from keras.optimizers import *
from tensorflow.keras.optimizers import SGD
import tensorflow as tf
from keras.layers import Dense, Input
from keras.models import Model
import matplotlib.pyplot as plt

def simple_setup():
    n_classes = 3
    X, y = make_classification(n_samples=500, n_features=1000, n_classes=n_classes, n_clusters_per_class=1)

    le = preprocessing.LabelBinarizer()
    y = le.fit_transform(y)

    inputs = Input(shape=X.shape[-1])

    dense = Dense(units=8, activation="relu")(inputs)
    dense = Dense(units=16, activation="relu")(dense)
    dense = Dense(units=32, activation="relu")(dense)
    dense = Dense(units=64, activation="relu")(dense)
    outputs = Dense(units=n_classes, activation="softmax")(dense)  #
    model = Model(inputs=inputs, outputs=outputs)

    opt = SGD(learning_rate=0.01)
    model.compile(loss='categorical_crossentropy', optimizer=opt)
    model.fit(X, y, epochs=10)
    return model, X, y

if __name__ == "__main__":
    np.random.seed(12227)

    model, X, y = simple_setup()

    theta = model.get_weights()
    delta = []
    eta = []
    
    for layer in model.get_weights():
        # set standard normal parameters
        dist_x = np.random.multivariate_normal([1], np.eye(1), layer.shape).reshape(layer.shape)
        dist_y = np.random.multivariate_normal([1], np.eye(1), layer.shape).reshape(layer.shape)
        delta.append(dist_x * layer / np.linalg.norm(dist_x))
        eta.append(dist_y * layer / np.linalg.norm(dist_y))
    
    grid_lenght = 20# Increasing this values enhances visualization details (and computacional cost)
    _range = 1
    
    spacing = np.linspace(-_range, _range, grid_lenght)
    x_axis, y_axis = np.meshgrid(spacing, spacing)
    Z = np.zeros(shape=(grid_lenght, grid_lenght))
    
    for i in range(0, grid_lenght):
    
        for j in range(0, grid_lenght):
            theta_star = [theta[x] + x_axis[i][j] * delta[x] + y_axis[i][j] * eta[x] for x in range(len(theta))]
    
            model.set_weights(theta_star)
            loss = model.evaluate(X, y, verbose=0, batch_size=32)
            print('{} {} {}'.format(x_axis[i][j], y_axis[i][j], loss), flush=True)
            Z[i][j] = loss
    
    
    fig, ax = plt.subplots()
    vmin = 0.1
    vmax = 10
    vlevel = 0.5
    ax = fig.add_subplot(111, projection='3d')

    #Z = (Z-np.min(Z)) / (np.max(Z) - np.max(Z))
    max_val = np.max(Z)
    min_val = np.min(Z)

    mat_min_subtracted = Z - min_val

    Z = mat_min_subtracted / (max_val - min_val)
    Z = np.log(Z+0.0001)
    surf = ax.plot_surface(x_axis, y_axis, Z, cmap='summer')
    plt.show()
    
