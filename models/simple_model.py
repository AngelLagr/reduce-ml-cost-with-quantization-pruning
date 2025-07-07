from tensorflow import keras
from keras import layers

def create_simple_model(input_shape):
    """
    Creates a simple feedforward neural network model.

    Parameters:
        input_shape (tuple): Shape of the input data.
    Return:
        keras.Model: A compiled Keras model.
    """
    model = keras.Sequential()
    model.add(layers.Input(shape=input_shape))
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(16, activation='relu'))
    model.add(layers.Dense(10, activation='relu'))
    
    model.add(layers.Dense(1, activation='sigmoid'))  # Output layer for binary classification
    return model

def compile_model(model):
    """
    Compiles the Keras model with an optimizer, loss function, and metrics.
    Parameters:
        model (keras.Model): The Keras model to compile.
    """

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

def train_model(model, x_train, y_train, epochs=10, batch_size=32):
    """
    Trains the Keras model on the training data.
    Parameters:
        model (keras.Model): The Keras model to train.
        X_train (array-like): Training data features.
        y_train (array-like): Training data labels.
        epochs (int): Number of epochs to train the model.
        batch_size (int): Size of the batches used in training.
    Return:
        keras.callbacks.History: History object containing training metrics.
    """
    return model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2)

def evaluate_model(model, x_test, y_test):
    """
    Evaluates the Keras model on the test data.
    Parameters:
        model (keras.Model): The Keras model to evaluate.
        X_test (array-like): Test data features.
        y_test (array-like): Test data labels.
    Return:
        tuple: Loss and accuracy of the model on the test data.
    """
    return model.evaluate(x_test, y_test)