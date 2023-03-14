import streamlit as st

st.title("AutoNN - Neural Network Automation Platform")
st.write("Train your NN here...")

num_neurons = st.sidebar.slider("Numbre of Neurons in the Hidden Layer:", 2, 64)
num_epochs = st.sidebar.slider("Number of epochs:",2,16)
activation = st.sidebar.text_input("Enter Activation Function:")

"The activation function is: " + activation

if st.button("Train a Model"):
    import tensorflow as tf
    from tensorflow.keras.datasets import mnist
    from tensorflow.keras.layers import *
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.callbacks import ModelCheckpoint

    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    def preprocess_images(images):
        images = images / 255
        return images
    X_train = preprocess_images(X_train)
    X_test = preprocess_images(X_test)

    model = Sequential()
    model.add(InputLayer((28, 28)))
    model.add(Flatten())
    model.add(Dense(num_neurons, activation))
    model.add(Dense(10))
    model.add(Softmax())
    model.compile(loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    cp = ModelCheckpoint('model', save_best_only=True)
    history_cp=tf.keras.callbacks.CSVLogger('history.csv', separator=",", append=False)
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=num_epochs, callbacks=[cp, history_cp])

if st.button("Plot the the Loss Curve"):
    import pandas as pd
    import matplotlib.pyplot as plt
    history = pd.read_csv('history.csv')
    fig = plt.figure()
    plt.plot(history['epoch'], history['accuracy'], )
    plt.plot(history['epoch'], history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['Train', 'Val'], loc='lower right')
    fig