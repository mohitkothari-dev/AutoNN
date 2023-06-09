import streamlit as st

st.title("AutoNN - Neural Network Automation Platform")
st.write("Training and Hyperparameter tuning on Mnist dataset")

explanation = '<p style="color:#ff4b4b; font-size: 18px;">As name suggests, this is an automated neural network training platform for mnist dataset. Here, user can create Neural Netowrk, fine tune hyperparameters, and compare the result of each epocs with validation and training accuracy.</p>'
st.markdown(explanation,unsafe_allow_html=True)

num_neurons = st.sidebar.slider("Numbre of Neurons in the Hidden Layer:", 2, 64)
num_epochs = st.sidebar.slider("Number of epochs:",2,16)
activation = st.sidebar.text_input("Enter Activation Function:")
num_hidden_layers = st.sidebar.text_input("Enter number of hidden layers: ")
if num_hidden_layers:
    try:
        num_hidden_layers = int(num_hidden_layers)
        st.write(f"Number of hidden layers are {num_hidden_layers}")
    except ValueError:
        st.write(f"Error: {num_hidden_layers} is not a valid integer")

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
    for n in range(num_hidden_layers-1):    
        model.add(Dense(10))
    model.add(Softmax())
    model.compile(loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    cp = ModelCheckpoint('model', save_best_only=True)
    history_cp=tf.keras.callbacks.CSVLogger('history.csv', separator=",", append=False)
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=num_epochs, callbacks=[cp, history_cp])

if st.button("Validate Model"):
    import pandas as pd
    import matplotlib.pyplot as plt
    history = pd.read_csv('history.csv')
    fig = plt.figure()
    plt.plot(history['epoch'], history['accuracy'], )
    plt.plot(history['epoch'], history['val_accuracy'])
    plt.title('Model Accuracy bs epochs')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['Train', 'Val'], loc='lower right')
    fig

if st.button("Get latest comparison"):
    import pandas as pd
    df = pd.read_csv('history.csv')
    st.write(df)

references = ["https://medium.com/omdena/streamlit101-deploying-an-automl-model-using-streamlit-e86c6508b5c2"]
for i,ref in enumerate(references):
    st.write(f"Reference: {i+1}", ref)