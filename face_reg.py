import streamlit as st
import numpy as np
from keras.models import Sequential, Model
from keras.layers import Dense, Flatten, Reshape, Embedding, SimpleRNN, LSTM, Input, LeakyReLU, Conv2D, MaxPooling2D, Conv2DTranspose
from keras.utils import to_categorical
from keras.datasets import mnist, imdb
from keras.preprocessing.sequence import pad_sequences
from sklearn.datasets import fetch_lfw_people
import tensorflow as tf

st.set_page_config(page_title="Deep Learning Demos", layout="wide")
st.title("🧠 Deep Learning Mini Projects")

# Sidebar navigation
option = st.sidebar.selectbox("Choose a demo", [
    "1. XOR Problem (DNN)",
    "2. MNIST Digit Recognition (CNN)",
    "3. Face Recognition (CNN)",
    "4. Character-level Language Model (RNN)",
    "5. Sentiment Analysis (IMDB - LSTM)",
    "6. POS Tagging (Seq2Seq)",
    "7. GAN (Image Generation)"
])

if option == "1. XOR Problem (DNN)":
    st.subheader("XOR Problem using DNN")
    X = np.array([[0,0], [0,1], [1,0], [1,1]])
    y = np.array([[0], [1], [1], [0]])
    model = Sequential([
        Dense(4, input_dim=2, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X, y, epochs=1000, verbose=0)
    preds = model.predict(X)
    st.write("Predictions:")
    st.write(preds)

elif option == "2. MNIST Digit Recognition (CNN)":
    st.subheader("Digit Recognition using CNN (MNIST)")
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = X_train.reshape(-1,28,28,1)/255.0
    y_train_cat = to_categorical(y_train)
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
        MaxPooling2D((2,2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train_cat, epochs=1, batch_size=32)
    loss, acc = model.evaluate(X_train, y_train_cat)
    st.write(f"Train Accuracy: {acc:.2f}")

elif option == "3. Face Recognition (CNN)":
    st.subheader("Face Recognition using CNN")
    faces = fetch_lfw_people(min_faces_per_person=70, resize=0.4)
    X = faces.images / 255.0
    y = faces.target
    X = X.reshape(-1, 50, 37, 1)
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=(50,37,1)),
        MaxPooling2D((2,2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(len(faces.target_names), activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(X, y, epochs=1)
    loss, acc = model.evaluate(X, y)
    st.write(f"Training Accuracy: {acc:.2f}")

elif option == "4. Character-level Language Model (RNN)":
    st.subheader("Character-level Language Model using RNN")
    text = "hello world"
    chars = sorted(list(set(text)))
    char_indices = {c:i for i,c in enumerate(chars)}
    indices_char = {i:c for i,c in enumerate(chars)}
    X = np.array([char_indices[c] for c in text])
    model = Sequential([
        Embedding(len(chars), 50, input_length=1),
        SimpleRNN(100),
        Dense(len(chars), activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
    model.fit(X[:-1].reshape(-1,1), X[1:], epochs=100, verbose=0)
    seed = "h"
    for _ in range(len(text)-1):
        pred = np.argmax(model.predict(np.array([char_indices[seed[-1]]]).reshape(1,1)), axis=-1)[0]
        seed += indices_char[pred]
    st.write("Generated text:", seed)

elif option == "5. Sentiment Analysis (IMDB - LSTM)":
    st.subheader("Sentiment Analysis using LSTM (IMDB)")
    max_features = 10000
    (X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=max_features)
    X_train = pad_sequences(X_train, maxlen=200)
    X_test = pad_sequences(X_test, maxlen=200)
    model = Sequential([
        Embedding(max_features, 128),
        LSTM(64),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=1, batch_size=64)
    loss, acc = model.evaluate(X_test, y_test)
    st.write(f"Test Accuracy: {acc:.2f}")

elif option == "6. POS Tagging (Seq2Seq)":
    st.subheader("POS Tagging using Sequence-to-Sequence Model")
    encoder_input = np.random.rand(100, 10, 50)
    decoder_input = np.random.rand(100, 10, 50)
    decoder_target = np.random.rand(100, 10, 50)
    encoder_inputs = Input(shape=(None, 50))
    encoder = LSTM(50, return_state=True)
    _, state_h, state_c = encoder(encoder_inputs)
    decoder_inputs = Input(shape=(None, 50))
    decoder_lstm = LSTM(50, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=[state_h, state_c])
    decoder_dense = Dense(50, activation='softmax')
    decoder_outputs = decoder_dense(decoder_outputs)
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    model.compile(optimizer='adam', loss='categorical_crossentropy')
    model.fit([encoder_input, decoder_input], decoder_target, epochs=1)
    st.write("Seq2Seq model trained on dummy data for POS tagging")

elif option == "7. GAN (Image Generation)":
    st.subheader("Simple GAN Image Generator")
    generator = Sequential([
        Dense(128, activation=LeakyReLU(0.2), input_dim=100),
        Dense(784, activation='sigmoid'),
        Reshape((28,28))
    ])
    discriminator = Sequential([
        Flatten(input_shape=(28,28)),
        Dense(128, activation=LeakyReLU(0.2)),
        Dense(1, activation='sigmoid')
    ])
    discriminator.compile(optimizer='adam', loss='binary_crossentropy')
    generator.compile(optimizer='adam', loss='binary_crossentropy')
    noise = np.random.normal(0, 1, (10, 100))
    gen_imgs = generator.predict(noise)
    for img in gen_imgs:
        st.image(img, width=100, clamp=True)

