import pandas as pd
import numpy as np
import re
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Embedding

# ================================
# 1. Load Dataset
# ================================
print("Loading dataset...")
df = pd.read_csv("eng_french.csv", names=["english", "french"])

# Clean text function
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-zA-ZÀ-ÿ?.!,¿']+", " ", text)  # keep French accented chars
    return text.strip()

df["english"] = df["english"].apply(clean_text)
df["french"] = df["french"].apply(clean_text)

# Add <start> and <end> tokens to French (target language)
df["french"] = df["french"].apply(lambda x: "<start> " + x + " <end>")

# ================================
# 2. Tokenization
# ================================
print("Tokenizing...")
eng_tokenizer = Tokenizer(filters='')
eng_tokenizer.fit_on_texts(df["english"])
eng_sequences = eng_tokenizer.texts_to_sequences(df["english"])
eng_word_index = eng_tokenizer.word_index

french_tokenizer = Tokenizer(filters='')
french_tokenizer.fit_on_texts(df["french"])
french_sequences = french_tokenizer.texts_to_sequences(df["french"])
french_word_index = french_tokenizer.word_index

max_eng_len = max(len(seq) for seq in eng_sequences)
max_french_len = max(len(seq) for seq in french_sequences)

encoder_input_data = pad_sequences(eng_sequences, maxlen=max_eng_len, padding="post")
decoder_input_data = pad_sequences(french_sequences, maxlen=max_french_len, padding="post")

# Decoder target data (shifted by one for teacher forcing)
decoder_target_data = np.zeros_like(decoder_input_data)
decoder_target_data[:, :-1] = decoder_input_data[:, 1:]

# ================================
# 3. Build Seq2Seq Model
# ================================
latent_dim = 256  # LSTM hidden size

# Encoder
encoder_inputs = Input(shape=(max_eng_len,))
enc_emb = Embedding(len(eng_word_index) + 1, latent_dim)(encoder_inputs)
encoder_lstm = LSTM(latent_dim, return_state=True)
encoder_outputs, state_h, state_c = encoder_lstm(enc_emb)
encoder_states = [state_h, state_c]

# Decoder
decoder_inputs = Input(shape=(max_french_len,))
dec_emb_layer = Embedding(len(french_word_index) + 1, latent_dim)
dec_emb = dec_emb_layer(decoder_inputs)
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(dec_emb, initial_state=encoder_states)
decoder_dense = Dense(len(french_word_index) + 1, activation="softmax")
decoder_outputs = decoder_dense(decoder_outputs)

# Seq2Seq Model
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

# ================================
# 4. Train
# ================================
model.compile(optimizer="rmsprop", loss="sparse_categorical_crossentropy")

print("Training model...")
history = model.fit(
    [encoder_input_data, decoder_input_data],
    np.expand_dims(decoder_target_data, -1),
    batch_size=64,
    epochs=5,
    validation_split=0.1
)

# ================================
# 5. Save Model & Tokenizers
# ================================
model.save("eng_french_seq2seq.h5")

import pickle
with open("eng_tokenizer.pkl", "wb") as f:
    pickle.dump(eng_tokenizer, f)
with open("french_tokenizer.pkl", "wb") as f:
    pickle.dump(french_tokenizer, f)

print("✅ Training complete! Model saved as eng_french_seq2seq.h5")
