from flask import Flask, render_template, request
import torch
import torch.nn as nn
import torch.optim as optim
import pickle

app = Flask(__name__)

# ==============================
# Load your trained model + tokenizer
# ==============================

# Example: loading saved objects (adjust paths as per your train.py save)
with open("tokenizer_eng.pkl", "rb") as f:
    tokenizer_eng = pickle.load(f)

with open("tokenizer_fr.pkl", "rb") as f:
    tokenizer_fr = pickle.load(f)

# Load model
model = torch.load("translation_model.pth", map_location=torch.device('cpu'))
model.eval()

# ==============================
# Simple function for translation
# ==============================
def translate_sentence(sentence):
    # Tokenize input
    tokens = [tokenizer_eng.get(w, tokenizer_eng["<UNK>"]) for w in sentence.lower().split()]
    input_tensor = torch.tensor(tokens).unsqueeze(0)

    with torch.no_grad():
        output = model(input_tensor)
    
    # Pick predicted words
    predicted_indices = output.argmax(2).squeeze().tolist()
    translated_words = [list(tokenizer_fr.keys())[list(tokenizer_fr.values()).index(idx)] 
                        for idx in predicted_indices if idx in tokenizer_fr.values()]

    return " ".join(translated_words)

# ==============================
# Routes
# ==============================
@app.route('/')
def index():
    return render_template("index.html")

@app.route('/translate', methods=['POST'])
def translate():
    english_text = request.form['english_text']
    french_translation = translate_sentence(english_text)
    return render_template("index.html", english_text=english_text, french_translation=french_translation)


if __name__ == "__main__":
    app.run(debug=True)
