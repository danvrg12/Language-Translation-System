# AI Neural Translator

A deep learning-powered English to French translation system built with TensorFlow/Keras and deployed via Flask web application.

## Table of Contents
- [Description](#description)
- [Installation](#installation)
- [Usage](#usage)
- [Examples](#examples)
- [Features](#features)
- [Model Architecture](#model-architecture)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Description

This project implements a sequence-to-sequence (seq2seq) neural machine translation model using LSTM networks. The system translates English sentences to French using an encoder-decoder architecture with attention mechanisms. The trained model is deployed through a modern web interface built with Flask, HTML, CSS, and JavaScript.

**Problem it solves**: Provides accurate, context-aware translation between English and French using deep learning, offering better semantic understanding compared to traditional rule-based translation systems.

## Installation

1. **Clone the repository**
```bash
git clone 
cd ai-neural-translator
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install tensorflow flask pandas numpy pickle-mixin
```

4. **Download dataset**
   - Place your `eng_french.csv` file in the project root
   - Format: Two columns without headers (english, french)

5. **Create project structure**
```bash
mkdir templates static
```

6. **Set up templates**
   - Save the HTML content as `templates/index.html`

## Usage

### Training the Model

1. **Prepare your dataset**
```bash
# Ensure eng_french.csv is in the project root
# Format: "Hello","Bonjour"
#         "How are you?","Comment allez-vous?"
```

2. **Train the seq2seq model**
```bash
python train_model.py
```

3. **Training output**
   - `eng_french_seq2seq.h5` - Trained Keras model
   - `eng_tokenizer.pkl` - English tokenizer
   - `french_tokenizer.pkl` - French tokenizer

### Running the Web Application

1. **Start the Flask server**
```bash
python app.py
```

2. **Access the application**
   - Open browser to `http://localhost:5000`
   - Enter English text in the input field
   - Click "Translate" to get French translation

3. **Web interface features**
   - Real-time character counting
   - Copy translation to clipboard
   - Responsive design for mobile devices
   - Loading indicators during translation

## Examples

### Training Data Format
```csv
Hello,Bonjour
How are you?,Comment allez-vous?
I love programming,J'adore programmer
What time is it?,Quelle heure est-il?
```

### Translation Examples
```
Input:  "Hello, how are you today?"
Output: "Bonjour, comment allez-vous aujourd'hui?"

Input:  "I love learning new languages"
Output: "J'adore apprendre de nouvelles langues"

Input:  "The weather is beautiful"
Output: "Le temps est magnifique"
```

### Model Training Output
```
Loading dataset...
Tokenizing...
Training model...
Epoch 1/5
loss: 2.1234 - val_loss: 1.8765
...
✅ Training complete! Model saved as eng_french_seq2seq.h5
```

## Features

- **🧠 Deep Learning Architecture** - LSTM-based seq2seq model with teacher forcing
- **🌐 Modern Web Interface** - Beautiful, responsive UI with gradient backgrounds
- **⚡ Real-time Translation** - Fast inference with optimized model loading
- **📊 Training Metrics** - Validation loss tracking and model evaluation
- **🔤 Text Preprocessing** - Automatic cleaning and tokenization
- **📱 Mobile Responsive** - Works seamlessly on desktop and mobile
- **📋 Copy Functionality** - One-click copy translations to clipboard
- **🎯 Context Awareness** - Maintains semantic meaning across languages
- **🔒 Privacy Focused** - Local processing, no data storage

## Model Architecture

### Encoder-Decoder Structure
```python
# Encoder
encoder_inputs = Input(shape=(max_eng_len,))
enc_emb = Embedding(vocab_size_eng, latent_dim)(encoder_inputs)
encoder_lstm = LSTM(latent_dim, return_state=True)
encoder_outputs, state_h, state_c = encoder_lstm(enc_emb)

# Decoder  
decoder_inputs = Input(shape=(max_french_len,))
dec_emb = Embedding(vocab_size_fr, latent_dim)(decoder_inputs)
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(dec_emb, initial_state=[state_h, state_c])
```

### Key Specifications
- **Architecture**: Seq2Seq with LSTM cells
- **Embedding Dimension**: 256
- **Hidden Units**: 256 LSTM units
- **Optimizer**: RMSprop
- **Loss Function**: Sparse Categorical Crossentropy
- **Training Epochs**: 5 (adjustable)
- **Batch Size**: 64

### Data Processing Pipeline
1. **Text Cleaning**: Remove special characters, lowercase conversion
2. **Tokenization**: Convert text to sequences of integers
3. **Padding**: Ensure uniform sequence lengths
4. **Target Shifting**: Implement teacher forcing for training

## Configuration

### Model Parameters
```python
latent_dim = 256          # LSTM hidden size
batch_size = 64           # Training batch size
epochs = 5                # Training epochs
validation_split = 0.1    # Validation data percentage
```

### Flask Settings
```python
app.run(debug=True)       # Development mode
# For production: app.run(host='0.0.0.0', port=5000)
```

### File Paths
```python
model_path = "eng_french_seq2seq.h5"
eng_tokenizer_path = "eng_tokenizer.pkl"
french_tokenizer_path = "french_tokenizer.pkl"
dataset_path = "eng_french.csv"
```

## Contributing

Contributions are welcome! Here's how you can help:

1. **Fork the repository**
2. **Create a feature branch**
   ```bash
   git checkout -b feature/attention-mechanism
   ```
3. **Make your changes**
4. **Test thoroughly**
5. **Submit a pull request**

### Development Guidelines
- Follow PEP 8 for Python code
- Add docstrings to functions
- Test with different sentence lengths
- Ensure model convergence before deployment
- Update documentation for new features

### Potential Improvements
- Add attention mechanism to decoder
- Implement bidirectional LSTM encoder
- Support for additional language pairs
- Beam search for better translation quality
- REST API endpoints for programmatic access
- Model quantization for faster inference
