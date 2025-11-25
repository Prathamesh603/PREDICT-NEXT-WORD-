# Next-Word Prediction with LSTM & GRU (Hamlet)

This project implements next-word prediction using neural sequence models (LSTM and GRU) trained on Shakespeare's "Hamlet" (from NLTK's Gutenberg). The notebook `experiments.ipynb` contains all data preparation, model definitions, training, prediction helpers, and model saving.

## Quick Summary
- **Task:** Given a short input sequence, predict the next word.
- **Dataset:** `shakespeare-hamlet.txt` from NLTK Gutenberg corpus (saved as `hamlet.txt`).
- **Models:** Two sequence models:
  - LSTM-based model (Embedding -> LSTM(150, return_sequences=True) -> Dropout(0.2) -> LSTM(100) -> Dense)
  - GRU-based model (same architecture but GRU layers)
- **Training:** Categorical cross-entropy loss, `adam` optimizer. LSTM trained for 150 epochs, GRU trained for 100 epochs in the notebook.

## Preprocessing & Sliding-window (how sequences are created)

The notebook uses this sequence construction procedure (a sliding / n-gram style approach):

- Load raw text and lowercase it.
- Tokenize the entire text with `Tokenizer()` to get integer indices for words. `total_word = len(tokenizer.word_index) + 1`.
- For each line in the text (split by `\n`), convert the line into a sequence of token ids `token_list`.
- For each position i in `token_list`, build an n-gram sequence `token_list[:i+1]`. This creates many input sequences of increasing length (1-grams up to the full line length).
- Pad all sequences to the same `max_sequence_len` using pre-padding: `pad_sequences(..., maxlen=max_sequence_len, padding='pre')`.
- Split each padded sequence into predictors and label:
  - `x = sequences[:, :-1]` (all tokens except last)
  - `y = sequences[:, -1]` (the next word)
- Convert `y` to one-hot with `tf.keras.utils.to_categorical(y, num_classes=total_word)`.
- Train/test split: `train_test_split(..., test_size=0.2)`.

Notes:
- The notebook builds sequences per-line (`text.split('\n')`) rather than sliding across whole-document boundaries. Depending on dataset formatting, you may want to join lines to allow longer context across sentences.
- `max_sequence_len` is computed as the maximum length among generated n-gram sequences.

## Model architectures (notebook)

- LSTM model (in notebook):
```
Embedding(total_word, 100, input_length=max_sequence_len-1)
LSTM(150, return_sequences=True)
Dropout(0.2)
LSTM(100)
Dense(total_word, activation='softmax')
```

- GRU model (in notebook): same as above but using `GRU(150, return_sequences=True)` and `GRU(100)`.

Both models are compiled with:
```
loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']
```

## Training
- The notebook trains the LSTM model for 150 epochs and the GRU model for 100 epochs using `model.fit` with validation on `x_test, y_test`.
- There is a mention of early stopping in the notebook header, but no `EarlyStopping` callback is instantiated in the executed code. Adding an `EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)` callback is recommended for quicker training and to avoid overfitting.

## Prediction helper

The notebook defines `predict_next_word(model, tokenizer, text, max_sequence_len)` which:

- Tokenizes the input `text`, truncates to the last `max_sequence_len-1` tokens if too long,
- Pads the sequence to `max_sequence_len-1`,
- Calls `model.predict(...)` and takes `argmax` to get predicted token index,
- Maps the predicted index back to the word via `tokenizer.word_index`.

Example usage from the notebook:
```
input_text = "That are so fortified against"
max_sequence_len = model.input_shape[1] + 1
next_word = predict_next_word(model, tokenize, input_text, max_sequence_len)
print(next_word)
```

## Saving / Loading
- The notebook saves the trained models and tokenizer:
  - `model.save('next_word_lstm.h5')`
  - `model_GRU.save('next_word_gru.h5')`
  - Tokenizer saved with `pickle` to `tokenizer.pickle`.

To load and use the saved model/tokenizer:
```python
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

model = load_model('next_word_lstm.h5')
with open('tokenizer.pickle','rb') as f:
    tokenizer = pickle.load(f)

def predict_next_word_from_saved(model, tokenizer, text, max_sequence_len):
    token_list = tokenizer.texts_to_sequences([text])[0]
    if len(token_list) >= max_sequence_len:
        token_list = token_list[-(max_sequence_len-1):]
    token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
    predicted = model.predict(token_list, verbose=0)
    predicted_index = int(predicted.argmax(axis=1)[0])
    for w, i in tokenizer.word_index.items():
        if i == predicted_index:
            return w
    return None
```

## Requirements
- Python 3.7+
- `tensorflow` (tested with TF 2.x)
- `nltk`, `scikit-learn`, `pandas`, `numpy`

Install with pip:
```
pip install tensorflow nltk scikit-learn pandas numpy
```

After installing NLTK data (in notebook):
```python
import nltk
nltk.download('gutenberg')
```

## Notes, caveats & suggested improvements
- The notebook tokenizes and builds sequences per line — consider tokenizing the entire text (or by sentences) to create richer n-grams across lines.
- Add `EarlyStopping` and ModelCheckpoint callbacks to save best model automatically.
- Consider using `text_to_word_sequence` and more aggressive cleaning (removing punctuation, handling contractions) depending on target quality.
- Use a validation split and inspect training/validation loss curves to detect overfitting.
- Try varying `embedding_dim`, LSTM/GRU units, and experimenting with bidirectional layers for improved performance.

## Files produced by the notebook
- `hamlet.txt` — raw text extracted from NLTK Gutenberg.
- `next_word_lstm.h5` — saved LSTM model (if training completed and saved).
- `next_word_gru.h5` — saved GRU model.
- `tokenizer.pickle` — saved tokenizer.

## Next steps I can help with
- Convert notebook steps into a runnable script (`train.py`) and include CLI args.
- Add a Streamlit app for interactive prediction.
- Add `EarlyStopping` and checkpointing to the notebook and retrain with tuned hyperparameters.

If you'd like, I can also convert the notebook into a clean, production-ready training script and produce a small Streamlit front-end for live prediction. Which of these would you like next?
