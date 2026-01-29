import streamlit as st
import pickle
import numpy as np
import json
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences


# ------------------------------
# Load saved files
# ------------------------------
@st.cache_resource
def load_resources():
    model = load_model("model/quote_generator_model.keras")
    with open("model/tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    with open("json/model_params.json", "r") as f:
        model_params = json.load(f)
        max_len = model_params["max_len"]
    return model, tokenizer, max_len


model, tokenizer, max_len = load_resources()


# ------------------------------
# Prediction function
reverse_word_index = {v: k for k, v in tokenizer.word_index.items()}


def generate_quote(seed_text, next_words=25, temperature=0.8, top_k=50):
    """
    Generate text with improved quality

    Args:
        seed_text: Starting text
        next_words: Number of words to generate
        temperature: Randomness (0.5=safe, 1.0=balanced, 1.5=creative)
        top_k: Only sample from top k predictions (prevents bad words)
    """
    result = seed_text.lower()

    for _ in range(next_words):
        # Tokenize current text
        token_list = tokenizer.texts_to_sequences([result])[0]
        token_list = pad_sequences([token_list], maxlen=max_len, padding="pre")

        # Get predictions
        predictions = model.predict(token_list, verbose=0)[0]

        # ============================================
        # FILTER BAD TOKENS
        # ============================================
        # Remove <PAD> and <UNK>
        predictions[0] = 0  # <PAD>
        predictions[1] = 0  # <UNK>

        # Renormalize
        if predictions.sum() == 0:
            break
        predictions = predictions / predictions.sum()

        # ============================================
        # TOP-K SAMPLING (Better quality)
        # ============================================
        # Get top k predictions
        top_k_indices = np.argsort(predictions)[-top_k:]
        top_k_probs = predictions[top_k_indices]

        # Apply temperature to top-k only
        top_k_probs = np.power(top_k_probs, 1 / temperature)
        top_k_probs = top_k_probs / top_k_probs.sum()

        # Sample from top-k
        predicted_id = np.random.choice(top_k_indices, p=top_k_probs)

        # ============================================
        # GET WORD
        # ============================================
        output_word = reverse_word_index.get(predicted_id, "")

        # Safety check
        if not output_word or output_word in ["<UNK>", "<PAD>"]:
            break

        result += " " + output_word

    return result


# ------------------------------
# Streamlit UI
# ------------------------------
st.set_page_config(page_title="Next Word Prediction", layout="centered")

st.title("🧠 Next Word Prediction (GRU)")
st.write("Enter a sentence and the model will predict the **next word**.")

user_input = st.text_input("✍️ Enter text:", placeholder="Type a sentence here...")

if st.button("Predict Next Word of Quote"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        next_word = generate_quote(user_input)
        st.success(f"**Predicted Next Word of Quote:** {next_word}")

# ------------------------------
# Footer
# ------------------------------
st.markdown("---")
st.caption("GRU-based Next Word Prediction using Streamlit")
