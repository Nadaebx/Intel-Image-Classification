"""
Intel Image Classifier — Streamlit Deployment App
Bonus deployment: Upload an image → get prediction from both models.
Run: streamlit run app/app.py
"""

import os
import io
import json
import numpy as np
import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title='Intel Scene Classifier',
    page_icon='🏔️',
    layout='wide'
)

CLASSES = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']
CLASS_EMOJIS = {
    'buildings': '🏢',
    'forest':    '🌲',
    'glacier':   '🧊',
    'mountain':  '⛰️',
    'sea':        '🌊',
    'street':    '🛣️',
}
MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')

IMG_SIZE_SCRATCH = 150
IMG_SIZE_TL      = 224


# ── Model loading (cached) ────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    import tensorflow as tf
    models_loaded = {}
    scratch_path = os.path.join(MODELS_DIR, 'cnn_scratch_final.keras')
    tl_path      = os.path.join(MODELS_DIR, 'tl_mobilenetv2_final.keras')

    if os.path.exists(scratch_path):
        models_loaded['scratch'] = tf.keras.models.load_model(scratch_path)
    if os.path.exists(tl_path):
        models_loaded['tl'] = tf.keras.models.load_model(tl_path)
    return models_loaded


def preprocess_image_scratch(img: Image.Image):
    """Resize + normalize to [0,1] for CNN scratch model."""
    img = img.convert('RGB').resize((IMG_SIZE_SCRATCH, IMG_SIZE_SCRATCH))
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)


def preprocess_image_tl(img: Image.Image):
    """Resize + MobileNetV2 preprocess ([-1,1]) for TL model."""
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
    img = img.convert('RGB').resize((IMG_SIZE_TL, IMG_SIZE_TL))
    arr = np.array(img, dtype=np.float32)
    arr = preprocess_input(arr)
    return np.expand_dims(arr, axis=0)


def make_confidence_chart(probs, title):
    fig, ax = plt.subplots(figsize=(5, 3.5))
    colors = ['#4E79A7' if p != max(probs) else '#E15759' for p in probs]
    bars = ax.barh(CLASSES, probs, color=colors)
    ax.set_xlim(0, 1)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_xlabel('Confidence')
    for bar, val in zip(bars, probs):
        ax.text(val + 0.01, bar.get_y() + bar.get_height()/2,
                f'{val:.1%}', va='center', fontsize=9)
    plt.tight_layout()
    return fig


# ── UI ─────────────────────────────────────────────────────────────────────────
st.title('🏔️ Intel Scene Classifier')
st.markdown(
    'Upload a scene image — the app runs it through **two trained models** '
    '(CNN from scratch + MobileNetV2 Transfer Learning) and compares predictions.'
)

models = load_models()
if not models:
    st.error('⚠️ No trained models found in `models/` directory. '
             'Please run the training notebooks first.')
    st.stop()

# Sidebar info
with st.sidebar:
    st.header('ℹ️ About')
    st.write('**Dataset:** Intel Image Classification (Kaggle)')
    st.write('**Classes:** 6 natural scene categories')
    st.write('**Models loaded:**')
    for key, label in [('scratch', 'CNN from Scratch'), ('tl', 'MobileNetV2 TL')]:
        icon = '✅' if key in models else '❌'
        st.write(f'  {icon} {label}')

    # Show comparison metrics if available
    scratch_path = os.path.join(RESULTS_DIR, 'scratch_metrics.json')
    tl_path      = os.path.join(RESULTS_DIR, 'tl_metrics.json')
    if os.path.exists(scratch_path) and os.path.exists(tl_path):
        st.subheader('📊 Test Set Metrics')
        with open(scratch_path) as f: sm = json.load(f)
        with open(tl_path)      as f: tm = json.load(f)
        st.write(f'**CNN Scratch:** {sm["accuracy"]*100:.1f}% accuracy')
        st.write(f'**MobileNetV2:** {tm["accuracy"]*100:.1f}% accuracy')

# Main upload area
uploaded = st.file_uploader(
    'Choose an image (JPG / PNG)',
    type=['jpg', 'jpeg', 'png']
)

if uploaded:
    image = Image.open(uploaded)

    col1, col2 = st.columns([1, 2])

    with col1:
        st.image(image, caption='Uploaded Image', use_column_width=True)
        st.write(f'**Size:** {image.size[0]} × {image.size[1]} px')

    with col2:
        st.subheader('Predictions')

        tab1, tab2 = st.tabs(['🧠 CNN from Scratch', '🚀 Transfer Learning'])

        with tab1:
            if 'scratch' in models:
                arr   = preprocess_image_scratch(image)
                probs = models['scratch'].predict(arr, verbose=0)[0]
                pred  = CLASSES[np.argmax(probs)]
                conf  = float(np.max(probs))

                st.metric('Predicted Class', f'{CLASS_EMOJIS[pred]}  {pred.capitalize()}')
                st.metric('Confidence', f'{conf:.1%}')
                st.pyplot(make_confidence_chart(probs.tolist(), 'CNN Scratch — Class Probabilities'))
            else:
                st.warning('CNN Scratch model not found.')

        with tab2:
            if 'tl' in models:
                arr   = preprocess_image_tl(image)
                probs = models['tl'].predict(arr, verbose=0)[0]
                pred  = CLASSES[np.argmax(probs)]
                conf  = float(np.max(probs))

                st.metric('Predicted Class', f'{CLASS_EMOJIS[pred]}  {pred.capitalize()}')
                st.metric('Confidence', f'{conf:.1%}')
                st.pyplot(make_confidence_chart(probs.tolist(), 'MobileNetV2 TL — Class Probabilities'))
            else:
                st.warning('Transfer Learning model not found.')

    # Show model agreement
    if 'scratch' in models and 'tl' in models:
        p_scratch = CLASSES[np.argmax(models['scratch'].predict(
            preprocess_image_scratch(image), verbose=0)[0])]
        p_tl      = CLASSES[np.argmax(models['tl'].predict(
            preprocess_image_tl(image), verbose=0)[0])]

        if p_scratch == p_tl:
            st.success(f'✅ Both models agree: **{p_scratch.capitalize()}**')
        else:
            st.warning(
                f'⚠️ Models disagree — Scratch: **{p_scratch}** | TL: **{p_tl}**'
            )

else:
    # Show example classes
    st.info('👆 Upload any outdoor/scene photo — buildings, forest, mountain, sea, street, or glacier.')

    st.subheader('Supported Scene Classes')
    cols = st.columns(6)
    for col, cls in zip(cols, CLASSES):
        with col:
            st.markdown(f'### {CLASS_EMOJIS[cls]}\n**{cls.capitalize()}**')
