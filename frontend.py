# frontend.py
import streamlit as st
import requests
import numpy as np
from PIL import Image
import tensorflow as tf
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import zipfile
import tempfile
import pathlib
from groq import Groq
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
GROQ_KEY = os.getenv("GROQ_API_KEY")
groq_client = Groq(api_key=GROQ_KEY)
FASTAPI_URL = "http://127.0.0.1:8000"
CLASS_NAMES = ["Non-Tumor", "Tumor"]

st.set_page_config(page_title="Brain Tumor AI", layout="wide")
st.title("🧠 Brain Tumor AI System")

# --- LOAD CNN MODEL ---
@st.cache_resource
def load_cnn_model():
    # FIXED: Changed to brain_tumor_model1.h5 to match your actual file
    model_path = "brain_tumor_model.h5" 
    if os.path.exists(model_path):
        return tf.keras.models.load_model(model_path)
    else:
        st.error(f"❌ Model file not found: {model_path}.")
        return None

model = load_cnn_model()

# --- HELPER FUNCTION ---
def preprocess_image(img: Image.Image) -> np.ndarray:
    img = img.convert("RGB").resize((128, 128))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)

# --- CREATE TABS ---
tab1, tab2, tab3, tab4 = st.tabs([
    "🔍 Single Image", 
    "📁 Batch Prediction", 
    "📊 Model Metrics", 
    "💬 AI Assistant"
])

# ==================================================================
# TAB 1: SINGLE IMAGE (CNN + LLM Report)
# ==================================================================
with tab1:
    st.header("Single Image Prediction & Report")
    col1, col2 = st.columns(2)

    with col1:
        symptoms = st.text_area("Patient Symptoms:", "I have severe headaches and blurred vision.")
        uploaded_file = st.file_uploader("Upload MRI (JPG/PNG)", type=["jpg", "png", "jpeg"])
        if uploaded_file:
            st.image(uploaded_file, caption="Uploaded MRI", use_container_width=True)

    with col2:
        if st.button("Run Diagnosis & Generate Report", type="primary"):
            if not uploaded_file:
                st.warning("Please upload an MRI image first.")
            elif model is None:
                st.error("Model not loaded.")
            else:
                with st.spinner("1. Running CNN Model..."):
                    img = Image.open(uploaded_file)
                    arr = preprocess_image(img)
                    prob = float(model.predict(arr, verbose=0)[0][0])
                    label = CLASS_NAMES[1] if prob > 0.5 else CLASS_NAMES[0]
                    conf = prob if prob > 0.5 else (1 - prob)
                    st.success(f"**CNN Prediction:** {label} ({conf*100:.2f}%)")
                
                with st.spinner("2. LLM generating educational report..."):
                    payload = {"prediction": label, "confidence": conf, "symptoms": symptoms}
                    try:
                        response = requests.post(f"{FASTAPI_URL}/analyze", json=payload)
                        if response.status_code == 200:
                            st.info(response.json()["report"])
                        else:
                            st.error(f"Backend Error: {response.text}")
                    except requests.exceptions.ConnectionError:
                        st.error("❌ Backend is not running. Please run `uvicorn backend:app --reload`")

# ==================================================================
# TAB 2: BATCH PREDICTION
# ==================================================================
with tab2:
    st.header("Batch Prediction")
    st.write("Upload multiple MRI images to predict all at once.")
    
    if "batch_uploader_key" not in st.session_state:
        st.session_state.batch_uploader_key = "uploader_1"

    batch_files = st.file_uploader(
        "Upload multiple MRIs", 
        type=["jpg", "png", "jpeg"], 
        accept_multiple_files=True,
        key=st.session_state.batch_uploader_key
    )
    
    col_btn1, col_btn2 = st.columns(2)
    
    with col_btn1:
        predict_btn = st.button("Predict Batch", type="primary")
        
    with col_btn2:
        if st.button("Clear All Uploads & Results"):
            st.session_state.batch_uploader_key = f"uploader_{np.random.randint(1000, 9999)}"
            if "batch_results_df" in st.session_state:
                del st.session_state.batch_results_df
            st.rerun()

    if predict_btn and batch_files:
        if model is None:
            st.error("Model not loaded.")
        else:
            results = []
            progress = st.progress(0, text=f"Processing {len(batch_files)} images...")
            
            for i, file in enumerate(batch_files):
                img = Image.open(file)
                arr = preprocess_image(img)
                prob = float(model.predict(arr, verbose=0)[0][0])
                label = CLASS_NAMES[1] if prob > 0.5 else CLASS_NAMES[0]
                
                results.append({
                    "Filename": file.name,
                    "Prediction": label,
                    "Confidence": f"{max(prob, 1-prob)*100:.2f}%"
                })
                progress.progress((i + 1) / len(batch_files), text=f"Processed {i+1}/{len(batch_files)}")
            
            st.session_state.batch_results_df = pd.DataFrame(results)

    if "batch_results_df" in st.session_state:
        st.dataframe(st.session_state.batch_results_df, use_container_width=True)
        
        st.download_button(
            label="📥 Download Results as CSV", 
            data=st.session_state.batch_results_df.to_csv(index=False).encode('utf-8'), 
            file_name="batch_results.csv", 
            mime="text/csv"
        )

# ==================================================================
# TAB 3: MODEL METRICS (All 4 Images + Live Evaluation)
# ==================================================================
with tab3:
    st.header("Model Metrics Evaluation")
    
    # Create three sub-tabs
    metric_tab1, metric_tab2, metric_tab3 = st.tabs([
        "📈 Training Graphs", 
        "🧮 Classification Results", 
        "🧪 Live Test on New Images"
    ])
    
    # --- SUB-TAB 1: ACCURACY & LOSS GRAPH ---
    with metric_tab1:
        st.write("This graph shows the model's learning progress over 10 epochs.")
        # FIXED: Changed to accuracy_loss.png
        if os.path.exists("accuracy_loss.png"):
            st.image("accuracy_loss.png", caption="Training and Validation Accuracy & Loss", use_container_width=True)
        else:
            st.warning("💡 Please save `accuracy_loss.png` into this folder to display it here.")
            
        st.success("**Final Validation Accuracy:** 99.83% | **Validation Loss:** 0.0093")

    # --- SUB-TAB 2: CONFUSION MATRIX, ROC & CLASSIFICATION REPORT ---
    with metric_tab2:
        st.write("Final evaluation metrics generated from the validation set during training.")
        
        # Row 1: Confusion Matrix and ROC Curve
        col_img1, col_img2 = st.columns(2)
        with col_img1:
            # FIXED: Changed to confusion_matrix.png
            if os.path.exists("confusion_matrix.png"):
                st.image("confusion_matrix.png", caption="Confusion Matrix (Validation Set)", use_container_width=True)
            else:
                st.info("💡 Place `confusion_matrix.png` in your project folder.")
                
        with col_img2:
            if os.path.exists("roc_auc.png"):
                st.image("roc_auc.png", caption="ROC Curve (Validation Set)", use_container_width=True)
            else:
                st.info("💡 Place `roc_auc.png` in your project folder.")

        # Row 2: Classification Report (Results)
        st.divider()
        st.subheader("Classification Report")
        if os.path.exists("results.png"):
            st.image("results.png", caption="Precision, Recall, and F1-Score", width=600)
        else:
            st.info("💡 Place `results.png` in your project folder.")

    # --- SUB-TAB 3: LIVE ZIP EVALUATION ---
    with metric_tab3:
        st.write("Upload a **ZIP file** containing two folders (`tumor/` and `non_tumor/`) to test the model on new data right now.")
        
        zip_file = st.file_uploader("Upload Test Dataset (.zip)", type=["zip"])
        
        if zip_file is not None and st.button("Calculate Live Metrics"):
            if model is None:
                st.error("Model not loaded.")
            else:
                with st.spinner("Extracting ZIP and running predictions..."):
                    tmpdir = tempfile.mkdtemp()
                    with zipfile.ZipFile(zip_file) as z:
                        z.extractall(tmpdir)
                    
                    root = pathlib.Path(tmpdir)
                    label_map = {}
                    for d in root.rglob("*"):
                        if d.is_dir():
                            name = d.name.lower()
                            if "non" in name or "no" in name: label_map[d] = 0
                            elif "tumor" in name: label_map[d] = 1
                    
                    y_true, y_prob = [], []
                    total_images = sum(len([p for p in d.iterdir() if p.is_file()]) for d in label_map)
                    st.info(f"Found {total_images} images. Running predictions...")
                    
                    for folder, lbl in label_map.items():
                        for img_path in folder.iterdir():
                            if img_path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                                img = Image.open(img_path)
                                arr = preprocess_image(img)
                                p = float(model.predict(arr, verbose=0)[0][0])
                                y_true.append(lbl)
                                y_prob.append(p)
                    
                    y_true = np.array(y_true)
                    y_prob = np.array(y_prob)
                    y_pred = (y_prob > 0.5).astype(int)
                    
                    # Confusion Matrix
                    cm = confusion_matrix(y_true, y_pred)
                    fig, ax = plt.subplots(figsize=(4, 3))
                    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
                    st.pyplot(fig)
                    
                    # ROC Curve
                    fpr, tpr, _ = roc_curve(y_true, y_prob)
                    roc_auc = auc(fpr, tpr)
                    fig2, ax2 = plt.subplots(figsize=(4, 3))
                    ax2.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.4f}')
                    ax2.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                    ax2.legend(loc="lower right")
                    st.pyplot(fig2)
                    
                    st.text("Classification Report:")
                    st.text(classification_report(y_true, y_pred, target_names=CLASS_NAMES))

# ==================================================================
# TAB 4: AI ASSISTANT CHATBOT
# ==================================================================
with tab4:
    st.header("💬 Project AI Assistant")
    st.write("Ask questions about the project, the CNN architecture, or how to interpret results.")
    
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "system", "content": "You are an AI assistant for a Brain Tumor project. The CNN achieves 99.8% accuracy on MRI images. Keep answers educational. Never give medical advice."}
        ]

    for msg in st.session_state.messages:
        if msg["role"] != "system":
            with st.chat_message(msg["role"]):
                st.write(msg["content"])

    if prompt := st.chat_input("Ask a question..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = groq_client.chat.completions.create(
                        messages=st.session_state.messages,
                        model="openai/gpt-oss-20b",
                        temperature=0.5
                    )
                    reply = response.choices[0].message.content
                    st.write(reply)
                    st.session_state.messages.append({"role": "assistant", "content": reply})
                except Exception as e:
                    st.error(f"Error: {e}")

    st.divider()
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = [
            {"role": "system", "content": "You are an AI assistant for a Brain Tumor project. The CNN achieves 99.8% accuracy on MRI images. Keep answers educational. Never give medical advice."}
        ]
        st.rerun()