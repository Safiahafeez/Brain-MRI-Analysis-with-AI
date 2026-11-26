# 🧠 Brain Tumor Detection & Educational Report Generation

An end-to-end Artificial Intelligence system that detects brain tumors from MRI scans using a Convolutional Neural Network (CNN) and generates patient-friendly educational reports using Large Language Models (LLMs).

## 💾 Datasets & Model Download   "Source: Kaggle Brain MRI Dataset"

Due to GitHub file size limits, the heavy datasets and the trained model are hosted on Google Drive. 

*   **Training Dataset:** 👉 **[Download Training Dataset here](https://drive.google.com/drive/folders/17pOuXf_xSqGWimrlDWCFxo40a6aHQ2oQ?usp=drive_link)**
*   **Testing Dataset:** 👉 **[Download Testing Dataset here](https://drive.google.com/drive/folders/1k6yYc1ZMJMis-9h7TJfCWI_o01RH4Cbr?usp=sharing)**

**Trained Model:**
*   **`brain_tumor_model1.h5`:** 👉 **[Download Model here](https://drive.google.com/file/d/17QaoQK-eUoRuUtJyvzzVYWMCou1dDY1J/view?usp=sharing)**

## 🚀 Key Features
- **High-Accuracy CNN:** 99.83% validation accuracy on MRI classification.
- **LLM Integration:** Uses Groq (Llama 3) and Google Gemini to generate structured medical reports based on CNN predictions and patient symptoms.
- **Secure Architecture:** Built with FastAPI to securely handle API keys, keeping them hidden from the frontend.
- **Interactive UI:** Streamlit frontend featuring:
  - Single Image & Symptom Analysis
  - Batch Prediction (CSV Export)
  - Live Metric Evaluation (Upload ZIP to get a live Confusion Matrix/ROC)
  - AI Chatbot Assistant

## 🛠️ Technologies Used
- **Deep Learning:** TensorFlow / Keras, NumPy
- **Generative AI:** Groq API, Google Gemini (Few-Shot Prompting)
- **Web Frameworks:** Streamlit (Frontend), FastAPI (Backend)
- **Evaluation:** Scikit-learn, Matplotlib, Seaborn

## 📊 Results
- **Accuracy:** 99.83%
- **ROC-AUC:** 1.0000
- *(See the generated graphs in this repository for Accuracy/Loss, Confusion Matrix, and Classification Report).*

## ⚙️ How to Run Locally

## ⚙️ How to Run Locally

1. **Download the Model:** Download `brain_tumor_model1.h5` from the Google Drive link above and place it in the project root folder.
2. **Add API Keys:** Create a `.env` file in the root folder and add:
   `GROQ_API_KEY=your_key_here`
3. **Install Requirements:** Open terminal and run: `pip install -r requirements.txt`
4. **Run Backend (Terminal 1):** Run: `uvicorn backend:app --reload`
5. **Run Frontend (Terminal 2):** Open a new terminal and run: `streamlit run frontend.py`

*(Windows users can just double-click the `run_project.bat` file to do steps 4 and 5 automatically).*

### 1. Clone the repository
```bash
git clone https://github.com/YourUsername/brain-tumor-ai.git
cd brain-tumor-ai