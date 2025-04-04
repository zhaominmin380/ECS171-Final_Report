# 🚗 Vehicle Rating Prediction Using Machine Learning

This project aims to predict the acceptability of cars based on various categorical features using machine learning models. The main model used in production is a **Random Forest Classifier**, chosen for its high accuracy, interpretability, and robustness to imbalanced data.

## 🧠 Models Used

- **Random Forest (main model)**
- Logistic Regression
- Decision Tree
- Artificial Neural Network (ANN)

> ✅ Final model achieved **97.8% cross-validation accuracy** and demonstrated **excellent generalizability**.

---

## 📂 Project Structure

```
.
├── app.py                  # Flask API for web deployment
├── model.py                # Data preprocessing, training, prediction logic
├── rf_model.pkl            # Trained Random Forest model
├── car.data                # UCI Car Evaluation Dataset
├── templates/
│   └── index.html          # Web interface (assumed present)
└── Final_Report.ipynb      # Full report and analysis notebook
```

---

## 📊 Dataset Description

- **Source**: UCI Machine Learning Repository  
- **Features** (all categorical):  
  - `buying`, `maint`, `doors`, `persons`, `lug_boot`, `safety`  
- **Target label**: `class` (one of `unacc`, `acc`, `good`, `vgood`)

In production, the model uses all features **except** `doors` due to its low feature importance.

---

## 🧪 How to Run

### 🔧 Install dependencies

```bash
pip install flask scikit-learn pandas numpy joblib
```

### ▶️ Run the app

```bash
python app.py
```

Open your browser at: [http://localhost:5000](http://localhost:5000)

---

## 📦 API Usage

**Endpoint:** `POST /predict`  
**Example Payload (JSON):**

```json
{
  "buying": 1,
  "maint": 2,
  "persons": 2,
  "lug_boot": 1,
  "safety": 2
}
```

**Response:**

```
"My suggestion: good"
```

---

## 📈 Model Training Summary

- **Preprocessing**: Categorical → numeric mapping  
- **Feature Selection**: Chi-Square, Feature Importance  
- **Best Model**: Random Forest (excluding unimportant features)  
- **Accuracy**: 97.69%  
- **Cross-validation mean score**: 97.8%

---

## 📚 Report & Demo

- 📄 [Full Report (PDF)](./IEEE_Conference_Template.pdf)  
- 📺 [Project Demo (YouTube)](https://youtu.be/NIRyB936b78)  
- 🔗 [GitHub Repository](https://github.com/zhaominmin380/ECS171-Final_Report)

---

## 🙌 Authors

- **Chen-Yu Fan** – UC Davis  
- **Yufan Chen** – UC Davis  
- **Chengyuan Liu** – UC Davis  
- **Kyle Fong** – UC Davis  
- **Riyan Townsley** – UC Davis
