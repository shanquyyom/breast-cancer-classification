# 🩺 Breast Cancer Classification – ML Project with FastAPI Deployment

This project aims to detect whether a tumor is malignant or benign using various machine learning models applied to the Breast Cancer Wisconsin Dataset. It also includes a fully functional FastAPI deployment for model inference.

---

## 📊 Project Overview

- Dataset: **Breast Cancer Wisconsin Dataset** (from `sklearn.datasets`)
- Models Applied:
  - Logistic Regression ✅
  - K-Nearest Neighbors (KNN) ✅
  - Decision Tree ✅
  - Random Forest ✅
  - Gradient Boosting ✅
  - AdaBoost, Bagging, and Voting Classifiers ✅
  - **PyCaret AutoML** for final selection ✅
- Final Model: **Random Forest Classifier** (chosen via performance comparison)
- Accuracy: **~97%**

---

## 📁 Project Structure

breast-cancer-classification/
│
├── models_comparison.ipynb # Colab notebook with model training and comparison
├── breast_cancer_api/
│ ├── main.py # FastAPI app for deployment
│ ├── requirements.txt # Required libraries
│ └── model/
│ ├── breast_cancer_model.pkl # Final trained ML model
│ └── scaler.pkl # Feature scaler
└── README.md # Project overview (this file)


---

## 🚀 Deployment

The model is deployed using **FastAPI**, allowing real-time predictions via POST requests.

### ✅ Example POST Request:

`URL:` `http://localhost:8000/predict`  
`Method:` `POST`  
`Body:`

```json
{
  "mean_radius": 14.2,
  "mean_texture": 20.1,
  "mean_perimeter": 90.2,
  "mean_area": 600.3,
  "mean_smoothness": 0.09
}

💡 Tools & Libraries Used
Python 🐍

Scikit-learn

Pandas, NumPy, Matplotlib

PyCaret (for automated ML)

FastAPI (for deployment)

Uvicorn (as ASGI server)

🙋‍♂️ Author
Shan Quyyoom
📍 Jhang, Pakistan
📧 shanquyyoom99@gmail.com
🔗 https://www.linkedin.com/in/shan-quyyoom-452923365
