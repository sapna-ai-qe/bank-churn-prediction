# 🏦 Bank Customer Churn Prediction
### Neural Network | ANN | TensorFlow/Keras | Deep Learning

![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow)
![Keras](https://img.shields.io/badge/Keras-2.x-red?logo=keras)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)
![Domain](https://img.shields.io/badge/Domain-Banking%20%7C%20Finance-blue)

---

## 🏦 Business Context

Banks must constantly battle **customer churn** — the phenomenon of customers leaving for competitor service providers. Identifying customers likely to churn **before they leave** allows management to intervene with targeted retention strategies, saving both revenue and acquisition cost.

> It costs **5–7x more** to acquire a new customer than to retain an existing one.

---

## 🎯 Objective

> Build a **neural network-based classifier** that predicts whether a customer will leave the bank within the **next 6 months**, enabling proactive retention strategies.

---

## 📊 Dataset

| Property | Value |
|---|---|
| Rows | 10,000 customers |
| Columns | 14 features |
| Target Variable | `Exited` (0 = Stayed, 1 = Churned) |
| Missing Values | None |
| Class Imbalance | ~20% churn rate |

**Key Features:**
- `CreditScore` — Customer's credit history score
- `Geography` — France, Spain, Germany
- `Gender` — Male / Female
- `Age` — Customer age
- `Tenure` — Years with the bank
- `Balance` — Account balance
- `NumOfProducts` — Number of bank products held
- `HasCrCard` — Has credit card (Yes/No)
- `IsActiveMember` — Active membership status
- `EstimatedSalary` — Estimated annual salary

---

## 🔬 Approach & Methodology

```
Raw Data → EDA → Preprocessing → ANN Design → Training → Regularization → Evaluation → Insights
```

### 1. Exploratory Data Analysis (EDA)
- Analyzed churn distribution across geography, gender, age, and product usage
- Key finding: **Germany customers churn at significantly higher rates** than France or Spain
- Identified age as a strong predictor — **middle-aged customers (40-50) churn more**
- Correlation analysis between all features and churn target

### 2. Data Preprocessing
- Label encoding for binary categorical variables (`Gender`, `HasCrCard`)
- One-hot encoding for multi-class categorical variable (`Geography`)
- Feature scaling using StandardScaler
- Train/test split: 80% train, 20% test
- Addressed class imbalance using class weights

### 3. Neural Network Architecture (ANN)
```
Input Layer  → 11 features
Dense Layer  → 64 neurons, ReLU activation
Dropout      → 0.3 (prevent overfitting)
Dense Layer  → 32 neurons, ReLU activation
Dropout      → 0.3
Dense Layer  → 16 neurons, ReLU activation
Output Layer → 1 neuron, Sigmoid activation (binary classification)
```

### 4. Training Configuration
- **Optimizer:** Adam
- **Loss Function:** Binary Crossentropy
- **Metrics:** Accuracy, Recall, AUC
- **Regularization:** L2 regularization + Dropout layers
- **Early Stopping:** Monitored validation loss to prevent overfitting
- **Epochs:** Up to 100 with early stopping

### 5. Model Evaluation
- Confusion Matrix analysis
- Classification report (Precision, Recall, F1-Score)
- ROC-AUC curve
- Focus on **Recall for class 1 (churned)** — catching actual churners is the business priority
- Test Recall for churn class: **~75%** (model correctly identifies 75% of customers who will churn)

---

## 📈 Key Results

| Metric | Value |
|---|---|
| Model | Optimized ANN with Dropout + L2 Regularization |
| Recall (Churn Class) | ~75% on test data |
| Key Insight | Germany customers and age 40-50 are highest churn risk |

---

## 💡 Business Insights & Recommendations

1. **Germany Customer Focus** — Germany shows disproportionately high churn; investigate local competitive factors and service quality
2. **Age-Based Retention** — Mid-age customers (40-50) are high-risk; create loyalty programs targeting this segment
3. **Single Product Customers** — Customers with only 1 product are far more likely to churn; cross-sell as a retention strategy
4. **Inactive Members** — Non-active members show very high churn; trigger re-engagement campaigns for inactive users
5. **Balance Monitoring** — Customers with very high or zero balances both show higher churn; tailor interventions accordingly

---

## 🛠 Tech Stack

| Tool | Purpose |
|---|---|
| Python 3.8+ | Core programming language |
| TensorFlow 2.x | Deep learning framework |
| Keras | High-level neural network API |
| Pandas | Data manipulation |
| NumPy | Numerical computation |
| Scikit-learn | Preprocessing and evaluation metrics |
| Matplotlib / Seaborn | Visualization |
| Google Colab | GPU-accelerated development environment |

---

## 📁 Project Structure

```
bank-churn-prediction/
│
├── Sapna_Bank_Churn_Full_code___.ipynb   # Full notebook with EDA, model, evaluation
├── README.md                              # Project documentation
└── data/                                  # Dataset (not included - proprietary)
```

---

## 🚀 How to Run

1. Clone this repository
2. Open the notebook in Google Colab (recommended for GPU acceleration)
3. Upload dataset to Google Drive and update file path
4. Run all cells sequentially

```bash
pip install tensorflow keras pandas numpy scikit-learn matplotlib seaborn
```

---

## 👩‍💻 Author

**Sapna** | Senior AI Quality Engineer  
Post Graduate in AI/ML — University of Texas at Austin  
GitHub: [@sapna-ai-qe](https://github.com/sapna-ai-qe)

---
*Part of AI/ML Portfolio — UT Austin Post Graduate Program*
