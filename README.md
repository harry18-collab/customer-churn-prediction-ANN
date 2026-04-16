#Customer Churn Prediction using ANN

A deep learning project that predicts whether a bank customer will churn (leave the bank) using an Artificial Neural Network (ANN) built with TensorFlow/Keras.
The project also includes a Streamlit web app for real-time predictions.

---

##Project Overview

Customer churn prediction helps banks identify customers who are likely to leave.
By predicting churn early, banks can take proactive steps to retain valuable customers.

---

##Tech Stack

* **Python 3.11**
* **TensorFlow / Keras** — ANN model
* **Scikit-learn** — preprocessing
* **Pandas / NumPy** — data manipulation
* **Streamlit** — web app
* **Pickle** — saving encoders and scaler

---

##Project Structure

```
customer-churn-prediction-ANN/
│
├── app.py
├── experiments.ipynb
├── model.h5
├── Label_encoder_gender.pkl
├── OneHotEncoder_geo.pkl
├── StandardScaler.pkl
├── requirements.txt
├── runtime.txt
└── README.md
```

---

##Model Architecture

* **Input Layer + Hidden Layer 1** → 64 neurons (ReLU)

* **Hidden Layer 2** → 32 neurons (ReLU)

* **Output Layer** → 1 neuron (Sigmoid)

* **Optimizer:** Adam (learning rate = 0.01)

* **Loss Function:** Binary Crossentropy

* **Metrics:** Accuracy

---

##Features Used

* CreditScore
* Geography
* Gender
* Age
* Tenure
* Balance
* NumOfProducts
* HasCrCard
* IsActiveMember
* EstimatedSalary

---

##How to Run Locally

### 1. Clone the repository

```
git clone https://github.com/harry18-collab/customer-churn-prediction-ANN.git
cd customer-churn-prediction-ANN
```

### 2. Install dependencies

```
pip install -r requirements.txt
```

### 3. Run the Streamlit app

```
streamlit run app.py
```

---

##How It Works

1. User inputs customer details in the web app
2. Gender → LabelEncoder
3. Geography → OneHotEncoder
4. Features → StandardScaler
5. Model predicts churn probability
6. If probability > 0.5 → Customer likely to churn

---

##Results

* Achieves good accuracy on test data
* Model training monitored using TensorBoard

---
