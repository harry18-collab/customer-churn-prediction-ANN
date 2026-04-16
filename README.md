Customer Churn Prediction using ANN

A deep learning project that predicts whether a bank customer will churn (leave the bank) using an Artificial Neural Network (ANN) built with TensorFlow/Keras. The project includes a Streamlit web app for real-time predictions.

Project Overview
Customer churn prediction helps banks identify customers who are likely to leave. By predicting churn early, banks can take proactive steps to retain valuable customers.

Tech Stack

Python 3.11
TensorFlow / Keras — ANN model
Scikit-learn — preprocessing (LabelEncoder, OneHotEncoder, StandardScaler)
Pandas / NumPy — data manipulation
Streamlit — web app
Pickle — saving/loading encoders and scaler


Project Structure
customer-churn-prediction-ANN/
│
├── app.py                        # Streamlit web app
├── experiments.ipynb             # Model training notebook
├── model.h5                      # Trained ANN model
├── Label_encoder_gender.pkl      # Saved LabelEncoder for Gender
├── OneHotEncoder_geo.pkl         # Saved OneHotEncoder for Geography
├── StandardScaler.pkl            # Saved StandardScaler
├── requirements.txt              # Project dependencies
├── runtime.txt                   # Python version for deployment
└── README.md

Model Architecture
LayerNeuronsActivationInput + Hidden Layer 164ReLUHidden Layer 232ReLUOutput Layer1Sigmoid

Optimizer: Adam (learning rate = 0.01)
Loss Function: Binary Crossentropy
Metrics: Accuracy


Features Used
FeatureDescriptionCreditScoreCustomer's credit scoreGeographyCountry (France, Germany, Spain)GenderMale / FemaleAgeCustomer's ageTenureYears with the bankBalanceAccount balanceNumOfProductsNumber of bank products usedHasCrCardHas credit card (0/1)IsActiveMemberIs active member (0/1)EstimatedSalaryEstimated annual salary

How to Run Locally

Clone the repository

bashgit clone https://github.com/harry18-collab/customer-churn-prediction-ANN.git
cd customer-churn-prediction-ANN

Install dependencies

bashpip install -r requirements.txt

Run the Streamlit app

bashstreamlit run app.py

How It Works

User inputs customer details in the web app
Gender is encoded using LabelEncoder
Geography is encoded using OneHotEncoder
All features are scaled using StandardScaler
The ANN model predicts churn probability
If probability > 0.5 → Customer likely to churn


Results
The model achieves good accuracy on the test set and is monitored using TensorBoard during training.

Author
harry18-collab
GitHub
