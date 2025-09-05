# **Assistive Loan Reviewer** 📊

**Streamlit App:** [Assistive Loan Reviewer](https://assistive-loan-reviewer.streamlit.app/)  

**Blog Post:** [Loan Default Prediction](https://www.tumblr.com/dheedataanalysis/793837058160525312/loan-default-prediction)


<br>

## 📄 **Overview**

The Assistive Loan Reviewer is a machine learning-powered classification system that predicts whether a loan applicant is likely to default. Based on these predictions, the system automatically approves low-risk applications and flags borderline or high-risk applications for human review. This hybrid machine-assisted approach accelerates loan processing, reduces the workload for loan officers, and ensures financial risk is carefully managed.

<br>

## ✨ **Features**
- Predicts whether a loan applicant is likely to default.
- Uses a voting ensemble of machine learning models (Logistic Regression, Random Forest, XGBoost, LightGBM, CatBoost) to improve prediction stability.
- Highlights risky applications for human review and auto-approval for non default customer applications.
- Supports new and returning customers.
- Provides model confidence scores alongside predictions.
- Built with Streamlit for easy deployment and user interaction.
- Features


<br>

## ⚙️ **Technical Approach**

**1. Data & Feature Engineering**
- Uses loan applicant data including financial metrics, repayment behavior, and demographics.
- Constructs engineered features like loan growth trends, repayment delays, debt-to-loan ratio, and risk levels.

**2. Modeling**
- Ensemble of classifiers: Logistic Regression, Random Forest, XGBoost, LightGBM, and CatBoost.
- Voting ensemble combines predictions for stable results.
- Decision thresholds tuned to handle class imbalance in loan defaults.

**3. Prediction Workflow**
- Low-risk applicants → auto-approved.
- High-risk or borderline applicants → flagged for human review.
- Returns predicted label and confidence score.

**4. Deployment**
- Streamlit app allows both customers and admins to interact with the system.
- Models are loaded with joblib and cloudpickle for compatibility.

<br>

## 📁 **Project Folder Structure**
```bash
Loan Default Prediction/
├─ Powerbi/
│  ├─ Loan_Prediction_Analysis.pbix
│  ├─ Loan_Prediction_Analysis.pdf
│  ├─ ng.json
├─ Python/
│  ├─ gps_shp_docs
│  ├─ streamlit/
│  │  ├─ models/
│  │  │  ├─ CatBoost.pkl
│  │  │  ├─ LightGBM.pkl
│  │  │  ├─ Logistic_Regression.pkl
│  │  │  ├─ Random_Forest.pkl
│  │  │  ├─ XGBoost.pkl
│  │  │  ├─ Vote.pkl
│  │  │  ├─ custom_transformer.pkl
│  │  │  ├─ risk_cluster.pkl
│  │  │  ├─ preprocessor.pkl
│  │  ├─ Loan_Default_App.py
│  │  ├─ customer_backend.py
│  │  ├─ customer_data.csv
│  ├─ Loan_default_prediction.ipynb
│  ├─ final_model.ipynb
│  ├─ model_data.csv
├─ pro_data_1.csv
├─ pro_data_2.csv
├─ pro_data_3.csv
├─ requirements.txt
```

<br>

## 🛠️ **Installation & Setup**

```bash
# Clone repo
git clone https://github.com/Ola-doyin/Loan-Default-Predictor.git
cd Loan-Default-Predictor

# Create and activate virtual environment
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/macOS
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run Streamlit app
streamlit run Python/streamlit/Loan_Default_App.py
```
Make sure the streamlit folder contains Loan_Default_App.py, customer_backend.py, customer_data.csv, and the models/ folder with all .pkl files.

<br>

## 🚀 **Future Improvements**

- Enhanced data collection for richer financial and behavioral patterns (income, payment frequency, marital status, etc.).
- Expanded feature engineering, especially for deep learning models.
- Integration with real-time transaction data for better predictions.

<br>

## 📜 **License** 

This project is part of my **Axia Africa Data Science Capstone** submission. All code and materials are provided for educational and demonstration purposes.


<br>

## Built by:
**Oladoyin Arewa - [GitHub](https://github.com/Ola-doyin)**

Electronic & Electrical Engineer | AI Engineer | Power Systems Researcher

