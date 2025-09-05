# Returning_customer.py
import pandas as pd
import numpy as np
import joblib
from models.custom_transformers import RiskClusterTransformer
import cloudpickle

# Paths to your saved models
RISK_CLUSTER_PATH = "models/risk_cluster.pkl"
PREPROCESSOR_PATH = "models/preprocessor.pkl"
VOTING_MODEL_PATH = "models/Vote.pkl"

# Internal caches
_preprocessor = None
_risk_cluster = None
_voting_model = None


features = ['num__prct_duedate_default', 'num__loan_growth_trend', 'num__credit_score', 'num__debt_to_loan_ratio',
            'cat__bank_name_clients', 'num__max_repay_delay', 'num__totaldue', 'cat__bank_account_type',
            'cat__state', 'num__avg_amount', 'num__last_loan_amount', 'num__avg_payout_time', 'num__loanamount',
            'num__age', 'cat__employment_status_clients', 'num__loannumber', 'num__avg_total_due', 'num__avg_term_days',
            'num__avg_loan_gap', 'num__std_amount', 'risk_level']




def get_models():
    """
    Lazy-load and cache the preprocessor, risk cluster, and voting model.
    Prevents Streamlit import issues with pickle.
    """
    global _preprocessor, _risk_cluster, _voting_model
    if _preprocessor is None:
        _preprocessor = joblib.load(PREPROCESSOR_PATH)
    if _risk_cluster is None:
        _risk_cluster = joblib.load(RISK_CLUSTER_PATH)
    if _voting_model is None:
        # Use cloudpickle to load the voting package with function
        with open(VOTING_MODEL_PATH, "rb") as f:
            vote_package = cloudpickle.load(f)
        _voting_model = vote_package
    return _preprocessor, _risk_cluster, _voting_model





def predict_returning_customer(customer_dict: dict):
    """
    Prepares features for a returning customer and returns prediction + confidence.
    """
    preprocessor, risk_cluster, voting_package = get_models()
    models = voting_package["models"]
    model_names = voting_package["model_names"]
    thresholds = voting_package["thresholds"]
    evaluate_voting = voting_package["voting_function"]

    # --- Step 1: Clean types ---
    clean_dict = {
        k: (v.item() if isinstance(v, (np.generic,)) else v)
        for k, v in customer_dict.items()
    }

    # --- Step 2: Build DataFrame ---
    df = pd.DataFrame([clean_dict])
    df = df.set_index("customerid")   # keep customerid only as index

    # --- Step 3: Apply preprocessing ---
    X_prep = preprocessor.transform(df)
    X_prep_df = pd.DataFrame(
        X_prep,
        index=df.index,
        columns=preprocessor.get_feature_names_out()
    )

    # --- Step 4: Apply clustering ---
    X_clustered = risk_cluster.transform(X_prep_df)
    X_clustered_df = pd.DataFrame(
        X_clustered,
        index=X_prep_df.index,
        columns=list(X_prep_df.columns) + ["risk_level"]
    )


    
    # --- Step 5: Predict ---
    X_final = X_clustered_df[features]

    # Use the voting function saved in the package
    y_pred_single, y_conf_single = evaluate_voting(models, model_names, thresholds, X_final)
    return y_pred_single[0], round(y_conf_single[0]*100, 2)