"""
Lesson 7 (part 2) — LOAD the saved model and predict (NO retraining)
====================================================================
This is the "predict" half. Notice: there is NO .fit() here, NO training data,
NO train_test_split. We just LOAD the saved model artifact and use it.

This is how prediction works in production: a service loads the model file once
at startup, then answers prediction requests instantly.

Run from the project root (with .venv activated), AFTER lesson_07_train_and_save:

    python lessons/lesson_07_load_and_predict.py
"""

import joblib
import pandas as pd

# --- LOAD the trained model from disk (no training happens!) ---
model = joblib.load("models/churn_model.joblib")
feature_columns = joblib.load("models/feature_columns.joblib")
print("Loaded trained model from disk. Ready to predict — no retraining.\n")

# --- Make up a NEW customer to score ---
# (In real life this would come from a web request, a database, etc.)
# We start from all-zeros, then set a few fields to describe the customer.
new_customer = {col: 0 for col in feature_columns}
new_customer["tenure"] = 2                 # only 2 months -> new customer
new_customer["MonthlyCharges"] = 95.0      # high monthly bill
new_customer["TotalCharges"] = 190.0       # 2 months * ~95
new_customer["InternetService_Fiber optic"] = 1   # fiber (a churn signal)
new_customer["PaperlessBilling_Yes"] = 1
new_customer["PaymentMethod_Electronic check"] = 1

# Turn the dict into a one-row DataFrame with columns in the SAME order.
X_new = pd.DataFrame([new_customer])[feature_columns]

# --- Predict ---
prob = model.predict_proba(X_new)[0][1]   # probability of churn (class 1)
label = model.predict(X_new)[0]           # 0 = stay, 1 = churn

print("New customer profile: 2 months tenure, $95/mo, fiber, e-check, paperless")
print(f"  Churn probability : {prob:.1%}")
print(f"  Prediction        : {'WILL CHURN' if label == 1 else 'will stay'}")
print("\nNo model.fit() was called here — we reused the trained artifact.")
