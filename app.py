import streamlit as st
import pandas as pd
import pickle
import numpy as np

# ═══════════════════════════════════════════════════════════════════════════
# LOAD THE MACHINE LEARNING "BRAIN"
# ═══════════════════════════════════════════════════════════════════════════
# We trained a model earlier - now we're waking it up from its pickle nap.
# Think of this like loading your saved game before you can play.

with open('model.pkl', 'rb') as file:
    model = pickle.load(file)  # The trained ML model - our prediction engine

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)  # The scaler that normalizes our data - keeps everything on the same playing field

# ═══════════════════════════════════════════════════════════════════════════
# BUILD THE USER INTERFACE
# ═══════════════════════════════════════════════════════════════════════════

st.title("Customer Churn Prediction Dashboard")

st.write("Enter customer details to predict churn risk:")

# Create input fields for all our features
# Pro tip: These min/max values aren't arbitrary - they match our training data ranges
credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=650)
age = st.slider("Age", 18, 100, 40)
tenure = st.slider("Tenure (Years)", 0, 10, 3)
balance = st.number_input("Account Balance", min_value=0.0, value=50000.0)
num_products = st.slider("Number of Products", 1, 4, 2)
has_card = st.checkbox("Has Credit Card?", value=True)
is_active = st.checkbox("Is Active Member?", value=True)
salary = st.number_input("Estimated Salary", min_value=0.0, value=50000.0)

# Categorical features get dropdown treatment
location = st.selectbox("Geography", ["France", "Germany", "Spain"])
gender = st.selectbox("Gender", ["Male", "Female"])

# ═══════════════════════════════════════════════════════════════════════════
# THE MAGIC BUTTON
# ═══════════════════════════════════════════════════════════════════════════
# Everything below only runs when the user clicks "Predict"
# This keeps us from making predictions on every slider wiggle

if st.button("Predict Churn Risk"):
    
    # ───────────────────────────────────────────────────────────────────────
    # STEP 1: Geography One-Hot Encoding
    # ───────────────────────────────────────────────────────────────────────
    # We turn "France/Germany/Spain" into binary flags
    # Why? ML models need numbers, not words!
    # Note: France is the "reference category" (both flags = 0 means France)
    geo_germany = 1 if location == "Germany" else 0
    geo_spain = 1 if location == "Spain" else 0
    
    # ───────────────────────────────────────────────────────────────────────
    # STEP 2: Gender Label Encoding
    # ───────────────────────────────────────────────────────────────────────
    # Convert Male/Female to 1/0 (Male=1, Female=0)
    # Simpler than one-hot since we only have two options
    gender_male = 1 if gender == "Male" else 0
    
    # ───────────────────────────────────────────────────────────────────────
    # STEP 3: Assemble the Feature Vector
    # ───────────────────────────────────────────────────────────────────────
    # CRITICAL: This order MUST match exactly how we trained the model!
    # If the order is wrong, we're feeding the model gibberish
    # Think of it like a form - every blank needs the right answer in the right spot
    input_data = np.array([[
        credit_score,      # Position 0
        age,               # Position 1
        tenure,            # Position 2
        balance,           # Position 3
        num_products,      # Position 4
        int(has_card),     # Position 5 - convert bool to 0/1
        int(is_active),    # Position 6 - convert bool to 0/1
        salary,            # Position 7
        geo_germany,       # Position 8
        geo_spain,         # Position 9
        gender_male        # Position 10
    ]])
    
    # ───────────────────────────────────────────────────────────────────────
    # STEP 4: Scale the Input
    # ───────────────────────────────────────────────────────────────────────
    # Same scaler we used during training - this normalizes the data
    # Why? A salary of 50,000 shouldn't drown out an age of 40 just because it's bigger
    input_scaled = scaler.transform(input_data)
    
    # ───────────────────────────────────────────────────────────────────────
    # STEP 5: Get Predictions
    # ───────────────────────────────────────────────────────────────────────
    prediction = model.predict(input_scaled)           # Hard prediction: 0 or 1
    probability = model.predict_proba(input_scaled)    # Soft prediction: probability for each class
    
    # ───────────────────────────────────────────────────────────────────────
    # STEP 6: Show the Results
    # ───────────────────────────────────────────────────────────────────────
    st.subheader("Results:")
    
    # Extract the probability of churn (class 1)
    # probability[0] = first (and only) prediction
    # probability[0][1] = probability of the positive class (churn)
    churn_prob = probability[0][1]
    
    # Format as percentage with 2 decimal places (e.g., "67.42%")
    formatted_prob = "{:.2%}".format(churn_prob)
    
    # ───────────────────────────────────────────────────────────────────────
    # Decision Logic: HIGH RISK PATH
    # ───────────────────────────────────────────────────────────────────────
    # If probability > 50%, this customer is more likely to leave than stay
    if churn_prob > 0.5:
        
        # Show the bad news with red error styling
        st.error(f"High Churn Risk! (Probability: {formatted_prob})")
        
        # ═══════════════════════════════════════════════════════════════════
        # BRINGING THAT BUSINESS VALUE
        # ═══════════════════════════════════════════════════════════════════
        # This is where we translate ML predictions into dollars and decisions
        # Because executives care about money, not model accuracy
        
        st.write("---")  # Visual separator - just a horizontal line
        st.subheader("Potential Loss Analysis")
        
        # Simple assumption: If customer leaves, we lose their entire balance
        # (In reality, retention costs vs. balance value would be more complex)
        st.write(f"If this customer leaves, the bank risks losing an account balance of:")
        
        # st.metric() creates those nice big number displays
        # The :,.2f formatting adds commas and 2 decimal places (50000.00 → 50,000.00)
        st.metric(label="Account Value At Risk", value=f"${balance:,.2f}")
        
        # Actionable recommendation for the business team
        # This turns a prediction into a strategy
        st.write("**Recommendation:** Consider offering a loyalty bonus or tenure-based interest rate upgrade.")
        
    # ───────────────────────────────────────────────────────────────────────
    # Decision Logic: LOW RISK PATH
    # ───────────────────────────────────────────────────────────────────────
    # If probability ≤ 50%, customer is likely to stick around
    else:
        
        # Green success message - all is well
        st.success(f"Low Churn Risk (Probability: {formatted_prob})")
        
        # Keep it simple - happy customers don't need intervention
        st.write("This customer is likely to stay. No immediate action needed.")