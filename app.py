import streamlit as st
import torch
import pandas as pd
import numpy as np

# Define the TitanicModel directly in this file
import torch.nn as nn

class TitanicModel(nn.Module):
    def __init__(self):
        super(TitanicModel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(5, 128),  # Expecting 5 input features now
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.network(x)

# Load the pre-trained model
model = TitanicModel()
model.load_state_dict(torch.load('titanic_model.pth'))
model.eval()  # Set the model to evaluation mode

# Center the layout by using Streamlit's columns
st.title("Titanic Survival Probability Prediction")
st.markdown("### Adjust the passenger features to see the survival probability")

# Define the layout with a centered column
with st.container():
    # Add a button to calculate survival probability
    col1, col2, col3 = st.columns([1, 2, 1])  # Center col2
    with col2:
        # Sidebar inputs for each feature
        pclass = st.selectbox("Pclass", [1, 2, 3], index=1)
        sex = st.selectbox("Sex", ["Female", "Male"], index=0)
        age = st.slider("Age", 0, 80, 25)
        parch = st.selectbox("Parch (Parents/Children Aboard)", [0, 1, 2, 3, 4, 5, 6], index=0)
        sibsp = st.selectbox("SibSp (Siblings/Spouses Aboard)", [0, 1, 2, 3, 4, 5, 8], index=0)

        # Add a button to calculate the survival probability
        if st.button("Calculate Survival Probability"):
            # Preprocess inputs similar to your training setup
            sex_encoded = 0 if sex == "Female" else 1
            age_scaled = 1 / (age + 1)  # Scaling age as done in your code

            # Apply weights for features, based on training configuration
            pclass_weighted = 1.5 if pclass == 1 else (1.2 if pclass == 2 else 1)
            sex_weighted = 1.5 if sex_encoded == 0 else 1
            parch_weights = {0: 1.0, 1: 1.6, 2: 1.5, 3: 1.7, 4: 0.8, 5: 0.9, 6: 0.8}
            parch_weighted = parch_weights[parch]
            sibsp_weights = {0: 1.0, 1: 1.3, 2: 1.2, 3: 0.9, 4: 0.8, 5: 0.7, 8: 0.5}
            sibsp_weighted = sibsp_weights[sibsp]

            # Combine features into a DataFrame, ensuring they align with model input
            input_data = pd.DataFrame({
                'Pclass_Weighted': [pclass_weighted],
                'Sex_Weighted': [sex_weighted],
                'Age_Scaled': [age_scaled],
                'Parch_Weighted': [parch_weighted],
                'SibSp_Weighted': [sibsp_weighted]
            })

            # Convert the input data to a PyTorch tensor
            input_tensor = torch.tensor(input_data.values, dtype=torch.float32)

            # Predict the survival probability
            with torch.no_grad():
                survival_probability = model(input_tensor).item()  # Get raw probability between 0 and 1

            # Display the result as a percentage
            st.write(f"Predicted Survival Probability: {survival_probability * 100:.2f}%")

# Style adjustments to center the input fields and enhance the look
st.markdown(
    """
    <style>
    .stApp {
        background-color: #2E2E2E;
        color: #FFFFFF;
    }
    .css-18e3th9 {
        width: 50%; /* Center column width */
        margin: 0 auto; /* Center on page */
    }
    .stButton > button {
        background-color: #0066cc;
        color: #FFFFFF;
        border-radius: 5px;
        width: 100%;
    }
    </style>
    """,
    unsafe_allow_html=True
)