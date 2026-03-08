import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download and load the model
model_path = hf_hub_download(
    repo_id="j907/tourism_package_prediction_model",
    filename="best_tourism_package_prediction_model_v1.joblib"
    )

model = joblib.load(model_path)

# Streamlit UI for Tourism Package Prediction
st.title("Tourism Package Prediction App")
st.write("""
This application predicts the likelihood of purchasing the Wellness Tourism Package based on its operational parameters.
Please enter the customer and interaction data below to get a prediction.
""")

type_options = ["Self Enquiry", "Company Invited"]
gender_options = ["Male", "Female"]
marital_options = ["Single", "Married", "Divorced"]

TypeofContact = st.selectbox("Type of Contact", type_options)
Gender = st.selectbox("Gender", gender_options)
MaritalStatus = st.selectbox("Marital Status", marital_options)

Occupation = st.text_input("Occupation", value="Salaried")
ProductPitched = st.text_input("Product Pitched", value="Basic")
Designation = st.text_input("Designation", value="Executive")

Age = st.number_input("Age", min_value=16.0, max_value=100.0, value=35.0, step=0.1)
CityTier = st.selectbox("City Tier", [1, 2, 3], index=0)
DurationOfPitch = st.number_input("Duration Of Pitch (min)", min_value=0.0, max_value=120.0, value=8.0, step=0.1)
NumberOfPersonVisiting = st.number_input("Number Of Persons Visiting", min_value=0, max_value=20, value=3, step=1)
NumberOfFollowups = st.number_input("Number Of Followups", min_value=0.0, max_value=100.0, value=2.0, step=1.0)
PreferredPropertyStar = st.number_input("Preferred Property Star (1-5)", min_value=1.0, max_value=5.0, value=3.0, step=0.5)
NumberOfTrips = st.number_input("Number Of Trips (per year)", min_value=0.0, max_value=100.0, value=1.0, step=1.0)
Passport = st.selectbox("Passport (0 = No, 1 = Yes)", [0, 1], index=1)
PitchSatisfactionScore = st.number_input("Pitch Satisfaction Score", min_value=0, max_value=10, value=3, step=1)
OwnCar = st.selectbox("Own Car (0 = No, 1 = Yes)", [0, 1], index=1)
NumberOfChildrenVisiting = st.number_input("Number Of Children Visiting", min_value=0.0, max_value=10.0, value=0.0, step=1.0)
MonthlyIncome = st.number_input("Monthly Income", min_value=0.0, max_value=1_000_000.0, value=20000.0, step=100.0, format="%.2f")

input_df = pd.DataFrame([{
    "Age": Age,
    "TypeofContact": TypeofContact,
    "CityTier": CityTier,
    "DurationOfPitch": DurationOfPitch,
    "Occupation": Occupation.strip(),
    "Gender": Gender,
    "NumberOfPersonVisiting": NumberOfPersonVisiting,
    "NumberOfFollowups": NumberOfFollowups,
    "ProductPitched": ProductPitched.strip(),
    "PreferredPropertyStar": PreferredPropertyStar,
    "MaritalStatus": MaritalStatus,
    "NumberOfTrips": NumberOfTrips,
    "Passport": Passport,
    "PitchSatisfactionScore": PitchSatisfactionScore,
    "OwnCar": OwnCar,
    "NumberOfChildrenVisiting": NumberOfChildrenVisiting,
    "Designation": Designation.strip(),
    "MonthlyIncome": MonthlyIncome
}])


if st.button("Predict"):
    prediction = model.predict(input_df)[0]
    result = "Customer purchased a package" if prediction == 1 else "Customer didn't purchase any package"
    st.subheader("Prediction Result:")
    st.success(f"The model predicts that : **{result}**")
