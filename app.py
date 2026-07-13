

import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Heart Disease Risk Predictor", page_icon="❤️", layout="centered")


FIELD_LABELS = {
    "male": "Sex",
    "age": "Age",
    "education": "Education Level (1 = lowest, 4 = highest)",
    "currentSmoker": "Current Smoker?",
    "cigsPerDay": "Cigarettes per Day",
    "BPMeds": "On Blood Pressure Medication?",
    "prevalentStroke": "History of Stroke?",
    "prevalentHyp": "Hypertension?",
    "diabetes": "Diabetes?",
    "totChol": "Total Cholesterol (mg/dL)",
    "sysBP": "Systolic Blood Pressure (mmHg)",
    "diaBP": "Diastolic Blood Pressure (mmHg)",
    "BMI": "BMI",
    "heartRate": "Heart Rate (bpm)",
    "glucose": "Glucose (mg/dL)",
}


@st.cache_resource
def load_model(path: str = "model.pkl"):
    return joblib.load(path)


def numeric_input(col, numeric_stats):
    stats = numeric_stats[col]
    lo, hi, default = stats["min"], stats["max"], stats["mean"]
    label = FIELD_LABELS.get(col, col)
    if col in ("age", "cigsPerDay", "education"):
        return st.number_input(
            label, min_value=int(lo), max_value=int(hi) + 5, value=int(round(default)), step=1, key=col
        )
    return st.number_input(
        label, min_value=float(lo) * 0.5, max_value=float(hi) * 1.2,
        value=float(default), step=0.5, key=col,
    )


def binary_input(col):
    label = FIELD_LABELS.get(col, col)
    if col == "male":
        choice = st.radio(label, options=["Female", "Male"], horizontal=True, key=col)
        return 1 if choice == "Male" else 0
    choice = st.radio(label, options=["No", "Yes"], horizontal=True, key=col)
    return 1 if choice == "Yes" else 0


def main():
    st.title("❤️ 10-Year Heart Disease Risk Predictor")
    st.write(
        "Enter the patient's information below to estimate their risk of "
        "developing coronary heart disease (CHD) within the next 10 years, "
        "based on the Framingham Heart Study dataset."
    )

    try:
        data = load_model()
    except FileNotFoundError:
        st.error(
            "Could not find 'model.pkl'. Please run `python train_model.py` "
            "first (with 'framingham.csv' in the same folder) to generate "
            "the trained model file."
        )
        return

    model = data["model"]
    feature_columns = data["feature_columns"]
    binary_fields = set(data["binary_fields"])
    numeric_stats = data["numeric_stats"]
    test_accuracy = data.get("test_accuracy")
    best_params = data.get("best_params")

    responses = {}

    tab1, tab2, tab3 = st.tabs(["👤 Demographics", "🩺 Medical History", "📈 Vitals & Labs"])

    with tab1:
        for col in ["male", "age", "education"]:
            if col in binary_fields:
                responses[col] = binary_input(col)
            else:
                responses[col] = numeric_input(col, numeric_stats)

    with tab2:
        for col in ["currentSmoker", "cigsPerDay", "BPMeds", "prevalentStroke", "prevalentHyp", "diabetes"]:
            if col in binary_fields:
                responses[col] = binary_input(col)
            else:
                responses[col] = numeric_input(col, numeric_stats)

    with tab3:
        for col in ["totChol", "sysBP", "diaBP", "BMI", "heartRate", "glucose"]:
            responses[col] = numeric_input(col, numeric_stats)

  
    placed = {"male", "age", "education", "currentSmoker", "cigsPerDay", "BPMeds",
              "prevalentStroke", "prevalentHyp", "diabetes", "totChol", "sysBP",
              "diaBP", "BMI", "heartRate", "glucose"}
    leftover = [c for c in feature_columns if c not in placed]
    if leftover:
        with st.expander("Additional fields"):
            for col in leftover:
                if col in binary_fields:
                    responses[col] = binary_input(col)
                else:
                    responses[col] = numeric_input(col, numeric_stats)

    st.divider()

    if st.button("Predict CHD Risk", type="primary"):
        input_df = pd.DataFrame([responses])[feature_columns]

        prediction = model.predict(input_df)[0]
        proba = model.predict_proba(input_df)[0][1]

        if prediction == 1:
            st.error(f"⚠️ **High Risk** of CHD within 10 years. (Estimated probability: {proba:.1%})")
        else:
            st.success(f"✅ **Low Risk** of CHD within 10 years. (Estimated probability: {proba:.1%})")

        st.progress(min(max(proba, 0.0), 1.0))
        st.caption(
            "This is a statistical estimate based on the Framingham dataset and is "
            "not a medical diagnosis. Please consult a healthcare professional."
        )

    st.divider()
    with st.expander("ℹ️ About this model"):
        st.write(
            f"""
            - **Model type:** Random Forest Classifier (with median/mean imputation for missing values)
            - **Target:** TenYearCHD — risk of coronary heart disease within 10 years
            - **Training approach:** minority class (CHD = Yes) was oversampled with
              `RandomOverSampler` before training; hyperparameters were tuned via
              5-fold `GridSearchCV` optimizing ROC-AUC.
            {f"- **Best hyperparameters:** {best_params}" if best_params else ""}
            {f"- **Held-out test accuracy:** {test_accuracy:.1%}" if test_accuracy is not None else ""}
            """
        )


if __name__ == "__main__":
    main()
