import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import time

# -------------------------
# 💅 CSS Animations
# -------------------------
st.markdown("""
    <style>
    /* Animate only the last 3 main buttons: Predict, Clear, About AQI */
    div.stButton > button:nth-last-of-type(3),
    div.stButton > button:nth-last-of-type(2),
    div.stButton > button:last-of-type {
      background-image: linear-gradient(135deg, #6EC6FF, #2196F3);
      color: white !important;
      padding: 0.6rem 1.3rem;
      border: none;
      border-radius: 8px;
      font-size: 0.95rem;
      font-weight: 600;
      box-shadow: 0 6px 10px rgba(33, 150, 243, 0.25);
      transition: all 0.25s ease-in-out;
      margin-top: 8px;
    }
    div.stButton > button:nth-last-of-type(3):hover,
    div.stButton > button:nth-last-of-type(2):hover,
    div.stButton > button:last-of-type:hover {
      background-image: linear-gradient(135deg, #42A5F5, #1E88E5);
      transform: scale(1.02);
      box-shadow: 0 8px 14px rgba(33, 150, 243, 0.35);
    }
    div.stButton > button:nth-last-of-type(3):active,
    div.stButton > button:nth-last-of-type(2):active,
    div.stButton > button:last-of-type:active {
      background-image: linear-gradient(135deg, #1E88E5, #1565C0);
      transform: scale(0.98);
      box-shadow: 0 4px 8px rgba(21, 101, 192, 0.4);
    }

    /* Animate Train Model button in sidebar */
    div[data-testid="stSidebar"] button {
      background-image: linear-gradient(135deg, #AB47BC, #8E24AA);
      color: white !important;
      border: none;
      border-radius: 8px;
      font-size: 0.95rem;
      font-weight: 600;
      padding: 0.5rem 1rem;
      box-shadow: 0 5px 8px rgba(142, 36, 170, 0.25);
      transition: all 0.25s ease-in-out;
      margin-bottom: 10px;
    }
    div[data-testid="stSidebar"] button:hover {
      background-image: linear-gradient(135deg, #BA68C8, #9C27B0);
      transform: scale(1.02);
      box-shadow: 0 8px 14px rgba(142, 36, 170, 0.35);
    }
    div[data-testid="stSidebar"] button:active {
      background-image: linear-gradient(135deg, #8E24AA, #6A1B9A);
      transform: scale(0.98);
      box-shadow: 0 4px 8px rgba(106, 27, 154, 0.4);
    }
    </style>
""", unsafe_allow_html=True)

st.set_page_config(page_title="AQI Prediction System")

# -------------------------
# 1. Title and Load Data
# -------------------------
st.title("🌫 AQI Prediction System")
st.markdown("Predict Air Quality Category based on various pollutants.")

df = pd.read_csv("city_day.csv")

# -------------------------
# 2. Sample Data
# -------------------------
st.subheader("📁 Sample Data (Source: AQI_Prediction GitHub Repo)")
st.dataframe(df.head())
st.info(f"Dataset Shape: {df.shape}")

# -------------------------
# 3. Preprocessing
# -------------------------
st.markdown("## 📊 Let’s Prepare the Data!")

df.dropna(inplace=True)
le = LabelEncoder()
df['AQI_Category'] = le.fit_transform(df['AQI_Bucket'])

features = ['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2', 'O3', 'Benzene', 'Toluene']
units = {
    'PM2.5': 'μg/m³', 'PM10': 'μg/m³', 'NO': 'μg/m³', 'NO2': 'μg/m³',
    'NOx': 'μg/m³', 'NH3': 'μg/m³', 'CO': 'mg/m³', 'SO2': 'μg/m³',
    'O3': 'μg/m³', 'Benzene': 'μg/m³', 'Toluene': 'μg/m³'
}

X = df[features]
y = df['AQI_Category']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

with st.expander("🔧 See Preprocessed Sample Data"):
    st.dataframe(pd.DataFrame(X_scaled, columns=features).round(2), use_container_width=True)

st.markdown("### 🔍 Dataset Overview")
st.info(f"📌 Total records: *{X_scaled.shape[0]}, Features: **{X_scaled.shape[1]}*")
st.success(f"🎯 Labels: *{len(np.unique(y))}* categories")
st.warning(f"📚 Training Samples: *{X_train.shape[0]}, 🧪 Testing Samples: **{X_test.shape[0]}*")

# -------------------------
# 4. Sidebar - Model Selection & Train Button FIRST
# -------------------------
st.sidebar.title("🧠 Choose Your Model")
model_choice = st.sidebar.selectbox(
    "Select Classifier",
    [
        "Logistic Regression",
        "Support Vector Machine",
        "Random Forest Classifier",
        "Gradient Boosting Classifier"
    ]
)

model = None
if model_choice == "Logistic Regression":
    model = LogisticRegression(max_iter=1000)
elif model_choice == "Support Vector Machine":
    model = SVC()
elif model_choice == "Random Forest Classifier":
    model = RandomForestClassifier()
elif model_choice == "Gradient Boosting Classifier":
    model = GradientBoostingClassifier()

if st.sidebar.button("Train Model"):
    with st.spinner("⏳ Training in progress..."):
        start = time.time()
        model.fit(X_train, y_train)
        end = time.time()
        st.sidebar.success(f"✅ Model trained in {end - start:.2f} seconds")

        st.markdown("## 📈 Model Evaluation")
        y_pred = model.predict(X_test)
        st.write(f"*Accuracy Score:* {accuracy_score(y_test, y_pred):.2f}")

        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=le.classes_, yticklabels=le.classes_, ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title("Confusion Matrix")
        st.pyplot(fig)

        st.text("Classification Report:")
        st.text(classification_report(y_test, y_pred, target_names=le.classes_))

        st.session_state["trained_model"] = model

# -------------------------
# 5. Sidebar - About AFTER Train
# -------------------------
st.sidebar.markdown("---")
st.sidebar.markdown("""
    <div style="background-color:#ede7f6;padding:1rem;border-radius:0.5rem">
        <h3 style="color:#512da8;">📘 About This App</h3>
        <p style="color:#311b92;">
        This Air Quality Index (AQI) Prediction app uses <b>Machine Learning</b> models
        to classify pollution levels based on key environmental features.
        </p>
        <p style="color:#311b92;">
        Trained on real data, it predicts categories like <i>Good</i>, <i>Moderate</i>, or <i>Poor</i>
        AQI from pollutants such as PM2.5, NOx, SO2, and CO.
        </p>
    </div>
""", unsafe_allow_html=True)

# -------------------------
# 6. Predict AQI Inputs
# -------------------------
st.markdown("## 🔮 Predict New AQI Category")
st.markdown("### 📏 Feature Units (Input Guidelines)")

sample_input = []
for feature in features:
    st.markdown(f"**{feature}** *(Unit: {units[feature]})*")
    value = st.number_input(f"Enter value for {feature}", min_value=0.0, step=1.0)
    sample_input.append(value)

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("Predict AQI"):
        if "trained_model" in st.session_state:
            input_array = np.array(sample_input).reshape(1, -1)
            input_scaled = scaler.transform(input_array)
            prediction = st.session_state["trained_model"].predict(input_scaled)
            category = le.inverse_transform(prediction)[0]
            st.success(f"✅ Predicted AQI Category: *{category}*")
        else:
            st.error("⚠ Please train a model first from the sidebar.")


with col2:
    if st.button("Clear Inputs"):
        st.warning("🔄 Input fields reset when the page reloads. You can manually adjust values.")

with col3:
    if st.button("About AQI"):
        st.info("""
        Air Quality Index (AQI) helps track pollution levels in a region.
        Common pollutants and their typical units:
        - PM2.5 / PM10 / NOx / SO2 / O3: **μg/m³**
        - CO: **mg/m³**
        - Benzene / Toluene: **μg/m³**
        Values are interpreted on health impact scales defined by CPCB & WHO.
        """)