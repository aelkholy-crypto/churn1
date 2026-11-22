import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

st.set_page_config(
    page_title="Decision Tree AI Dashboard",
    page_icon="🌳",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .stApp {
        background-color: #08080C;
        color: #F0F2F6;
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0D1117 0%, #160B14 100%);
        border-right: 2px solid #2A1B3D;
    }

    .gradient-title {
        font-size: 2.6rem;
        font-weight: 800;
        background: linear-gradient(90deg, #1E90FF 0%, #9B51E0 50%, #FF2E4C 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }

    .subtitle {
        color: #A0AAB5;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }

    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, rgba(30, 144, 255, 0.12) 0%, rgba(255, 46, 76, 0.12) 100%);
        border: 1px solid rgba(155, 81, 224, 0.4);
        border-radius: 12px;
        padding: 15px 20px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.5);
    }

    div[data-testid="stMetricValue"] {
        background: linear-gradient(90deg, #38BDF8, #FB7185);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
    }

    .stButton > button {
        width: 100%;
        background: linear-gradient(90deg, #1E90FF 0%, #FF2E4C 100%);
        color: white;
        font-weight: bold;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 1rem;
        transition: all 0.3s ease;
    }

    .stButton > button:hover {
        opacity: 0.9;
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(255, 46, 76, 0.4);
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }

    .stTabs [data-baseweb="tab"] {
        background-color: #12141D;
        border-radius: 8px 8px 0 0;
        color: #CBD5E1;
        padding: 10px 20px;
        border: 1px solid #1E293B;
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(90deg, rgba(30, 144, 255, 0.25), rgba(255, 46, 76, 0.25)) !important;
        border-bottom: 3px solid #FF2E4C !important;
        color: #FFFFFF !important;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

model=joblib.load("test.pkl")

@st.cache_data
def load_sample_data():
    df = pd.read_csv("111.csv.csv")

    X, y = make_classification(
        n_samples=600,
        n_features=4,
        n_informative=3,
        n_redundant=1,
        random_state=42
    )
    df = pd.DataFrame(X, columns=["Feature_1", "Feature_2", "Feature_3", "Feature_4"])
    df["Target"] = y
    return df

with st.sidebar:
    st.markdown("## Model Settings (DT)")
    st.markdown("---")

    data_source = st.radio(
        "Data Source:",
        ["Sample Dataset", "Upload CSV File"],
        index=0
    )

    if data_source == "Upload CSV File":
        uploaded_file = st.file_uploader("Upload Dataset (Last column must be Target)", type=["csv"])
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
        else:
            st.info("Please upload a CSV file or select the sample dataset.")
            df = load_sample_data()
    else:
        df = load_sample_data()

    st.markdown("### Decision Tree Hyperparameters")
    criterion = st.selectbox("Split Criterion", ["gini", "entropy", "log_loss"])
    max_depth = st.slider("Max Tree Depth (max_depth)", min_value=1, max_value=15, value=4)
    min_samples_split = st.slider("Min Samples Split (min_samples_split)", min_value=2, max_value=20, value=2)
    test_size = st.slider("Test Set Ratio (Test Size)", min_value=0.1, max_value=0.5, value=0.2, step=0.05)

    st.markdown("---")
    st.caption("Dark Mode UI with Blue-to-Red Gradient Theme")

X = df.iloc[:, :-1]
y = df.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

dt_model = DecisionTreeClassifier(
    criterion=criterion,
    max_depth=max_depth,
    min_samples_split=min_samples_split,
    random_state=42
)
dt_model.fit(X_train, y_train)
y_pred = dt_model.predict(X_test)

train_acc = accuracy_score(y_train, dt_model.predict(X_train))
test_acc = accuracy_score(y_test, y_pred)

st.markdown('<div class="gradient-title">Decision Tree Classifier Studio</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Interactive dashboard for data analysis and Decision Tree model training</div>', unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Test Accuracy", f"{test_acc * 100:.2f}%")
col2.metric("Train Accuracy", f"{train_acc * 100:.2f}%")
col3.metric("Total Samples", f"{df.shape[0]}")
col4.metric("Actual Tree Depth", f"{dt_model.get_depth()}")

st.markdown("---")

tab1, tab2, tab3, tab4 = st.tabs([
    "Interactive Prediction",
    "Tree Visualization",
    "Model Evaluation",
    "Dataset Explorer"
])

with tab1:
    st.subheader("Enter Feature Values for Real-Time Prediction")
    input_data = {}
    cols = st.columns(2)

    for idx, col_name in enumerate(X.columns):
        with cols[idx % 2]:
            min_val = float(X[col_name].min())
            max_val = float(X[col_name].max())
            mean_val = float(X[col_name].mean())
            input_data[col_name] = st.number_input(
                f"Value for {col_name}",
                min_value=min_val - abs(min_val),
                max_value=max_val + abs(max_val),
                value=mean_val,
                format="%.4f"
            )

    if st.button("Run Prediction"):
        input_df = pd.DataFrame([input_data])
        prediction = dt_model.predict(input_df)[0]
        probabilities = dt_model.predict_proba(input_df)[0]

        st.markdown(f"""
        <div style="
            background: linear-gradient(90deg, rgba(30, 144, 255, 0.2), rgba(255, 46, 76, 0.2));
            border-left: 5px solid #FF2E4C;
            padding: 20px;
            border-radius: 10px;
            margin-top: 15px;
        ">
            <h3 style="margin:0; color:#FFFFFF;">Predicted Class: <span style="color:#38BDF8;">{prediction}</span></h3>
            <p style="margin:5px 0 0 0; color:#CBD5E1;">Prediction Confidence: <b>{max(probabilities)*100:.2f}%</b></p>
        </div>
        """, unsafe_allow_html=True)

with tab2:
    st.subheader("Decision Tree Structure Visualization")
    fig, ax = plt.subplots(figsize=(16, 8), facecolor="#08080C")
    plot_tree(
        dt_model,
        feature_names=list(X.columns),
        class_names=[str(c) for c in dt_model.classes_],
        filled=True,
        rounded=True,
        fontsize=9,
        ax=ax
    )
    st.pyplot(fig, width="stretch")

with tab3:
    c1, c2 = st.columns([1, 1])

    with c1:
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        fig_cm, ax_cm = plt.subplots(figsize=(6, 4), facecolor="#08080C")
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap=sns.blend_palette(["#1E90FF", "#9B51E0", "#FF2E4C"], as_cmap=True),
            cbar=False,
            ax=ax_cm,
            annot_kws={"size": 14, "weight": "bold", "color": "white"}
        )
        ax_cm.set_facecolor("#08080C")
        ax_cm.set_xlabel("Predicted", color="white")
        ax_cm.set_ylabel("Actual", color="white")
        ax_cm.tick_params(colors="white")
        st.pyplot(fig_cm, width="stretch")

    with c2:
        st.subheader("Feature Importance")
        feat_imp = pd.DataFrame({
            "Feature": X.columns,
            "Importance": dt_model.feature_importances_
        }).sort_values(by="Importance", ascending=False)

        st.dataframe(feat_imp, width="stretch", hide_index=True)

        st.subheader("Classification Report")
        report = classification_report(y_test, y_pred, output_dict=True)
        st.dataframe(pd.DataFrame(report).transpose(), width="stretch")

with tab4:
    st.subheader("Dataset Preview")
    st.dataframe(df.head(25), width="stretch")