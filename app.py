# ─────────────────────────────────────────────────────────────────────────────
# app.py — Student Grade Prediction System
# Iyinoluwa Don-Taiwo | McPherson University
#
# Run with:  streamlit run app.py
#
# This app loads trained models saved by the notebook and provides:
#   🏠 Home            — overview and dataset stats
#   🔮 Predict         — enter student details, get Pass/Fail + confidence
#   📊 Data Explorer   — browse data, distributions, correlation heatmap
#   🤖 Model Comparison — compare all models with metrics, matrices, ROC curves
#   💡 Feature Insights — feature importance, pass rate by feature
# ─────────────────────────────────────────────────────────────────────────────

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, roc_curve, auc
)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG — must be the very first Streamlit call in the script
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Student Grade Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────────────────────
# ENCODING MAP
#
# LabelEncoder assigns integers alphabetically per column.
# We reproduce that mapping here so the Predict page can convert
# user-selected text options to the same integers the models were trained on.
# ─────────────────────────────────────────────────────────────────────────────
ENCODING = {
    'school':     {'GP': 0, 'MS': 1},
    'sex':        {'F': 0, 'M': 1},
    'address':    {'R': 0, 'U': 1},
    'famsize':    {'GT3': 0, 'LE3': 1},
    'Pstatus':    {'A': 0, 'T': 1},
    'Mjob':       {'at_home': 0, 'health': 1, 'other': 2, 'services': 3, 'teacher': 4},
    'Fjob':       {'at_home': 0, 'health': 1, 'other': 2, 'services': 3, 'teacher': 4},
    'reason':     {'course': 0, 'home': 1, 'other': 2, 'reputation': 3},
    'guardian':   {'father': 0, 'mother': 1, 'other': 2},
    'schoolsup':  {'no': 0, 'yes': 1},
    'famsup':     {'no': 0, 'yes': 1},
    'paid':       {'no': 0, 'yes': 1},
    'activities': {'no': 0, 'yes': 1},
    'nursery':    {'no': 0, 'yes': 1},
    'higher':     {'no': 0, 'yes': 1},
    'internet':   {'no': 0, 'yes': 1},
    'romantic':   {'no': 0, 'yes': 1},
}

# The exact column order the models were trained on — input MUST match this
FEATURE_ORDER = [
    'school', 'sex', 'age', 'address', 'famsize', 'Pstatus',
    'Medu', 'Fedu', 'Mjob', 'Fjob', 'reason', 'guardian',
    'traveltime', 'studytime', 'failures', 'schoolsup', 'famsup',
    'paid', 'activities', 'nursery', 'higher', 'internet', 'romantic',
    'famrel', 'freetime', 'goout', 'Dalc', 'Walc', 'health', 'absences'
]


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# @st.cache_data — result is cached so the CSV is only read once per session
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data
def load_data(subject):
    filename = 'student-mat.csv' if subject == 'Mathematics' else 'student-por.csv'
    path = os.path.join('data', filename)
    if not os.path.exists(path):
        st.error(f"Dataset not found at '{path}'. Make sure data/ contains both CSV files.")
        st.stop()
    return pd.read_csv(path, sep=';')


@st.cache_data
def preprocess(df):
    """
    Apply the same preprocessing as the notebook:
    1. Create binary pass/fail target
    2. Drop G1, G2, G3
    3. Label-encode categorical columns
    4. Separate X and y
    """
    df = df.copy()
    df['pass'] = df['G3'].apply(lambda x: 1 if x >= 10 else 0)
    df_model = df.drop(columns=['G1', 'G2', 'G3'])
    categorical_cols = df_model.select_dtypes(include='object').columns.tolist()
    le = LabelEncoder()
    for col in categorical_cols:
        df_model[col] = le.fit_transform(df_model[col])
    X = df_model.drop(columns=['pass'])
    y = df_model['pass']
    return X, y, df


# ─────────────────────────────────────────────────────────────────────────────
# MODEL LOADING
# @st.cache_resource — for objects like models that shouldn't be re-instantiated
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_models(subject):
    prefix = 'mat' if subject == 'Mathematics' else 'por'
    model_keys = ['logistic_regression', 'random_forest', 'xgboost', 'svm', 'knn', 'decision_tree']
    models = {}
    for key in model_keys:
        path = os.path.join('models', f'{prefix}_{key}.pkl')
        if os.path.exists(path):
            with open(path, 'rb') as f:
                models[key.replace('_', ' ').title()] = pickle.load(f)
    return models


@st.cache_resource
def load_scaler(subject):
    prefix = 'mat' if subject == 'Mathematics' else 'por'
    path = os.path.join('models', f'{prefix}_scaler.pkl')
    if os.path.exists(path):
        with open(path, 'rb') as f:
            return pickle.load(f)
    return None


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR — Navigation & Dataset Selector
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🎓 Grade Predictor")
    st.markdown("---")
    page = st.radio("Navigate", [
        "🏠 Home",
        "🔮 Predict",
        "📊 Data Explorer",
        "🤖 Model Comparison",
        "💡 Feature Insights",
    ])
    st.markdown("---")
    subject = st.selectbox("Subject", ["Mathematics", "Portuguese"])
    st.markdown("---")
    st.caption("Iyinoluwa Don-Taiwo · iyinoluwadontaiwo@gmail.com")


# Load everything for the selected subject (cached after first run)
df_raw        = load_data(subject)
X, y, df_full = preprocess(df_raw)
models        = load_models(subject)
scaler        = load_scaler(subject)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 1 — HOME
# ═════════════════════════════════════════════════════════════════════════════
if page == "🏠 Home":

    st.title("🎓 Student Grade Prediction System")
    st.markdown("#### Predicting pass/fail outcomes from student background and lifestyle — no prior grades needed.")

    st.markdown("""
    This application applies machine learning to the **UCI Student Performance Dataset**
    to predict whether a secondary school student will pass their final exam.

    **Key design decision:** G1 and G2 (first and second period grades) are excluded from all models.
    This makes the prediction useful at the **start of term**, before any grades exist,
    enabling early identification of at-risk students.
    """)

    st.markdown("---")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Students",    len(df_raw))
    c2.metric("Pass Rate",         f"{df_full['pass'].mean()*100:.1f}%")
    c3.metric("Features Used",     X.shape[1])
    c4.metric("Models Available",  len(models) if models else "Run notebook first")

    st.markdown("---")
    st.markdown("### What each page does")
    st.markdown("""
| Page | Description |
|---|---|
| 🔮 **Predict** | Enter a student's details → instant Pass/Fail prediction with confidence score and risk factor summary |
| 📊 **Data Explorer** | Browse the dataset, plot any feature's distribution, explore the correlation heatmap |
| 🤖 **Model Comparison** | All 6 models side by side — metrics table, confusion matrices, ROC curves |
| 💡 **Feature Insights** | Feature importance chart, pass rate by feature value |
    """)

    st.markdown("---")
    st.markdown(
        "**Dataset:** Cortez, P. (2008). *Student Performance* [Dataset]. "
        "Available on [Kaggle](https://www.kaggle.com/datasets/dskagglemt/student-performance-data-set)"
    )


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 2 — PREDICT
# ═════════════════════════════════════════════════════════════════════════════
elif page == "🔮 Predict":

    st.title("🔮 Predict Student Outcome")
    st.markdown("Fill in the student's details to get a Pass/Fail prediction.")

    if not models:
        st.warning("⚠️ No saved models found in models/. Run the notebook end-to-end first.")
        st.stop()
    if scaler is None:
        st.warning("⚠️ Scaler not found in models/. Run the notebook end-to-end first.")
        st.stop()

    model_choice = st.selectbox("Choose a model", list(models.keys()))
    st.markdown("---")
    st.markdown("### Student Details")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("**Demographics**")
        school    = st.selectbox("School", ["GP", "MS"], help="GP = Gabriel Pereira | MS = Mousinho da Silveira")
        sex       = st.selectbox("Sex", ["F", "M"])
        age       = st.slider("Age", 15, 22, 17)
        address   = st.selectbox("Address", ["U", "R"], help="U = Urban | R = Rural")
        famsize   = st.selectbox("Family Size", ["GT3", "LE3"], help="GT3 = more than 3 | LE3 = 3 or fewer")
        Pstatus   = st.selectbox("Parents' Status", ["T", "A"], help="T = together | A = apart")

    with c2:
        st.markdown("**Family Background**")
        Medu      = st.slider("Mother's Education (0–4)", 0, 4, 2, help="0=none 1=primary 2=middle 3=secondary 4=higher")
        Fedu      = st.slider("Father's Education (0–4)", 0, 4, 2)
        Mjob      = st.selectbox("Mother's Job", ["at_home", "health", "services", "teacher", "other"])
        Fjob      = st.selectbox("Father's Job", ["at_home", "health", "services", "teacher", "other"])
        guardian  = st.selectbox("Guardian", ["mother", "father", "other"])
        reason    = st.selectbox("School Choice Reason", ["reputation", "home", "course", "other"])

    with c3:
        st.markdown("**Academic Profile**")
        studytime = st.slider("Weekly Study Hours (1–4)", 1, 4, 2, help="1=<2h 2=2-5h 3=5-10h 4=>10h")
        failures  = st.slider("Past Failures", 0, 3, 0)
        absences  = st.slider("School Absences", 0, 93, 4)
        traveltime= st.slider("Travel Time (1–4)", 1, 4, 1, help="1=<15min 2=15-30min 3=30-60min 4=>1h")
        higher    = st.selectbox("Wants Higher Education?", ["yes", "no"])
        internet  = st.selectbox("Internet at Home?", ["yes", "no"])

    c4, c5 = st.columns(2)

    with c4:
        st.markdown("**Support & Activities**")
        schoolsup  = st.selectbox("School Extra Support?", ["no", "yes"])
        famsup     = st.selectbox("Family Educational Support?", ["yes", "no"])
        paid       = st.selectbox("Extra Paid Classes?", ["no", "yes"])
        activities = st.selectbox("Extracurricular Activities?", ["no", "yes"])
        nursery    = st.selectbox("Attended Nursery School?", ["yes", "no"])
        romantic   = st.selectbox("In a Romantic Relationship?", ["no", "yes"])

    with c5:
        st.markdown("**Social & Lifestyle**")
        famrel   = st.slider("Family Relationship Quality (1–5)", 1, 5, 4)
        freetime = st.slider("Free Time After School (1–5)", 1, 5, 3)
        goout    = st.slider("Going Out with Friends (1–5)", 1, 5, 2)
        Dalc     = st.slider("Weekday Alcohol Use (1–5)", 1, 5, 1)
        Walc     = st.slider("Weekend Alcohol Use (1–5)", 1, 5, 1)
        health   = st.slider("Health Status (1–5)", 1, 5, 3)

    st.markdown("---")

    if st.button("🔮 Predict Now", use_container_width=True, type="primary"):

        # Build the input row — encode each text value using the training mapping
        input_dict = {
            'school':     ENCODING['school'][school],
            'sex':        ENCODING['sex'][sex],
            'age':        age,
            'address':    ENCODING['address'][address],
            'famsize':    ENCODING['famsize'][famsize],
            'Pstatus':    ENCODING['Pstatus'][Pstatus],
            'Medu':       Medu,
            'Fedu':       Fedu,
            'Mjob':       ENCODING['Mjob'][Mjob],
            'Fjob':       ENCODING['Fjob'][Fjob],
            'reason':     ENCODING['reason'][reason],
            'guardian':   ENCODING['guardian'][guardian],
            'traveltime': traveltime,
            'studytime':  studytime,
            'failures':   failures,
            'schoolsup':  ENCODING['schoolsup'][schoolsup],
            'famsup':     ENCODING['famsup'][famsup],
            'paid':       ENCODING['paid'][paid],
            'activities': ENCODING['activities'][activities],
            'nursery':    ENCODING['nursery'][nursery],
            'higher':     ENCODING['higher'][higher],
            'internet':   ENCODING['internet'][internet],
            'romantic':   ENCODING['romantic'][romantic],
            'famrel':     famrel,
            'freetime':   freetime,
            'goout':      goout,
            'Dalc':       Dalc,
            'Walc':       Walc,
            'health':     health,
            'absences':   absences,
        }

        # Match column order exactly, then apply the same scaler used in training
        input_df     = pd.DataFrame([input_dict])[FEATURE_ORDER]
        input_scaled = scaler.transform(input_df)

        model      = models[model_choice]
        prediction = model.predict(input_scaled)[0]

        st.markdown("---")
        st.markdown("### Result")
        r1, r2 = st.columns([1, 2])

        with r1:
            if prediction == 1:
                st.success("## ✅ PASS")
                st.markdown("This student is likely to **pass** their final exam.")
            else:
                st.error("## ❌ AT RISK")
                st.markdown("This student is at **risk of failing**. Early support is recommended.")

        with r2:
            if hasattr(model, 'predict_proba'):
                proba     = model.predict_proba(input_scaled)[0]
                pass_prob = proba[1] * 100
                fail_prob = proba[0] * 100

                st.markdown("**Prediction Probabilities**")
                fig, ax = plt.subplots(figsize=(5, 2))
                ax.barh(['Pass', 'Fail'], [pass_prob, fail_prob], color=['#2ecc71', '#e74c3c'])
                ax.set_xlim(0, 100)
                ax.set_xlabel('Probability (%)')
                for i, val in enumerate([pass_prob, fail_prob]):
                    ax.text(val + 1, i, f'{val:.1f}%', va='center', fontsize=10)
                ax.spines[['top', 'right']].set_visible(False)
                plt.tight_layout()
                st.pyplot(fig)
            else:
                st.info(f"Model **{model_choice}** does not output probabilities.")

        # Automatic risk factor summary based on known research findings
        st.markdown("---")
        st.markdown("### ⚠️ Risk Factor Summary")
        risks = []
        if failures > 0:
            risks.append(f"**{failures} past failure(s)** — the strongest single predictor of future failure")
        if studytime <= 1:
            risks.append("**Very low study time** — less than 2 hours per week")
        if absences > 10:
            risks.append(f"**High absence count** — {absences} absences")
        if Dalc >= 3:
            risks.append(f"**Elevated weekday alcohol use** — score {Dalc}/5")
        if higher == "no":
            risks.append("**No aspiration for higher education** — strongly correlated with lower performance")

        if risks:
            for r in risks:
                st.markdown(f"- {r}")
        else:
            st.markdown("✅ No major risk factors detected for this student profile.")


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 3 — DATA EXPLORER
# ═════════════════════════════════════════════════════════════════════════════
elif page == "📊 Data Explorer":

    st.title("📊 Data Explorer")
    st.markdown(f"Exploring the **{subject}** dataset — {len(df_raw)} students, {df_raw.shape[1]} columns.")

    tab1, tab2, tab3 = st.tabs(["Overview", "Feature Distributions", "Correlation Heatmap"])

    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**First 10 rows**")
            st.dataframe(df_raw.head(10), use_container_width=True)
        with c2:
            st.markdown("**Summary Statistics**")
            st.dataframe(df_raw.describe().round(2), use_container_width=True)

        st.markdown("---")
        st.markdown("**Pass / Fail Distribution**")
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
        counts = df_full['pass'].value_counts()

        axes[0].bar(['Fail', 'Pass'], [counts.get(0, 0), counts.get(1, 0)],
                    color=['#e74c3c', '#2ecc71'], edgecolor='white')
        axes[0].set_ylabel('Number of Students')
        axes[0].set_title('Count')

        axes[1].pie([counts.get(0, 0), counts.get(1, 0)],
                    labels=['Fail', 'Pass'], colors=['#e74c3c', '#2ecc71'],
                    autopct='%1.1f%%', startangle=90)
        axes[1].set_title('Proportion')

        plt.tight_layout()
        st.pyplot(fig)

    with tab2:
        feature = st.selectbox("Select a feature to explore", X.columns.tolist())

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # Overall distribution
        axes[0].hist(X[feature], bins=15, color='#3498db', edgecolor='white')
        axes[0].set_title(f'{feature} — Overall Distribution')
        axes[0].set_xlabel(feature)
        axes[0].set_ylabel('Count')

        # Split by outcome — shows whether the feature separates Pass from Fail
        pass_vals = X[feature][y == 1]
        fail_vals = X[feature][y == 0]
        axes[1].hist([fail_vals, pass_vals], bins=15,
                     label=['Fail', 'Pass'], color=['#e74c3c', '#2ecc71'],
                     edgecolor='white', alpha=0.75)
        axes[1].set_title(f'{feature} — By Outcome')
        axes[1].set_xlabel(feature)
        axes[1].legend()

        plt.tight_layout()
        st.pyplot(fig)

        # Pass rate at each unique value of the feature
        st.markdown(f"**Pass rate by {feature} value**")
        explore      = pd.DataFrame({'val': X[feature].values, 'pass': y.values})
        pass_by_val  = explore.groupby('val')['pass'].mean() * 100
        overall_rate = y.mean() * 100

        fig2, ax2 = plt.subplots(figsize=(8, 3.5))
        ax2.bar(pass_by_val.index.astype(str), pass_by_val.values, color='#3498db')
        ax2.axhline(overall_rate, color='red', linestyle='--',
                    label=f'Overall pass rate ({overall_rate:.1f}%)')
        ax2.set_xlabel(feature)
        ax2.set_ylabel('Pass Rate (%)')
        ax2.set_ylim(0, 100)
        ax2.set_title(f'Pass Rate by {feature}')
        ax2.legend()
        plt.tight_layout()
        st.pyplot(fig2)

    with tab3:
        st.markdown("""
**How to read this heatmap:**
- Values range from **-1** (perfect negative) to **+1** (perfect positive)
- Find the `pass` row/column — strongest colours = best predictors of outcome
- Features that strongly correlate with each other carry redundant information
        """)

        full_num = X.copy()
        full_num['pass'] = y.values
        corr = full_num.corr()

        # Show lower triangle only — matrix is symmetric, upper half is redundant
        mask = np.triu(np.ones_like(corr, dtype=bool), k=1)

        fig, ax = plt.subplots(figsize=(14, 11))
        sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm',
                    mask=mask, ax=ax, annot_kws={'size': 7}, linewidths=0.5)
        ax.set_title('Correlation Matrix', fontsize=14)
        plt.tight_layout()
        st.pyplot(fig)

        # Top 10 features most correlated with the target
        st.markdown("**Top 10 features by correlation with pass/fail**")
        pass_corr = corr['pass'].drop('pass').abs().sort_values(ascending=False).head(10)

        fig3, ax3 = plt.subplots(figsize=(7, 4))
        ax3.barh(pass_corr.index[::-1], pass_corr.values[::-1], color='#9b59b6')
        ax3.set_xlabel('|Correlation| with pass')
        ax3.set_title('Top 10 Features by Correlation with Outcome')
        plt.tight_layout()
        st.pyplot(fig3)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 4 — MODEL COMPARISON
# ═════════════════════════════════════════════════════════════════════════════
elif page == "🤖 Model Comparison":

    st.title("🤖 Model Comparison")

    if not models:
        st.warning("⚠️ No saved models found. Run the notebook first.")
        st.stop()
    if scaler is None:
        st.warning("⚠️ Scaler not found. Run the notebook first.")
        st.stop()

    # Apply the same scaler and split as training
    X_scaled = scaler.transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    results = []
    with st.spinner("Evaluating all models..."):
        for name, model in models.items():
            y_pred = model.predict(X_test)
            cv_f1  = cross_val_score(model, X_scaled, y, cv=cv, scoring='f1')
            results.append({
                'Model':      name,
                'Accuracy':   accuracy_score(y_test, y_pred),
                'Precision':  precision_score(y_test, y_pred, zero_division=0),
                'Recall':     recall_score(y_test, y_pred, zero_division=0),
                'F1 Score':   f1_score(y_test, y_pred, zero_division=0),
                'CV F1 Mean': cv_f1.mean(),
                'CV F1 Std':  cv_f1.std(),
            })

    results_df = pd.DataFrame(results).set_index('Model')

    st.markdown("### Metrics Table")
    st.markdown("Green = best value per column. For CV F1 Std, lower is better (more stable).")
    st.dataframe(
        results_df.style
            .highlight_max(axis=0, color='#d4edda',
                           subset=['Accuracy', 'Precision', 'Recall', 'F1 Score', 'CV F1 Mean'])
            .highlight_min(axis=0, color='#d4edda', subset=['CV F1 Std'])
            .format("{:.4f}"),
        use_container_width=True
    )

    st.markdown("---")
    tab1, tab2, tab3 = st.tabs(["F1 Comparison", "Confusion Matrices", "ROC Curves"])

    with tab1:
        fig, ax = plt.subplots(figsize=(9, 4))
        names = results_df.index.tolist()
        x     = np.arange(len(names))
        w     = 0.35

        # Grouped bars: test F1 vs cross-validation F1 (with std error bars)
        bars1 = ax.bar(x - w/2, results_df['F1 Score'],   w, label='Test F1',
                       color='#3498db', edgecolor='white')
        ax.bar(x + w/2, results_df['CV F1 Mean'], w, label='CV F1 (mean)',
               color='#2ecc71', edgecolor='white',
               yerr=results_df['CV F1 Std'], capsize=4, error_kw={'elinewidth': 1.5})

        ax.set_ylabel('F1 Score')
        ax.set_title('Test F1 vs Cross-Validation F1\n(error bars = std across 5 folds)')
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=15, ha='right')
        ax.set_ylim(0, 1)
        ax.legend()
        ax.spines[['top', 'right']].set_visible(False)

        for bar in bars1:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=8)
        plt.tight_layout()
        st.pyplot(fig)

        st.markdown("""
**Reading this chart:**
- A model with high Test F1 but much lower CV F1 may be overfitting
- Small error bars = stable model that generalises well across different data splits
        """)

    with tab2:
        model_items  = list(models.items())
        cols_per_row = 3

        for row_start in range(0, len(model_items), cols_per_row):
            cols = st.columns(cols_per_row)
            for col_idx, (name, model) in enumerate(model_items[row_start:row_start + cols_per_row]):
                y_pred = model.predict(X_test)
                cm     = confusion_matrix(y_test, y_pred)
                with cols[col_idx]:
                    fig, ax = plt.subplots(figsize=(3.5, 3))
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                                xticklabels=['Fail', 'Pass'], yticklabels=['Fail', 'Pass'])
                    ax.set_title(name, fontsize=10)
                    ax.set_ylabel('Actual')
                    ax.set_xlabel('Predicted')
                    plt.tight_layout()
                    st.pyplot(fig)

        st.markdown("""
**Reading the confusion matrix:**
- **Top-left (TN):** Correctly predicted Fail
- **Bottom-right (TP):** Correctly predicted Pass
- **Top-right (FP):** Predicted Pass, actually Failed — false alarm
- **Bottom-left (FN):** Predicted Fail, actually Passed — the costly miss
        """)

    with tab3:
        fig, ax = plt.subplots(figsize=(8, 6))
        colors  = ['#3498db', '#2ecc71', '#e74c3c', '#9b59b6', '#f39c12', '#1abc9c']

        for (name, model), color in zip(models.items(), colors):
            if hasattr(model, 'predict_proba'):
                y_prob      = model.predict_proba(X_test)[:, 1]
                fpr, tpr, _ = roc_curve(y_test, y_prob)
                roc_auc     = auc(fpr, tpr)
                ax.plot(fpr, tpr, color=color, linewidth=2,
                        label=f'{name} (AUC = {roc_auc:.3f})')

        # Diagonal = random guessing — a useful model must sit above this line
        ax.plot([0, 1], [0, 1], 'k--', label='Random Classifier (AUC = 0.500)')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curves — All Models')
        ax.legend(loc='lower right', fontsize=9)
        ax.spines[['top', 'right']].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)

        st.markdown("""
**Reading the ROC curve:**
- Curve closer to the **top-left** = better model
- **AUC = 1.0** = perfect | **AUC = 0.5** = random guessing
- This shows performance across ALL decision thresholds, not just the default 0.5
        """)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 5 — FEATURE INSIGHTS
# ═════════════════════════════════════════════════════════════════════════════
elif page == "💡 Feature Insights":

    st.title("💡 Feature Insights")

    if 'Random Forest' not in models:
        st.warning("⚠️ Random Forest model not found. Run the notebook first.")
        st.stop()

    rf = models['Random Forest']
    tab1, tab2 = st.tabs(["Feature Importance", "Pass Rate by Feature"])

    with tab1:
        st.markdown("""
**Feature importance** measures how much each attribute reduced prediction error
across all trees in the Random Forest. Higher = more influential in predictions.
        """)

        importances = rf.feature_importances_
        feat_df = pd.DataFrame({
            'Feature':    X.columns,
            'Importance': importances
        }).sort_values('Importance', ascending=True)

        # Highlight top 5 features in red
        colors = ['#e74c3c' if i >= len(feat_df) - 5 else '#3498db'
                  for i in range(len(feat_df))]

        fig, ax = plt.subplots(figsize=(8, 9))
        ax.barh(feat_df['Feature'], feat_df['Importance'], color=colors)
        ax.set_xlabel('Importance Score')
        ax.set_title('Feature Importance — Random Forest\n(red = top 5 most influential features)')
        ax.spines[['top', 'right']].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)

        st.markdown("### Top 5 Most Influential Features")
        top5 = feat_df.tail(5)[::-1]
        for rank, (_, row) in enumerate(top5.iterrows(), 1):
            st.markdown(f"**{rank}. {row['Feature']}** — importance score: `{row['Importance']:.4f}`")

    with tab2:
        st.markdown("Select a feature to see how its values relate to the pass rate.")
        feature = st.selectbox("Feature", X.columns.tolist(), key="insights_feature")

        explore      = pd.DataFrame({'val': X[feature].values, 'pass': y.values})
        pass_by_val  = explore.groupby('val')['pass'].mean() * 100
        count_by_val = explore.groupby('val')['pass'].count()
        overall_rate = y.mean() * 100

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # Pass rate per feature value vs the overall average (red dashed line)
        axes[0].bar(pass_by_val.index.astype(str), pass_by_val.values, color='#3498db')
        axes[0].axhline(overall_rate, color='red', linestyle='--',
                        label=f'Overall ({overall_rate:.1f}%)')
        axes[0].set_title(f'Pass Rate by {feature}')
        axes[0].set_xlabel(feature)
        axes[0].set_ylabel('Pass Rate (%)')
        axes[0].set_ylim(0, 100)
        axes[0].legend()

        # Student count per value — gives context (a 100% pass rate with 1 student isn't meaningful)
        axes[1].bar(count_by_val.index.astype(str), count_by_val.values, color='#95a5a6')
        axes[1].set_title(f'Student Count by {feature}')
        axes[1].set_xlabel(feature)
        axes[1].set_ylabel('Number of Students')

        plt.tight_layout()
        st.pyplot(fig)

        st.markdown("**Summary table**")
        summary = pd.DataFrame({
            'Value':         pass_by_val.index,
            'Pass Rate (%)': pass_by_val.values.round(1),
            'Student Count': count_by_val.values,
        })
        st.dataframe(summary, use_container_width=True)
