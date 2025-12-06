# streamlit_app/app.py
# Advanced Streamlit app: manual + CSV upload + SHAP generation + dashboard + ROI
# Requirements: streamlit, pandas, numpy, joblib, shap, matplotlib, scikit-learn, xgboost, pillow

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
from pathlib import Path
import os
import time
import warnings
from PIL import Image

warnings.filterwarnings("ignore")
plt.style.use("default")

# -------------------------
# CONFIG: change ROOT if needed
# -------------------------
ROOT = Path("D:/Major Project/models")                # <<-- change this to your project path
MODEL_PATH = ROOT / "xgboost_model.pkl"
SCALER_PATH = ROOT / "scaler.pkl"
ENCODERS_PATH = ROOT / "label_encoders.pkl"
FEATURES_PATH = ROOT / "feature_columns.txt"   # optional; fallback provided
ASSETS_DIR = Path("D:/Major Project/streamlit_assets")        # where app saves/reads SHAP images
TMP_DIR =  Path("D:/Major Project/streamlit_tmp")

# ensure folders exist
ASSETS_DIR.mkdir(parents=True, exist_ok=True)
TMP_DIR.mkdir(parents=True, exist_ok=True)

###LOGO

LOGO_PATH = "D:/Major Project/streamlit_app/logo.png"
logo = None
if os.path.exists(LOGO_PATH):
    try:
        logo = Image.open(LOGO_PATH)
    except:
        logo = None


###PAGE CONFIG

col1, col2 = st.columns([1, 5])

with col1:
    st.image(logo, width=120)   # <---- LEFT SIDE LOGO

with col2:
    st.title("Telecom Customer Churn Prediction")

st.write("""
        Welcome to the Churn Prediction System.
        """)

# -------------------------
# UI: Custom CSS + Dark Mode toggle + Logo
# -------------------------
custom_css = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Poppins', sans-serif;
}

/* card */
.section-card {
    padding: 16px;
    border-radius: 12px;
    box-shadow: var(--shadow);
    background: var(--card-bg);
    margin-bottom: 16px;
}

/* tabs */
.stTabs [role="tablist"] button {
    padding: 0.6rem 1rem;
    border-radius: 10px;
    margin-right: 6px;
    font-weight: 600;
}

/* subtle metric styling */
.css-1q8dd3e .stMetric > div {
    background: transparent;
}
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)


dark_css = """
<style>
/* Global dark mode */
body, .stApp {
    background-color: #0E1117 !important;
    color: white !important;
}



/* Inputs, dropdowns, textboxes */
div[data-baseweb="input"], div[data-baseweb="select"] {
    background-color: #1A1D23 !important;
    color: white !important;
    border: 1px solid #444 !important;
}

/* Table */
.dataframe {
    color: white !important;
}

/* Buttons */
.stButton>button {
    background-color: #3B82F6 !important;
    color: white !important;
    border-radius: 6px;
    border: none;
}

.stButton>button:hover {
    background-color: #1E40AF !important;
}
</style>
"""



# -------------------------
# Helpers: load artifacts robustly
# -------------------------
@st.cache_resource
def load_pickle(path):
    return joblib.load(path)

def load_artifacts():
    model = scaler = encoders = feature_cols = None
    load_errors = []
    try:
        model = load_pickle(MODEL_PATH)
    except Exception as e:
        load_errors.append(f"Model load error: {e}")
    try:
        scaler = load_pickle(SCALER_PATH)
    except Exception as e:
        load_errors.append(f"Scaler load error: {e}")
    try:
        encoders = load_pickle(ENCODERS_PATH)
    except Exception as e:
        load_errors.append(f"Encoders load error: {e}")
    # feature columns
    if FEATURES_PATH.exists():
        try:
            feature_cols = [ln.strip() for ln in open(FEATURES_PATH).read().splitlines() if ln.strip()]
        except Exception as e:
            load_errors.append(f"Feature file read error: {e}")
    else:
        # try to infer from model if available (scikit-learn models often have feature_names_in_)
        try:
            if hasattr(model, "feature_names_in_"):
                feature_cols = list(model.feature_names_in_)
        except Exception:
            feature_cols = None
    return model, scaler, encoders, feature_cols, load_errors

model, scaler, encoders, feature_cols, load_errors = load_artifacts()



# -------------------------
# Utility: encode df using saved label encoders dict
# encoders should be dict {col: LabelEncoder()}
# -------------------------
def apply_label_encoders(df, encoders):
    df_enc = df.copy()
    for col, le in encoders.items():
        if col in df_enc.columns:
            # safe transform: handle unseen categories by mapping to -1 (or most common)
            try:
                df_enc[col] = le.transform(df_enc[col].astype(str))
            except Exception:
                # fallback: create mapping from classes_
                classes = list(le.classes_)
                mapping = {c: i for i, c in enumerate(classes)}
                df_enc[col] = df_enc[col].map(mapping).fillna(-1).astype(int)
    return df_enc

# -------------------------
# SHAP helpers (safe)
# (same as your original function; preserved)
# -------------------------
def compute_shap_and_save(model, X_scaled, X_original, save_prefix="shap"):
    outputs = {}
    try:
        explainer = shap.TreeExplainer(model)
    except Exception:
        # fallback (slower)
        explainer = shap.KernelExplainer(model.predict_proba, shap.sample(X_original, 100))

    MAX_ROWS = 2000
    if isinstance(X_original, pd.DataFrame):
        n_rows = len(X_original)
    else:
        n_rows = X_scaled.shape[0]

    if n_rows > MAX_ROWS:
        sample_idx = np.random.choice(range(n_rows), size=MAX_ROWS, replace=False)
        X_sample = X_original.iloc[sample_idx] if isinstance(X_original, pd.DataFrame) else X_original[sample_idx]
        X_scaled_sample = X_scaled[sample_idx] if isinstance(X_scaled, np.ndarray) else X_scaled.iloc[sample_idx]
    else:
        X_sample = X_original
        X_scaled_sample = X_scaled

    try:
        shap_values = explainer.shap_values(X_scaled_sample if hasattr(X_scaled_sample, "shape") else X_sample)
        if isinstance(shap_values, list) and len(shap_values) == 2:
            shap_vals_pos = shap_values[1]
        elif isinstance(shap_values, list):
            shap_vals_pos = shap_values[0]
        else:
            shap_vals_pos = shap_values
    except Exception:
        shap_vals_pos = explainer.shap_values(X_sample)
        if isinstance(shap_vals_pos, list) and len(shap_vals_pos) == 2:
            shap_vals_pos = shap_vals_pos[1]

    # summary (beeswarm)
    try:
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_vals_pos, X_sample, show=False, max_display=20)
        path = ASSETS_DIR / f"{save_prefix}_summary_beeswarm.png"
        plt.title("SHAP Summary (beeswarm)")
        plt.tight_layout()
        plt.savefig(path, dpi=200, bbox_inches="tight", facecolor='white')
        plt.close()
        outputs['beeswarm'] = str(path)
    except Exception as e:
        outputs['beeswarm_error'] = str(e)

    # importance bar
    try:
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_vals_pos, X_sample, plot_type="bar", show=False, max_display=20)
        path = ASSETS_DIR / f"{save_prefix}_importance_bar.png"
        plt.title("SHAP Feature Importance")
        plt.tight_layout()
        plt.savefig(path, dpi=200, bbox_inches="tight", facecolor='white')
        plt.close()
        outputs['importance_bar'] = str(path)
    except Exception as e:
        outputs['importance_bar_error'] = str(e)

    # dependence top3
    try:
        mean_abs = np.abs(shap_vals_pos).mean(axis=0)
        top_idx = np.argsort(mean_abs)[-3:][::-1]
        top_feats = list(X_original.columns[top_idx])
        fig, axes = plt.subplots(1, len(top_feats), figsize=(6*len(top_feats), 4))
        if len(top_feats) == 1:
            axes = [axes]
        for ax, feat in zip(axes, top_feats):
            shap.dependence_plot(feat, shap_vals_pos, X_sample, show=False, ax=ax)
        path = ASSETS_DIR / f"{save_prefix}_dependence_top3.png"
        plt.tight_layout()
        plt.savefig(path, dpi=200, bbox_inches="tight", facecolor='white')
        plt.close()
        outputs['dependence_top3'] = str(path)
    except Exception as e:
        outputs['dependence_error'] = str(e)

    # waterfall for top-risk
    try:
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X_scaled_sample)[:, 1]
            top_idx_local = int(np.argmax(probs))
            try:
                base = explainer.expected_value
                single_shap = shap_vals_pos[top_idx_local]
                explanation = shap.Explanation(values=single_shap,
                                               base_values=base,
                                               data=X_sample.iloc[top_idx_local],
                                               feature_names=X_sample.columns)
                plt.figure(figsize=(8, 6))
                shap.waterfall_plot(explanation, show=False, max_display=10)
                path = ASSETS_DIR / f"{save_prefix}_waterfall_toprisk.png"
                plt.tight_layout()
                plt.savefig(path, dpi=200, bbox_inches="tight", facecolor='white')
                plt.close()
                outputs['waterfall_toprisk'] = str(path)
            except Exception as e:
                outputs['waterfall_error'] = str(e)
    except Exception as e:
        outputs['waterfall_error2'] = str(e)

    return outputs

# -------------------------
# Layout: use tabs for navigation (keeps same pages)
# -------------------------
tab_home, tab_single, tab_batch, tab_dashboard, tab_roi, tab_about = st.tabs(
    ["Home", "Predict (single)", "Batch Predict (CSV + SHAP)", "Dashboard", "ROI Calculator", "About"]
)

# -------------------------
# HOME
# -------------------------


with tab_home:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown("""
    ##  Do business the smart way with churn prediction

    Companies often have a scatter-gun approach when it comes to dealing with churn—setting prices as low as possible, gunning for the maximum amount of customer growth possible per month—everything but carefully assessing the factors behind your churn, and making this data work for you.

    Ultra-aggressive selling techniques might be effective in the short-term, but you're treating the symptoms, not the illness.

    A sensible approach to churn prediction—through an integration or your own solution—will allow you to understand the reasons a user might churn and respond to them.

    A user’s happiness with your product can hinge on the most minor-seeming of conditions. Embracing a predictive solution to handling your churn will give you clarity on each of them, and ultimately help you beat the churn.
    """)

    st.image("D:\Major Project\images\customer_churn.jpeg")

    st.markdown("""
    ---

    ##  Identify customers with high churn risk

    Churn occurs for a variety of reasons. To better understand why a customer has churned, it is key to have proper customer segmentation.

    A user’s churn probability depends on their overall profile, their customer behavior when using your product, and their needs. A customer’s needs may change over the period of their subscription—not all churn happens in the first few months.
    """)

    st.markdown("""
    ---

    ##  Engage customers before they churn

    Harnessed properly, churn prediction can be a major asset in getting a clearer picture of your customers’ experience with your product.

    Although the range of potential factors behind churn can be complex, stopping churn often revolves around a tailored approach to improving customer experience. Churn prediction gives you the chance to improve a customer’s experience before they leave for good.
    """)

    st.image("D:\Major Project\images\photo.avif")

    st.markdown("""
    ---

    #  ROI Calculator — Make Better Financial Decisions

    An ROI (Return on Investment) calculator is a financial tool that helps you measure the profitability of your investment. It calculates the percentage gain or loss relative to the money you invested, making it easier to understand the effectiveness of your spending.

    ### How It Works
    - **Input Initial Investment Cost:** The total amount of money spent on a project, campaign, or asset (in INR).  
    - **Input Returns or Gains:** The money earned or generated from the investment (in INR).  
    - **Calculate Net Profit:**  
    `Net Profit = Returns − Investment Cost`  
    - **Calculate ROI:**  
    `ROI = (Net Profit ÷ Investment Cost) × 100`

    Once you enter the inputs, the calculator automatically shows the result using this formula.
    """)

    st.image("D:\Major Project\images\photo2.avif")

    st.markdown("""
    ---

    ##  Benefits of Using ROI Calculator

    - Shows **absolute return** and **annualized return**.  
    - Helps you pick the right investment based on goals and risk tolerance.  
    - Helps evaluate performance over different periods.  
    - Compares the worth of an investment against benchmarks.  
    - Measures profit against the cost of investment.
    """)


# -------------------------
# Predict single customer (manual)
# -------------------------
with tab_single:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.header("Predict for a single customer (manual input)")
    if model is None or scaler is None or encoders is None:
        st.error("Model/scaler/encoders not loaded — place required files in ROOT.")
    else:
        # choose features order: try feature_cols else build from encoders + numeric guesses
        if feature_cols is None:
            feature_cols = ["gender","SeniorCitizen","Partner","Dependents","tenure",
                            "PhoneService","MultipleLines","InternetService","OnlineSecurity",
                            "OnlineBackup","DeviceProtection","TechSupport","StreamingTV",
                            "StreamingMovies","Contract","PaperlessBilling","PaymentMethod",
                            "MonthlyCharges","TotalCharges"]
        with st.form("single_form"):
            st.subheader("Categorical fields")
            user_vals = {}
            for col in encoders.keys():
                opt = list(encoders[col].classes_)
                try:
                    default_idx = 0
                except:
                    default_idx = 0
                user_vals[col] = st.selectbox(col, opt, index=default_idx)
            st.subheader("Numeric fields")
            user_vals["tenure"] = st.number_input("tenure (months)", min_value=0, max_value=72, value=12)
            user_vals["MonthlyCharges"] = st.number_input("MonthlyCharges", min_value=0.0, value=50.0)
            user_vals["TotalCharges"] = st.number_input("TotalCharges", min_value=0.0, value=600.0)
            submitted = st.form_submit_button("Predict Churn")
        if submitted:
            # create df single (use feature_cols to maintain order when possible)
            df_user = pd.DataFrame([user_vals])
            # align columns with feature_cols if possible
            try:
                df_user = df_user.reindex(columns=feature_cols).fillna(0)
            except:
                pass
            df_enc = apply_label_encoders(df_user, encoders)
            X_scaled = scaler.transform(df_enc)
            prob = model.predict_proba(X_scaled)[:,1][0]
            pred = model.predict(X_scaled)[0]
            st.metric("Churn probability", f"{prob:.2%}")
            st.write("Class:", "Churn" if pred==1 else "No churn")

            # Option: compute SHAP for this single instance
            if st.button("Generate SHAP for this single sample"):
                with st.spinner("Computing SHAP for single sample..."):
                    out = compute_shap_and_save(model, X_scaled, df_enc, save_prefix=f"single_{int(time.time())}")
                st.success("SHAP generation complete.")
                for k,v in out.items():
                    if isinstance(v,str) and v.endswith(".png"):
                        st.image(v, caption=k, use_column_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# -------------------------
# Batch Predict (CSV upload) + SHAP generation
# -------------------------
with tab_batch:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.header("Batch predict: upload CSV, predict + generate SHAP")
    st.write("CSV must contain the same feature columns as training (raw values).")

    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded:
        try:
            df_in = pd.read_csv(uploaded)
            st.write("Preview:")
            st.dataframe(df_in.head())
            if encoders is None or scaler is None or model is None:
                st.error("Model/scaler/encoders missing.")
            else:
                run_and_shap = st.checkbox("Also generate SHAP plots after prediction (may take time)", value=False)
                if st.button("Run batch predict"):
                    with st.spinner("Encoding, scaling, predicting..."):
                        df_enc = apply_label_encoders(df_in, encoders)
                        for col in ["tenure","MonthlyCharges","TotalCharges","ChargesPerMonth"]:
                            if col not in df_enc.columns:
                                df_enc[col] = 0
                        if feature_cols is None:
                            X_for_model = df_enc
                        else:
                            X_for_model = df_enc.reindex(columns=feature_cols).fillna(0)
                        X_scaled = scaler.transform(X_for_model)
                        probs = model.predict_proba(X_scaled)[:,1]
                        preds = model.predict(X_scaled)
                        df_results = df_in.copy()
                        df_results["churn_prob"] = probs
                        df_results["prediction"] = preds
                        ts = int(time.time())
                        res_path = TMP_DIR / f"batch_results_{ts}.csv"
                        df_results.to_csv(res_path, index=False)
                        st.success(f"Predictions saved to {res_path}")
                    if run_and_shap:
                        with st.spinner("Computing SHAP plots (may take several minutes)..."):
                            out = compute_shap_and_save(model, X_scaled, X_for_model, save_prefix=f"batch_{ts}")
                        st.success("SHAP plots generated and saved to assets.")
                        for k,v in out.items():
                            if isinstance(v,str) and v.endswith(".png"):
                                st.image(v, caption=k, use_column_width=True)
        except Exception as e:
            st.error(f"Error reading CSV: {e}")
    st.markdown('</div>', unsafe_allow_html=True)

# -------------------------
# Dashboard: show images in ASSETS_DIR
# -------------------------
with tab_dashboard:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.header("Dashboard - SHAP images & artifacts")
    st.write(f"Images in {ASSETS_DIR}:")
    imgs = sorted(list(ASSETS_DIR.glob("*.png")), key=os.path.getctime, reverse=True)
    if not imgs:
        st.info("No SHAP images found. Generate them from a prediction run (Batch Predict or Predict single).")
    else:
        cols = st.columns(2)
        for i, img in enumerate(imgs):
            target_col = cols[i % 2]
            try:
                target_col.image(str(img), caption=img.name, use_column_width=True)
            except Exception:
                target_col.write(img.name)
    st.markdown('</div>', unsafe_allow_html=True)

# -------------------------
# ROI Calculator  (AUTO from model predictions)
# -------------------------
with tab_roi:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.header("ROI Calculator (Auto from Model Predictions)")

    batch_files = list(TMP_DIR.glob("batch_results_*.csv"))

    if not batch_files:
        st.warning("No batch prediction results found.\n\nUpload a CSV in *Batch Predict (CSV + SHAP)* first.")
    else:
        latest_file = max(batch_files, key=os.path.getctime)
        st.success(f"Using latest prediction file:\n📄 {latest_file.name}")
        df_pred = pd.read_csv(latest_file)
        if "churn_prob" not in df_pred.columns:
            st.error("Latest file doesn't contain churn_prob column. Re-run batch predictions.")
        else:
            baseline_churn = df_pred["churn_prob"].mean()
            st.metric("Baseline Churn (from model predictions)", f"{baseline_churn:.2%}")
            colA, colB = st.columns(2)
            with colA:
                avg_rev = st.number_input("Avg monthly revenue per customer", value=50.0)
                clv_months = st.number_input("Customer lifetime (months)", value=12)
                offer_cost = st.number_input("Cost per retention offer", value=10.0)
            with colB:
                expected_reduction = st.number_input("Expected churn reduction from offer (%)", value=5.0) / 100
                targeted = len(df_pred)
                st.number_input("Customers targeted (auto)", value=targeted, disabled=True)
            if st.button("Calculate ROI"):
                saved = targeted * baseline_churn * expected_reduction
                revenue_saved = saved * avg_rev * clv_months
                total_cost = targeted * offer_cost
                roi = (revenue_saved - total_cost) / total_cost if total_cost != 0 else float("inf")
                st.metric("Customers Saved (predicted)", f"{int(saved)}")
                st.metric("Revenue Saved", f"${revenue_saved:,.2f}")
                st.metric("ROI", f"{roi:.2%}")
    st.markdown('</div>', unsafe_allow_html=True)

# -------------------------
# About
# -------------------------
with tab_about:
    st.markdown("""
    <div style='text-align: center; max-width: 700px; margin: auto; line-height: 1.6;'>

    <h2 style='color: #333;'>About ChurnGuru</h2>

    <p style='font-size: 16px; color: #555;'>
    ChurnGuru is a smart analytics platform to predict customer churn and help businesses take proactive retention decisions. 
    It also includes an ROI Calculator to measure the impact of retention strategies.
    </p>

    <p style='font-size: 16px; color: #555;'>
    Empowering businesses to understand their customers better and improve loyalty.
    </p>

    <p style='font-size: 14px; color: gray; margin-top: 20px;'>
    Made with ❤️ by <strong>Aastha, Riya, Siddhi</strong>
    </p>

    <p style='font-size: 12px; color: gray; margin-top: 10px;'>
    © 2025 ChurnGuru. All rights reserved. Content and images are for educational and analytical purposes only.
    </p>

    </div>
    """, unsafe_allow_html=True)


    

# End of app