import streamlit as st
import os
import joblib
import numpy as np
import pandas as pd
from math import sqrt
from sklearn.metrics import mean_squared_error
from keras.models import load_model
from keras.layers import BatchNormalization
import plotly.graph_objects as go
import plotly.express as px


st.set_page_config(layout="wide", page_title="PrognosAI Remaining Usefull Life")

MODEL_PATH = "D:/Spring_Board/PrognosAI/Models"

RUL_CLIP_VALUE = 126
SEQ_LEN = 30
WARNING_THRESHOLD = 30
CRITICAL_THRESHOLD = 10

# --- HELPER FUNCTIONS ---

@st.cache_resource
def load_model_assets(dataset_name, model_path):
    try:
        model_file = f"{model_path}/best_{dataset_name}.keras"
        feature_file = f"{model_path}/feature_cols_{dataset_name}.pkl"
        scaler_file = f"{model_path}/scaler_{dataset_name}.pkl"
        
        # Load assets
        model = load_model(model_file, custom_objects={'BatchNormalization': BatchNormalization}) 
        feature_cols = joblib.load(feature_file) 
        scaler = joblib.load(scaler_file)
        return model, feature_cols, scaler
    except Exception as e:
        st.error(f"Error loading model assets for {dataset_name}: {e}")
        return None, None, None

def process_uploaded_data(test_file, rul_file, feature_cols, scaler):
    # Column names definition
    col_names = (["unit_number", "time","setting_1", "setting_2", "setting_3"] + [f"sensor_{i}" for i in range(1, 22)])
    
    try:
        df_test_raw = pd.read_csv(test_file, sep=r"\s+", header=None, names=col_names)
        df_true_rul = pd.read_csv(rul_file, sep=r"\s+", header=None, names=['RUL'])
    except Exception as e:
        st.error(f"Error reading uploaded files: {e}")
        return None, None, None, None

    X_test_list = []
    
    for unit in sorted(df_test_raw["unit_number"].unique()):
        u_raw = df_test_raw[df_test_raw["unit_number"] == unit].copy()
        n_cycles = len(u_raw)
        
        # 1. Scale
        data_to_scale = u_raw[feature_cols]
        u_raw[feature_cols] = scaler.transform(data_to_scale)
        arr = u_raw[feature_cols].values
        
        # 2. Sequence
        if n_cycles >= SEQ_LEN:
            X_test_list.append(arr[-SEQ_LEN:])
        else:
            pad_n = SEQ_LEN - n_cycles
            pad = np.repeat(arr[0:1], pad_n, axis=0)
            X_test_list.append(np.vstack([pad, arr]))
            
    X_test = np.array(X_test_list)
    y_test_true = df_true_rul['RUL'].values
    y_capped = np.clip(y_test_true, a_min=None, a_max=RUL_CLIP_VALUE)
    
    return X_test, y_capped, df_test_raw["unit_number"].unique(), df_true_rul

def get_status(rul_value):
    if rul_value <= CRITICAL_THRESHOLD:
        return "Critical"
    elif rul_value <= WARNING_THRESHOLD:
        return "Maintenance Required"
    else:
        return "Normal"

def main():
    if 'prediction_done' not in st.session_state:
        st.session_state.prediction_done = False

    # --- SIDEBAR ---
    with st.sidebar:
        st.header("Control Panel")
        
        # 1. File Uploads
        uploaded_test_file = st.file_uploader("Upload Test Data (.txt)", type=["txt"])
        uploaded_rul_file = st.file_uploader("Upload RUL Data (.txt)", type=["txt"])
        
        st.markdown("---")
        
        # 2. Dataset Selection
        dataset_name = st.selectbox(
            "Select Data Set Type",
            ("FD001", "FD002", "FD003", "FD004")
        )
        
        st.markdown("---")
        
        # 3. & 4. Predict and Reset Buttons
        col_btn1, col_btn2 = st.columns(2)
        
        with col_btn1:
            predict_btn = st.button("Predict", type="primary", use_container_width=True)
            
        with col_btn2:
            reset_btn = st.button("Reset", use_container_width=True)

    # Reset Logic
    if reset_btn:
        st.session_state.prediction_done = False
        st.rerun()

    # --- MAIN CONTENT AREA ---
    st.title("PrognosAI")
    st.subheader("RUL Prediction")
    # with center():
    #     st.title("Your Centered Title")
    #     st.subheader("Your Centered Subheading")
    if predict_btn and uploaded_test_file and uploaded_rul_file:
        st.session_state.prediction_done = True
        
    if st.session_state.prediction_done and uploaded_test_file and uploaded_rul_file:
        
        with st.spinner('Loading Model and Processing Data...'):
            # Load Assets
            model, feature_cols, scaler = load_model_assets(dataset_name, MODEL_PATH)
            
            if model:
                # Process Data
                X_test, y_true_capped, unit_ids, df_true_raw = process_uploaded_data(
                    uploaded_test_file, uploaded_rul_file, feature_cols, scaler
                )
                
                # Predict
                y_pred = model.predict(X_test).flatten()
                y_pred_clipped = np.clip(y_pred, a_min=0, a_max=RUL_CLIP_VALUE)
                
                # Calculations
                calc_rmse = float(sqrt(mean_squared_error(y_true_capped, y_pred_clipped)))
                
                # Create Result DataFrame
                results_df = pd.DataFrame({
                    "Unit ID": unit_ids,
                    "Actual RUL": y_true_capped,
                    "Predicted RUL": np.round(y_pred_clipped, 2)
                })
                
                # Assign Status
                results_df['Predicted Status'] = results_df['Predicted RUL'].apply(get_status)
                results_df['Actual Status'] = results_df['Actual RUL'].apply(get_status) # For PI Chart 1

                # Counts for Metrics
                status_counts = results_df['Predicted Status'].value_counts()
                count_critical = status_counts.get("Critical", 0)
                count_maint = status_counts.get("Maintenance Required", 0)
                count_normal = status_counts.get("Normal", 0)

                # --- UI: METRICS ROW ---
                st.markdown("### Model Performance Metrics")
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("RMSE", f"{calc_rmse:.4f}")
                m2.metric("No of Critical Units", int(count_critical))
                m3.metric("No of Maint. Req. Units", int(count_maint))
                m4.metric("No of Normal Units", int(count_normal))

                st.markdown("---")

                # --- UI: GRAPH 1 (Line Chart) ---
                st.subheader("Graph 1: Predicted vs Actual RUL")
                fig_line = go.Figure()
                fig_line.add_trace(go.Scatter(y=y_true_capped, mode='lines', name='Actual Capped RUL', line=dict(color='blue')))
                fig_line.add_trace(go.Scatter(y=y_pred_clipped, mode='lines', name='Predicted RUL', line=dict(color='red', dash='dash')))
                fig_line.update_layout(
                    title=f"Predicted vs. Actual RUL on {dataset_name}",
                    xaxis_title="Test Unit Index",
                    yaxis_title="RUL (Cycles)",
                    height=400,
                    margin=dict(l=20, r=20, t=40, b=20)
                )
                st.plotly_chart(fig_line, use_container_width=True)

                st.markdown("---")

                # --- UI: PIE CHARTS ---
                col_pi1, col_pi2 = st.columns(2)

                # Pie Chart 1: Actual Values
                with col_pi1:
                    st.subheader("Pie Chart 1: Actual Values")
                    actual_counts = results_df['Actual Status'].value_counts().reset_index()
                    actual_counts.columns = ['Status', 'Count']
                    fig_pi1 = px.pie(actual_counts, values='Count', names='Status', 
                                     color='Status',
                                     color_discrete_map={
                                         "Critical": "red", 
                                         "Maintenance Required": "orange", 
                                         "Normal": "green"
                                     })
                    st.plotly_chart(fig_pi1, use_container_width=True)

                # Pie Chart 2: Predicted Values
                with col_pi2:
                    st.subheader("Pie Chart 2: Predicted Values")
                    pred_counts = results_df['Predicted Status'].value_counts().reset_index()
                    pred_counts.columns = ['Status', 'Count']
                    fig_pi2 = px.pie(pred_counts, values='Count', names='Status',
                                     color='Status',
                                     color_discrete_map={
                                         "Critical": "red", 
                                         "Maintenance Required": "orange", 
                                         "Normal": "green"
                                     })
                    st.plotly_chart(fig_pi2, use_container_width=True)

                st.markdown("---")

                # --- UI: TABLES ---
                
                # Table 1: Critical Units
                st.subheader("Table 1: Critical Units (Predicted)")
                df_crit = results_df[results_df['Predicted Status'] == "Critical"]
                st.dataframe(df_crit[["Unit ID", "Actual RUL", "Predicted RUL"]], use_container_width=True, hide_index=True)

                # Table 2: Maintenance Required Units
                st.subheader("Table 2: Maintenance Required Units (Predicted)")
                df_warn = results_df[results_df['Predicted Status'] == "Maintenance Required"]
                st.dataframe(df_warn[["Unit ID", "Actual RUL", "Predicted RUL"]], use_container_width=True, hide_index=True)

                # Table 3: Normal Units
                st.subheader("Table 3: Normal Units (Predicted)")
                df_norm = results_df[results_df['Predicted Status'] == "Normal"]
                st.dataframe(df_norm[["Unit ID", "Actual RUL", "Predicted RUL"]], use_container_width=True, hide_index=True)

    elif st.session_state.prediction_done and (not uploaded_test_file or not uploaded_rul_file):
        st.error("Please upload both Test and RUL files before predicting.")
    
    else:
        st.info("Upload files and click 'Predict' to start.")

if __name__ == "__main__":
    main()