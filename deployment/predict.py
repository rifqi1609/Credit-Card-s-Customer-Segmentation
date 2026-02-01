# Import libraries
import pandas as pd                                       
import numpy as np
import dill                                             
import streamlit as st

# Load Model
with open('deployment/kmeans_model.pkl', 'rb') as f:
    model_loaded = dill.load(f)

def run():
    st.title('Customer Segmetation')
    st.markdown('## Input Single Data')
   
    # Form
    with st.form('form_input'):
        CUST_ID = st.number_input('Input Customer ID', value=0, step=1)
        BALANCE = st.number_input('Input Balance', value=0.0, step=0.1, format="%.1f")
        BALANCE_FREQUENCY = st.number_input('Input Balance Frequency', value=0.0, step=0.1, format="%.1f")
        PURCHASES = st.number_input('Input Purchases', value=0.0, step=0.1, format="%.1f")
        ONEOFF_PURCHASES = st.number_input('Input One-Off Purchases', value=0.0, step=0.1, format="%.1f")
        INSTALLMENTS_PURCHASES = st.number_input('Input Installments Purchases', value=0.0, step=0.1, format="%.1f")
        CASH_ADVANCE = st.number_input('Input Cash Advance', value=0.0, step=0.1, format="%.1f")
        PURCHASES_FREQUENCY = st.number_input('Input Purchases Frequency', value=0.0, step=0.1, format="%.1f")
        ONEOFF_PURCHASES_FREQUENCY = st.number_input('Input One-Off Purchases Frequency', value=0.0, step=0.1, format="%.1f") 
        PURCHASES_INSTALLMENTS_FREQUENCY = st.number_input('Input Purchases Installments Frequency', value=0.0, step=0.1, format="%.1f") 
        CASH_ADVANCE_FREQUENCY = st.number_input('Input Cash Advance Frequency', value=0.0, step=0.1, format="%.1f") 
        CASH_ADVANCE_TRX = st.number_input('Input Cash Advance Transaction', value=0, step=1) 
        PURCHASES_TRX = st.number_input('Input Purchases Transaction', value=0, step=1) 
        CREDIT_LIMIT = st.number_input('Input Credit Limit', value=0.0, step=0.1, format="%.1f") 
        PAYMENTS = st.number_input('Input Payments', value=0.0, step=0.1, format="%.1f") 
        MINIMUM_PAYMENTS = st.number_input('Input Minimum Payments', value=0.0, step=0.1, format="%.1f") 
        PRC_FULL_PAYMENT = st.number_input('Input Percentage of Full Payment', value=0.0, step=0.1, format="%.1f") 
        TENURE = st.number_input('Input Tenure', value=0, step=1) 
        submit_btn = st.form_submit_button('Predict')

    if submit_btn:
        df_inf={
                'CUST_ID': CUST_ID,
                'BALANCE': BALANCE,
                'BALANCE_FREQUENCY': BALANCE_FREQUENCY,
                'PURCHASES': PURCHASES,
                'ONEOFF_PURCHASES': ONEOFF_PURCHASES,
                'INSTALLMENTS_PURCHASES': INSTALLMENTS_PURCHASES,
                'CASH_ADVANCE': CASH_ADVANCE,
                'PURCHASES_FREQUENCY': PURCHASES_FREQUENCY, 
                'ONEOFF_PURCHASES_FREQUENCY': ONEOFF_PURCHASES_FREQUENCY,
                'PURCHASES_INSTALLMENTS_FREQUENCY': PURCHASES_INSTALLMENTS_FREQUENCY,
                'CASH_ADVANCE_FREQUENCY': CASH_ADVANCE_FREQUENCY,
                'CASH_ADVANCE_TRX': CASH_ADVANCE_TRX,
                'PURCHASES_TRX': PURCHASES_TRX,
                'CREDIT_LIMIT': CREDIT_LIMIT,
                'PAYMENTS': PAYMENTS,
                'MINIMUM_PAYMENTS': MINIMUM_PAYMENTS,
                'PRC_FULL_PAYMENT': PRC_FULL_PAYMENT,
                'TENURE': TENURE}
        df_inf = pd.DataFrame([df_inf])

        # Predict
        y_pred_inf = model_loaded.predict(df_inf)

        # Display Data
        df_inf['CLUSTER']=y_pred_inf
        st.success("Prediction Complete!")
        st.dataframe(df_inf)
        cluster_result = df_inf['CLUSTER'].iloc[0]
        st.write(f"The segment of this customer is **{cluster_result}**")

if __name__ == '__main__':
    run()


def run_file():
    st.markdown('## Input Multiple Data')

    # 2. Upload File CSV
    uploaded_file = st.file_uploader("Upload Data", type=["csv"])

    if uploaded_file is not None:
        # Load New Data
        df_inf = pd.read_csv(uploaded_file)

        # Tombol Prediksi
        if st.button('Predict All'):
            try:
                # Prediksi
                y_pred_inf = model_loaded.predict(df_inf)

                # Display Data
                df_inf['CLUSTER']=y_pred_inf

                # Menampilkan Hasil
                st.success("Prediction Complete!")
                st.dataframe(df_inf)

                # Tombol Download Hasil
                csv = df_inf.to_csv(index=False).encode('utf-8')
                st.download_button("Download Prediction", csv, "prediction.csv", "text/csv")
            
            except Exception as e:
                st.error(f"Error: {e}")

if __name__ == '__main__':
    run_file()

