# Import libraries
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Untuk menjalankan streamlit di eda
# "streamlit run eda.py"

def run():
    # Make title
    st.title("Customer Segmentation")
    st.image("deployment/image.jpg")
    st.markdown("## Background")
    st.markdown("""Understanding our customer is a must in the crowded competition recent days. Maintaining and improving our customer could lengthen their lifetime period with higher quality transactions.
                 This project focuses on the segmentation of credit card holders based on their behavioral patterns and distinct characteristics.
                """)

    st.markdown("## Objective")
    st.markdown("""The objective of this clustering is to identify the most effective, personalized strategies for increasing credit card usage frequency across different consumer segments.
                """)
    
    # Data Preparation
    df_result = pd.read_csv('deployment/final_dataset.csv')

    # Menampilkan Visualisasi EDA
    st.markdown("## Exploratory Data Analysis of Segmentation")

    # 1
    st.markdown("### Customer Segmentation")
    st.image("deployment/cluster.png", use_container_width=True)

    # Define columns' list without CLUSTER and CUST_ID
    cols = df_result.drop(columns=['CLUSTER','CUST_ID']).columns.tolist()

    # Determine mean as default aggregation function
    agg_dict = {col: 'mean' for col in cols}

    # Adjust for TENURE using median because it is discrete column
    if 'TENURE' in agg_dict:
        agg_dict['TENURE'] = 'median'

    # Make table for aggregation value
    profil_cluster = df_result.groupby('CLUSTER').agg(agg_dict)
    
    st.markdown("### Customer Balance")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(x=profil_cluster.index, y=profil_cluster['BALANCE'], ax=ax,palette='tab10')
    ax.set_title("Average of Customer's Balance", fontsize=14)
    st.pyplot(fig)

    st.markdown("### Customer Paying Full Bill")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(x=profil_cluster.index, y=profil_cluster['PRC_FULL_PAYMENT'], ax=ax,palette='tab10')
    ax.set_title('Average Percentage of Customer Paying Full Bill', fontsize=14)
    st.pyplot(fig)

    st.markdown("### Customer Purchasing Directly")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(x=profil_cluster.index, y=profil_cluster['ONEOFF_PURCHASES'], ax=ax,palette='tab10')
    ax.set_title("Average Purchases of Customer Purchasing Directly", fontsize=14)
    st.pyplot(fig)

    st.markdown("### Customer Purchasing With Installment")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(x=profil_cluster.index, y=profil_cluster['INSTALLMENTS_PURCHASES'], ax=ax,palette='tab10')
    ax.set_title("Average Purchases of Customer Purchasing With Installment", fontsize=14)
    st.pyplot(fig)

    st.markdown("### Customer Purchasing With Cash Withdrawal")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(x=profil_cluster.index, y=profil_cluster['CASH_ADVANCE'], ax=ax,palette='tab10')
    ax.set_title("Average Cash Withdrawal of Customer", fontsize=14)
    st.pyplot(fig)

    # 3
    # Make new columns for total expense
    profil_cluster['TOTAL_EXPENSE'] = profil_cluster['ONEOFF_PURCHASES'] + profil_cluster['INSTALLMENTS_PURCHASES'] + profil_cluster['CASH_ADVANCE']

    # Make new columns for displaying proportion of spend behavior
    profil_cluster['% of ONEOFF_PURCHASES'] = (profil_cluster['ONEOFF_PURCHASES'] / profil_cluster['TOTAL_EXPENSE']) * 100
    profil_cluster['% of INSTALLMENTS_PURCHASES'] = (profil_cluster['INSTALLMENTS_PURCHASES'] / profil_cluster['TOTAL_EXPENSE']) * 100
    profil_cluster['% of CASH_ADVANCE'] = (profil_cluster['CASH_ADVANCE'] / profil_cluster['TOTAL_EXPENSE']) * 100

    # Define columns to display by table
    result_cols = [
        'BALANCE', 'PRC_FULL_PAYMENT', 'TOTAL_EXPENSE', 
        '% of ONEOFF_PURCHASES', '% of INSTALLMENTS_PURCHASES', '% of CASH_ADVANCE']

    profil_analysis = profil_cluster[result_cols]
    st.markdown("### Summary Features of Customer Segmentation")
    st.dataframe(profil_analysis)
    st.markdown("""
                From the information above, we can take insights that:

                1. **Cluster 0 (Segment A)**: High balance (debt), weak ability to pay bills with almost all in the form of cash withdrawals
                2. **Cluster 1 (Segment B)**: Moderate balance, strong ability to pay bills with the largest nominal amount in the form of direct purchases
                3. **Cluster 2 (Segment C)**: High balance, weak ability to pay bills with the largest nominal amount in the form of cash withdrawals
                4. **Cluster 3 (Segment D)**: Low balance, moderate ability to pay bills with the largest nominal amount in the form of direct purchases
                5. **Cluster 4 (Segment E)**: Low balance, strong ability to pay bills with the largest nominal amount in the form of installment purchases
                """)
    
if __name__=='__main__':
    run()