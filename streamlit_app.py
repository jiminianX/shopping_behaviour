## Step 00 - Import of the packages

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import numpy as np
import sklearn

# from ydata_profiling import ProfileReport
from streamlit.components.v1 import html

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
# from streamlit_pandas_profiling import st_profile_report

st.set_page_config(
    page_title="Online vs In-Store Shopping Behaviour 🛍️",
    layout="centered",
    page_icon="🛍️",
)


## Step 01 - Setup
st.sidebar.title("Shopping Behaviour 🛍️")
page = st.sidebar.selectbox("Select Page",["Business Case 📘","Visualization 📊", "Prediction", "Insights and Recommendations 🧠"])


#st.video("video.mp4")

st.image("shopping.avif")

st.write("   ")
st.write("   ")
st.write("   ")
df=pd.read_csv('shopping.csv')

df.head()  

## Step 02 - Load dataset
if page == "Business Case 📘":

    st.subheader("Online vs In-Store Shopping Behaviour Dashboard")

    st.markdown("""
    ## 🎯 Business Problem

    Online and in-store shopping behaviors vary significantly, leading to:

    - Missed revenue opportunities
    - Inefficient inventory management
    - Suboptimal marketing strategies

    These inefficiencies lead to hidden financial losses.
    """)

    st.markdown("""
    ## Our Solution

    1. Data Analysis: Identify key factors contributing to revenue leakage.
    2. Visualization: Create interactive dashboards to monitor shopping behaviors.
    3. Predictive Modeling: Forecast future shopping trends
    """)

    st.markdown("##### Data Preview")
    rows = st.slider("Select a number of rows to display",5,20,5)
    st.dataframe(df.head(rows))


    st.markdown("##### Missing values")
    missing_values = df.isnull().sum()
    
    st.write(missing_values)
    
    st.markdown("##### Data Shape")
    st.write("Shopping Data:", df.shape)
    

    if missing_values.sum() == 0:
        st.success("✅ No missing values found")
    else:
        st.warning("⚠️ you have missing values")

     

    st.markdown("##### 📈 Summary Statistics")
    if st.button("Show Describe Table"):
        st.dataframe(df.describe())

elif page == "Visualization 📊":

    ## Step 03 - Data Viz
    st.subheader("02 Data Vizualization")

    # col_x = st.selectbox("Select X-axis variable", df.columns, index=0)
    # col_y = st.selectbox("Select Y-axis variable", df.columns, index=1)

    tab1, tab2, tab3, tab4 = st.tabs(["Revenue Leakage 📊", "Patient Lifetime Value 💰", "Correlation Matrix 🔥" ,"No-Show Rate by Clinic"])

    with tab1:
        st.subheader("Revenue Leakage Bar Chart")
        df['revenue_loss'] = df['revenue_expected'] - df['revenue_realized']
        revenue_by_clinic = df.groupby("clinic_location").agg(
            expected_revenue=('revenue_expected', 'sum'),
            realized_revenue=('revenue_realized', 'sum'),
            revenue_loss=('revenue_loss', 'sum')
        ).reset_index()
        revenue_by_clinic['revenue_loss_percent'] = (
            revenue_by_clinic['revenue_loss'] / revenue_by_clinic['expected_revenue'] * 100
        )
        fig1, ax1 = plt.subplots(figsize=(10, 5))
        sns.barplot(data=revenue_by_clinic, x='clinic_location', y='revenue_loss_percent', ax=ax1)
        ax1.set_title("Revenue Loss % by Clinic", fontsize=16)
        ax1.set_ylabel("% Revenue Lost")
        ax1.set_xlabel("Clinic")
        st.pyplot(fig1)

    with tab2:
        st.subheader("Patient Lifetime Value")
        ltv = patients.copy()
        ltv['expected_future_visits'] = 10 - ltv['total_lifetime_visits']
        ltv['estimated_LTV'] = (
            ltv['total_lifetime_revenue']
            + (ltv['expected_future_visits'] * ltv['total_lifetime_revenue']
            / ltv['total_lifetime_visits'].replace(0, 1))
        )
        ltv["revenue_per_visit"]=ltv["total_lifetime_revenue"]/ltv["total_lifetime_visits"]
        st.dataframe(ltv.head())
        avgrev_insurance=ltv.groupby('insurance_type')['total_lifetime_revenue'].mean()
        avgrev_insurance_visit=ltv.groupby('insurance_type')['revenue_per_visit'].mean()
        col1, col2 = st.columns([1, 1])
    
        with col1:
            # Display the data in a table
            st.dataframe(
                avgrev_insurance.reset_index().rename(
                    columns={'insurance_type': 'Insurance Type', 
                            'total_lifetime_revenue': 'Average Revenue ($)'}
                ),
                use_container_width=True
            )
            st.dataframe(
                avgrev_insurance_visit.reset_index().rename(
                    columns={'insurance_type': 'Insurance Type', 
                            'revenue_per_visit': 'Average Revenue per Visit ($)'}
                ),
                use_container_width=True
            )
            
            # Show summary statistics
            st.info(f"**Overall Average Revenue:** ${df['total_lifetime_revenue'].mean():,.2f}")
        
        with col2:
            # Create the pie chart
            fig5, ax5 = plt.subplots(figsize=(10, 8))
            fig6, ax6 = plt.subplots(figsize=(10, 8))
            
            # Create pie chart
            colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']
            wedges, texts, autotexts = ax5.pie(
                avgrev_insurance.values,
                labels=avgrev_insurance.index,
                autopct='%1.1f%%',
                colors=colors,
                startangle=90,
                textprops={'fontsize': 12}
            )
            colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']
            wedges, texts, autotexts = ax6.pie(
                avgrev_insurance_visit.values,
                labels=avgrev_insurance_visit.index,
                autopct='%1.1f%%',
                colors=colors,
                startangle=90,
                textprops={'fontsize': 12}
            )
            # Add title
            ax5.set_title('Average Revenue Distribution by Insurance Type', 
                        fontsize=16, fontweight='bold', pad=20)
            ax6.set_title('Average Revenue Distribution per Visit by Insurance Type', 
                        fontsize=16, fontweight='bold', pad=20)
            
            # Make the percentage text more readable
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
            
            # Display the chart in Streamlit
            st.pyplot(fig5)
            st.pyplot(fig6)
    with tab3:
        st.subheader("Correlation Matrix")
        fig3, ax3 = plt.subplots(figsize=(10, 6))
        sns.heatmap(
            df[['age', 'chronic_condition_flag', 'total_lifetime_visits',
                'revenue_expected', 'revenue_realized', 'no_show_flag']].corr(),
            annot=True,
            cmap='Blues',
            fmt=".2f",
            ax=ax3
        )
        ax3.set_title("Correlation Heatmap")
        st.pyplot(fig3)

    with tab4:
        st.subheader("No-Show Rate by Clinic")
        no_show_by_clinic = df.groupby("clinic_location")["no_show_flag"].mean().reset_index()
        no_show_by_clinic.rename(columns={"no_show_flag": "no_show_rate"}, inplace=True)
        fig4, ax4 = plt.subplots(figsize=(10, 5))
        sns.barplot(data=no_show_by_clinic, x='clinic_location', y='no_show_rate', ax=ax4)
        ax4.set_title("No-Show Rate by Clinic", fontsize=16)
        ax4.set_ylabel("No-show Rate")
        ax4.set_xlabel("Clinic")
        st.pyplot(fig4)

# elif page == "Automated Report 📑":
#     st.subheader("03 Automated Report")
#     if st.button("Generate Report"):
#         with st.spinner("Generating report..."):
#             profile = ProfileReport(df,title="Clinic Revenue Report",explorative=True,minimal=True)
#             html(profile.to_html(), height=1400)

#         export = profile.to_html()
#         st.download_button(label="📥 Download full Report",data=export,file_name="clinic_revenue_report.html",mime='text/html')


elif page == "Prediction":
    st.subheader("03 Prediction with Linear Regression")
    df2 = df
    ## Data Preprocessing

    ### removing missing values 
    df2 = df2.dropna()

    ### Label Encoder to change text categories into number categories
    
    le = LabelEncoder()

    df2["ocean_proximity"] = le.fit_transform(df2["ocean_proximity"])

    list_var = list(df2.columns)

    features_selection = st.sidebar.multiselect("Select Features (X)",list_var,default=list_var)
    target_selection  = st.sidebar.selectbox("Select target variable (Y))",list_var)
    selected_metrics = st.sidebar.multiselect("Metrics to display", ["Mean Squared Error (MSE)","Mean Absolute Error (MAE)","R² Score"],default=["Mean Absolute Error (MAE)"])

    ### i) X and y
    X = df2[features_selection]
    y = df2[target_selection]

    st.dataframe(X.head())
    st.dataframe(y.head())

    ### ii) train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)


    ## Model 

    ### i) Definition model
    model = LinearRegression()

    ### ii) Training model
    model.fit(X_train,y_train)

    ### iii) Prediction
    predictions = model.predict(X_test)

    ### iv) Evaluation  
    if "Mean Squared Error (MSE)" in selected_metrics:
        mse = metrics.mean_squared_error(y_test, predictions)
        st.write(f"- **MSE** {mse:,.2f}")
    if "Mean Absolute Error (MAE)" in selected_metrics:
        mae = metrics.mean_absolute_error(y_test, predictions)
        st.write(f"- **MAE** {mae:,.2f}")
    if "R² Score" in selected_metrics:
        r2 = metrics.r2_score(y_test, predictions)
        st.write(f"- **R2** {r2:,.3f}")

    st.success(f"My model performance is of ${np.round(mae,2)}")

    fig, ax = plt.subplots()
    ax.scatter(y_test,predictions,alpha=0.5)
    ax.plot([y_test.min(),y_test.max()],
           [y_test.min(),y_test.max() ],"--r",linewidth=2)
    ax.set_xlabel("Actual Value")
    ax.set_ylabel("Predicted Value")
    ax.set_title("Actual vs Predicted")
    st.pyplot(fig)

elif page == "Insights and Recommendations 🧠":
    st.subheader("05 Insights and Recommendations")
    st.markdown("""
    ## Key Insights

    1. 
    2. 
    3. 

    ## Recommendations

    1. 
    2. 
    3. 
    """)