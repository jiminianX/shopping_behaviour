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

# st.image("shopping.jpg")

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

    tab1, tab2, tab3, tab4 = st.tabs(["Distribution of Shopping Preference 📊", "Spending Analysis 💰", "Correlation Matrix 🔥", "Touch & Feel vs Store 🖐️"])

    with tab1:
        st.subheader("Distribution of Shopping Preference")

        col1, col2 = st.columns(2)

        with col1:
            fig1a, ax1a = plt.subplots(figsize=(5, 4))
            sns.countplot(data=df, x="shopping_preference", palette="Set2", ax=ax1a)
            ax1a.set_title("Overall Distribution")
            ax1a.set_xlabel("Shopping Preference")
            ax1a.set_ylabel("Count")
            st.pyplot(fig1a)

        with col2:
            fig1b, ax1b = plt.subplots(figsize=(5, 4))
            sns.countplot(data=df, x="city_tier", hue="shopping_preference", palette="Set2", ax=ax1b)
            ax1b.set_title("Preference by City Tier")
            ax1b.set_xlabel("City Tier")
            ax1b.set_ylabel("Count")
            ax1b.legend(title="Preference", fontsize=8)
            st.pyplot(fig1b)

        fig1c, ax1c = plt.subplots(figsize=(8, 4))
        sns.countplot(data=df, x="gender", hue="shopping_preference", palette="Set2", ax=ax1c)
        ax1c.set_title("Preference by Gender")
        ax1c.set_xlabel("Gender")
        ax1c.set_ylabel("Count")
        ax1c.legend(title="Preference")
        st.pyplot(fig1c)

    with tab2:
        st.subheader("Spending Analysis")

        spend_by_pref = df.groupby("shopping_preference")[["avg_online_spend", "avg_store_spend"]].mean().reset_index()

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Average Spend by Shopping Preference**")
            fig2a, ax2a = plt.subplots(figsize=(5, 4))
            x = range(len(spend_by_pref))
            width = 0.35
            ax2a.bar([i - width/2 for i in x], spend_by_pref["avg_online_spend"], width, label="Online Spend", color="steelblue")
            ax2a.bar([i + width/2 for i in x], spend_by_pref["avg_store_spend"],  width, label="Store Spend",  color="tomato")
            ax2a.set_xticks(list(x))
            ax2a.set_xticklabels(spend_by_pref["shopping_preference"])
            ax2a.set_ylabel("Average Spend ($)")
            ax2a.set_title("Online vs Store Spend")
            ax2a.legend()
            st.pyplot(fig2a)

        with col2:
            st.markdown("**Avg Online Spend by City Tier**")
            spend_by_city = df.groupby("city_tier")[["avg_online_spend", "avg_store_spend"]].mean().reset_index()
            fig2b, ax2b = plt.subplots(figsize=(5, 4))
            x2 = range(len(spend_by_city))
            ax2b.bar([i - width/2 for i in x2], spend_by_city["avg_online_spend"], width, label="Online Spend", color="steelblue")
            ax2b.bar([i + width/2 for i in x2], spend_by_city["avg_store_spend"],  width, label="Store Spend",  color="tomato")
            ax2b.set_xticks(list(x2))
            ax2b.set_xticklabels(spend_by_city["city_tier"])
            ax2b.set_ylabel("Average Spend ($)")
            ax2b.set_title("Online vs Store Spend by City Tier")
            ax2b.legend()
            st.pyplot(fig2b)

        st.markdown("**Summary Statistics**")
        st.dataframe(
            spend_by_pref.rename(columns={
                "shopping_preference": "Preference",
                "avg_online_spend": "Avg Online Spend ($)",
                "avg_store_spend": "Avg Store Spend ($)"
            }).style.format({"Avg Online Spend ($)": "{:,.0f}", "Avg Store Spend ($)": "{:,.0f}"}),
            use_container_width=True
        )
    with tab3:
        st.subheader("Correlation Matrix")
        fig3, ax3 = plt.subplots(figsize=(16, 6))
        sns.heatmap(
            df.drop(['gender', 'city_tier', 'shopping_preference'], axis=1).corr(),
            annot=True,
            cmap='coolwarm',
            fmt=".2f",
            ax=ax3
        )
        ax3.set_title("Correlation Heatmap")
        st.pyplot(fig3)

    with tab4:
        st.subheader("Does need_touch_feel_score Drive Store Preference?")
        fig4, ax4 = plt.subplots(figsize=(8, 5))
        sns.boxplot(data=df, x="shopping_preference", y="need_touch_feel_score", palette="Set2", ax=ax4)
        ax4.set_title("Need to Touch & Feel Score by Shopping Preference", fontsize=14)
        ax4.set_xlabel("Shopping Preference")
        ax4.set_ylabel("Need Touch & Feel Score")
        st.pyplot(fig4)
        st.markdown("""
        **Interpretation:** If Store shoppers show a noticeably higher median `need_touch_feel_score`
        than Online or Hybrid shoppers, it confirms that tactile preference is a key driver of
        in-store shopping behaviour.
        """)

elif page == "Prediction":
    st.subheader("03 Prediction with Linear Regression")

    ## Data Preprocessing
    df2 = df.copy()

    ### removing missing values
    df2 = df2.dropna()

    ### Label Encoder to change text categories into number categories
    le = LabelEncoder()
    df2["gender"]    = le.fit_transform(df2["gender"])
    df2["city_tier"] = le.fit_transform(df2["city_tier"])

    # shopping_preference is categorical — drop it so only numeric columns remain
    df2 = df2.drop(columns=["shopping_preference"])

    list_var = list(df2.columns)

    default_features = [c for c in list_var if c != "avg_online_spend"]
    features_selection = st.sidebar.multiselect("Select Features (X)", list_var, default=default_features)
    target_selection   = st.sidebar.selectbox("Select target variable (Y)", list_var, index=list_var.index("avg_online_spend"))
    test_size          = st.sidebar.slider("Test Set Size (%)", 10, 40, 20) / 100
    selected_metrics   = st.sidebar.multiselect("Metrics to display", ["Mean Squared Error (MSE)","Mean Absolute Error (MAE)","R² Score"], default=["Mean Absolute Error (MAE)", "R² Score"])

    # Make sure target is not also a feature
    features_selection = [f for f in features_selection if f != target_selection]

    if len(features_selection) == 0:
        st.warning("Please select at least one feature different from the target.")
        st.stop()

    ### i) X and y
    X = df2[features_selection]
    y = df2[target_selection]

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Features preview (X)**")
        st.dataframe(X.head())
    with col2:
        st.markdown("**Target preview (y)**")
        st.dataframe(y.head())

    ### ii) train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    ## Model

    ### i) Definition model
    model = LinearRegression()

    ### ii) Training model
    model.fit(X_train, y_train)

    ### iii) Prediction
    predictions = model.predict(X_test)

    ### iv) Evaluation
    st.markdown("### Model Performance")
    mae = metrics.mean_absolute_error(y_test, predictions)
    mse = metrics.mean_squared_error(y_test, predictions)
    r2  = metrics.r2_score(y_test, predictions)

    metric_cols = st.columns(3)
    if "Mean Squared Error (MSE)" in selected_metrics:
        metric_cols[0].metric("MSE", f"{mse:,.2f}")
    if "Mean Absolute Error (MAE)" in selected_metrics:
        metric_cols[1].metric("MAE", f"{mae:,.2f}")
    if "R² Score" in selected_metrics:
        metric_cols[2].metric("R²", f"{r2:.3f}")

    st.success(f"Model trained on {len(X_train)} rows — MAE: {mae:,.2f}")

    ### v) Actual vs Predicted scatter plot
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(y_test, predictions, alpha=0.4, color="steelblue", edgecolors="white", linewidths=0.3)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "--r", linewidth=2, label="Perfect fit")
    ax.set_xlabel("Actual Value")
    ax.set_ylabel("Predicted Value")
    ax.set_title(f"Actual vs Predicted — {target_selection}")
    ax.legend()
    st.pyplot(fig)

    ### vi) Feature coefficients bar chart
    # st.markdown("### Feature Coefficients")
    # coef_df = pd.DataFrame({
    #     "Feature": features_selection,
    #     "Coefficient": model.coef_
    # }).sort_values("Coefficient", key=abs, ascending=False)

    # fig2, ax2 = plt.subplots(figsize=(8, max(3, len(features_selection) * 0.4)))
    # colors = ["steelblue" if c >= 0 else "tomato" for c in coef_df["Coefficient"]]
    # ax2.barh(coef_df["Feature"], coef_df["Coefficient"], color=colors)
    # ax2.axvline(0, color="black", linewidth=0.8)
    # ax2.set_title("Linear Regression Coefficients")
    # ax2.set_xlabel("Coefficient Value")
    # st.pyplot(fig2)

    ### vii) Predict for a new customer
    # st.markdown("### Predict for a New Customer")
    # with st.form("predict_form"):
    #     input_vals = {}
    #     form_cols = st.columns(3)
    #     for i, feat in enumerate(features_selection):
    #         col_min  = float(df2[feat].min())
    #         col_max  = float(df2[feat].max())
    #         col_mean = float(df2[feat].mean())
    #         input_vals[feat] = form_cols[i % 3].number_input(
    #             feat, min_value=col_min, max_value=col_max, value=round(col_mean, 2)
    #         )
    #     submitted = st.form_submit_button("Predict")

    # if submitted:
    #     input_df = pd.DataFrame([input_vals])
    #     result = model.predict(input_df)[0]
    #     st.info(f"Predicted **{target_selection}**: **{result:,.2f}**")

elif page == "Insights and Recommendations 🧠":
    st.subheader("05 Insights and Recommendations")

    st.markdown("## Key Insights")

    insights = [
        ("Tech-savvy users lean toward Online", "Users with high `tech_savvy_score` are significantly more likely to shop online."),
        ("High need-to-touch-feel → Store", "Customers with a high `need_touch_feel_score` strongly prefer in-store shopping."),
        ("High time pressure → Online / Hybrid", "Time-pressured shoppers gravitate toward online or hybrid channels for convenience."),
        ("Tier 1 cities show more online adoption", "Urban Tier 1 customers are more comfortable with digital transactions."),
        ("Internet hours correlate with online behaviour", "More daily internet usage is associated with higher online shopping frequency."),
        ("Gender differences are moderate", "Gender influences shopping preference but the effect size is smaller than behavioural factors."),
        ("Discount sensitivity drives hybrid behaviour", "Price-sensitive shoppers often split purchases across online and in-store channels."),
        ("High impulse buyers lean hybrid", "Impulse buyers are drawn to both online convenience and in-store spontaneity."),
        ("Delivery sensitivity affects online choice", "Customers sensitive to delivery fees and wait times are less likely to shop purely online."),
        ("Brand loyalty impacts store preference", "Highly brand-loyal customers tend to prefer the in-store experience for reassurance."),
    ]

    for i, (title, detail) in enumerate(insights, 1):
        st.markdown(f"**{i}. {title}**")
        st.write(detail)

    st.markdown("---")
    st.markdown("## Business Recommendations")

    recommendations = [
        ("Target high tech-savvy, time-pressured users with online promotions",
         "Run personalised digital campaigns for customers scoring high on both `tech_savvy_score` and `time_pressure_level` — they are the easiest to convert fully online."),
        ("Improve the in-store experience for high tactile-need customers",
         "Invest in experiential retail (product demos, sensory displays) to retain customers with a high `need_touch_feel_score` who cannot be converted online."),
        ("Offer free returns to convert hybrid users fully online",
         "Free and easy returns remove the biggest risk for hybrid shoppers. Incentivising return-friendly policies can shift them to fully online."),
        ("Focus Tier 1 cities for aggressive digital marketing",
         "Tier 1 city residents already show higher online adoption — concentrate digital ad spend and loyalty programmes there for the best ROI."),
    ]

    for i, (title, detail) in enumerate(recommendations, 1):
        st.markdown(f"**{i}. {title}**")
        st.write(detail)