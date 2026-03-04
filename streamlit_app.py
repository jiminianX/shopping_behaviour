## Step 00 - Import packages
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import numpy as np
from sklearn.metrics import roc_curve, precision_recall_curve
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, classification_report
from sklearn.metrics import precision_recall_curve

st.set_page_config(
    page_title="Telco Customer Churn Analysis 📡",
    layout="centered",
    page_icon="📡",
)

## Step 01 - Setup
st.sidebar.title("Telco Churn Analysis 📡")
page = st.sidebar.selectbox("Select Page", ["Business Case 📘", "Visualization 📊", "Prediction 🤖", "Insights and Recommendations 🧠"])

st.write("   ")

# cached loader returns the dataframe plus stats about cleaning
@st.cache_data
def load_data(path='WA_Fn-UseC_-Telco-Customer-Churn.csv'):
    df_local = pd.read_csv(path)
    # keep original row count for reporting
    original_count = len(df_local)
    df_local['SeniorCitizen'] = df_local['SeniorCitizen'].astype(int)
    # convert TotalCharges and report how many became NaN
    coerced = pd.to_numeric(df_local['TotalCharges'], errors='coerce')
    coerced_missing = coerced.isna().sum()
    df_local['TotalCharges'] = coerced
    return df_local, original_count, coerced_missing

# load once and reuse
(df, original_row_count, totalcharges_missing) = load_data()

# note: further cleaning (dropna when modeling) happens later

## Step 02 - Pages
if page == "Business Case 📘":

    st.subheader("Telco Customer Churn Dashboard")

    st.image("Gemini_Generated_Image_l7ijjel7ijjel7ij.png", use_container_width=True)

    # dataset provenance
    st.markdown("[🔗 View dataset source on Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)")

    st.markdown("""
    ## 🎯 Business Problem

    Customer churn in the telecom industry leads to:

    - **Revenue loss** from departing customers
    - **High acquisition costs** to replace churned customers
    - **Reduced customer lifetime value**
    - **Competitive disadvantage** in a saturated market

    Understanding churn drivers allows proactive retention strategies.
    """)

    st.markdown("""
    ## Our Solution

    1. **Data Analysis:** Identify key factors driving customer churn
    2. **Visualization:** Interactive dashboards to explore churn patterns
    3. **Predictive Modeling:** Logistic regression to predict churn probability
    """)

    st.markdown("""
    ## Why Retention Matters

    - **Churn costs:** According to [Harvard Business Review](https://hbr.org/2014/10/the-value-of-keeping-the-right-customers), acquiring a new customer can be as much as **25× more expensive** than keeping one.
    - **Retention impact:** [McKinsey](https://www.mckinsey.com/business-functions/marketing-and-sales/our-insights/the-three-cs-of-customer-satisfaction-consistency-consistency-consistency) estimates that a **1% improvement in retention** can boost profits by around **8%**.
    """)

    # data dictionary
    with st.expander("📘 Data Dictionary (click to expand)"):
        col_desc = {
            'customerID': 'Unique ID for each customer',
            'gender': 'Customer gender (Male/Female)',
            'SeniorCitizen': 'Whether customer is a senior citizen (1,0)',
            'Partner': 'Whether customer has a partner',
            'Dependents': 'Whether customer has dependents',
            'tenure': 'Number of months the customer has stayed',
            'PhoneService': 'Whether the customer has phone service',
            'MultipleLines': 'Whether the customer has multiple lines',
            'InternetService': 'Customer internet service provider',
            'OnlineSecurity': 'Whether the customer has online security',
            'OnlineBackup': 'Whether the customer has online backup',
            'DeviceProtection': 'Whether the customer has device protection',
            'TechSupport': 'Whether the customer has tech support',
            'StreamingTV': 'Whether the customer streams TV',
            'StreamingMovies': 'Whether the customer streams movies',
            'Contract': 'Customer contract term',
            'PaperlessBilling': 'Whether the customer uses paperless billing',
            'PaymentMethod': 'Payment method of the customer',
            'MonthlyCharges': 'The amount charged to the customer monthly',
            'TotalCharges': 'The total amount charged to the customer',
            'Churn': 'Whether the customer churned (Yes/No)'
        }
        for col, desc in col_desc.items():
            st.write(f"**{col}**: {desc}")

    st.markdown("##### Data Preview")
    rows = st.slider("Select number of rows to display", 5, 20, 5)
    st.dataframe(df.head(rows))

    # interactive filter
    with st.expander("Filter data by column"):
        filt_col = st.selectbox("Column", df.columns, index=0)
        filt_val = st.text_input("Value contains (case-insensitive)")
        if filt_val:
            filtered = df[df[filt_col].astype(str).str.contains(filt_val, case=False)]
            st.dataframe(filtered)
        else:
            st.write("Enter a value to filter the table")

    st.markdown("##### Data Shape")
    st.write("Telco Churn Data:", df.shape)

    st.markdown("##### Data Cleaning Summary")
    st.write(f"Original rows read: {original_row_count}")
    st.write(f"Rows with unconvertible TotalCharges: {totalcharges_missing}")

    st.markdown("##### Missing Values")
    missing_values = df.isnull().sum()
    missing_values = missing_values[missing_values > 0]
    if len(missing_values) > 0:
        st.write(missing_values)
        st.warning(f"⚠️ {missing_values.sum()} missing values found")
    else:
        st.success("✅ No missing values found")

    st.markdown("##### Summary Statistics")
    st.dataframe(df[['tenure', 'MonthlyCharges', 'TotalCharges']].describe())



elif page == "Visualization 📊":

    st.subheader("02 Data Visualization")

    # sidebar filters
    st.sidebar.markdown("---")
    st.sidebar.write("### Filters")
    genders = df['gender'].unique().tolist()
    contracts = df['Contract'].unique().tolist()
    services = df['InternetService'].unique().tolist()
    gender_filter = st.sidebar.multiselect("Gender", genders, default=genders)
    contract_filter = st.sidebar.multiselect("Contract", contracts, default=contracts)
    service_filter = st.sidebar.multiselect("Internet Service", services, default=services)

    @st.cache_data
    def apply_filters(df_in, genders, contracts, services):
        return df_in[(df_in['gender'].isin(genders)) &
                     (df_in['Contract'].isin(contracts)) &
                     (df_in['InternetService'].isin(services))]

    df_vis = apply_filters(df, gender_filter, contract_filter, service_filter)

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Churn Distribution 📊", "Churn by Contract 📋", "Correlation Matrix 🔥", "Churn by Service 📡", "Other Metrics 📈", "Data Quality 🔍"])

    with tab1:
        st.subheader("Customer Churn Distribution")
        st.write("This pie chart shows the proportion of customers who left vs. those who stayed. High churn slice is a red flag.")
        churn_counts = df_vis['Churn'].value_counts()
        display_index = churn_counts.index.map(lambda x: 'churned (left)' if x == 'Yes' else 'retained (stayed)')
        fig1, ax1 = plt.subplots(figsize=(8, 5))
        colors = ['#2ecc71', '#e74c3c']
        wedges, texts, autotexts = ax1.pie(churn_counts.values, labels=display_index, autopct='%1.1f%%',
                colors=colors, startangle=90, textprops={'fontsize': 14})
        for text in texts:
            text.set_color('black')
        ax1.set_title("Churned vs Retained", fontsize=16)
        st.pyplot(fig1)

        col1, col2 = st.columns(2)
        col1.metric("Total Customers", f"{len(df_vis):,}")
        col2.metric("Churn Rate", f"{(df_vis['Churn'] == 'Yes').mean() * 100:.1f}%")

    with tab2:
        st.subheader("Churn Rate by Contract Type")
        st.write("Different contract lengths have different churn profiles; month-to-month tends to be highest.")
        @st.cache_data
        def compute_by_contract(data):
            res = data.groupby('Contract')['Churn'].apply(lambda x: (x == 'Yes').mean() * 100).reset_index()
            res.columns = ['Contract', 'Churn Rate (%)']
            return res
        churn_by_contract = compute_by_contract(df_vis)
        fig2, ax2 = plt.subplots(figsize=(10, 5))
        sns.barplot(data=churn_by_contract, x='Contract', y='Churn Rate (%)', ax=ax2, palette='Reds_r')
        ax2.set_title("Churn Rate by Contract Type", fontsize=16)
        ax2.set_ylabel("Churn Rate (%)")
        ax2.set_xlabel("Contract Type")
        st.pyplot(fig2)

    with tab3:
        st.subheader("Correlation Matrix")
        st.write("Interactive heatmap allows you to hover over cells to see the exact correlation coefficient.")
        # make a copy so we don't add encoded columns back into df_vis
        df_corr = df_vis.copy()
        for col in ['Partner','Churn','Contract','InternetService','Dependents','PaymentMethod']:
            df_corr[col] = df_corr[col].astype('category').cat.codes
        numeric_cols = df_corr.select_dtypes(include=[np.number]).columns.tolist()
        corr = df_corr[numeric_cols].corr()
        
        fig3, ax3 = plt.subplots(figsize=(10, 6))
        sns.heatmap(corr, annot=True, cmap='Blues', fmt=".2f", ax=ax3)
        ax3.set_title("Correlation Heatmap")
        st.pyplot(fig3)

    with tab4:
        st.subheader("Churn by Internet Service")
        st.write("Compare churn rates across different internet providers or technologies.")
        @st.cache_data
        def compute_by_internet(data):
            res = data.groupby('InternetService')['Churn'].apply(lambda x: (x == 'Yes').mean() * 100).reset_index()
            res.columns = ['InternetService', 'Churn Rate (%)']
            return res
        churn_by_internet = compute_by_internet(df_vis)
        churn_by_internet = churn_by_internet.sort_values('Churn Rate (%)', ascending=False)
        fig4, ax4 = plt.subplots(figsize=(10, 5))
        sns.barplot(data=churn_by_internet, x='InternetService', y='Churn Rate (%)', ax=ax4, palette='Reds_r')
        ax4.set_title("Churn Rate by Internet Service", fontsize=16)
        ax4.set_ylabel("Churn Rate (%)")
        ax4.set_xlabel("Internet Service")
        st.pyplot(fig4)

    with tab5:
        st.subheader("Additional Metrics")
        st.write("Here are a few extra views: tenure distribution, monthly charges, and churn trend over tenure.")
        # tenure histogram
        fig5, ax5 = plt.subplots()
        sns.histplot(df_vis['tenure'], bins=30, kde=False, ax=ax5)
        ax5.set_title("Tenure Distribution")
        ax5.set_xlabel("Months")
        st.pyplot(fig5)

        # monthly charges boxplot
        fig6, ax6 = plt.subplots()
        sns.boxplot(x=df_vis['MonthlyCharges'], ax=ax6)
        ax6.set_title("Monthly Charges Boxplot")
        ax6.set_xlabel("USD")
        st.pyplot(fig6)

        # churn rate by tenure
        churn_tenure = df_vis.groupby('tenure')['Churn'].apply(lambda x: (x == 'Yes').mean()).reset_index()
        fig7, ax7 = plt.subplots()
        sns.lineplot(data=churn_tenure, x='tenure', y='Churn', ax=ax7)
        ax7.set_title("Churn Rate by Tenure")
        ax7.set_ylabel("Churn Rate")
        ax7.set_xlabel("Tenure (months)")
        st.pyplot(fig7)

    with tab6:
        st.subheader("Data Quality Checks")
        units = {'tenure': 'months', 'MonthlyCharges': 'USD', 'TotalCharges': 'USD'}
        # examine only the original numeric columns; drop any artificial encoded fields
        numeric_check_cols = ['SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges']
        
        # Collect all unique categorical values to create consistent color mapping
        all_cat_values = set()
        all_cat_values.update(['non-senior', 'senior'])  # SeniorCitizen values
        nonnum = df_vis.select_dtypes(include=['object', 'category']).columns.tolist()
        nonnum = [c for c in nonnum if c not in ('customerID', 'Churn', 'gender', 'Partner')]
        for col in nonnum:
            all_cat_values.update(df_vis[col].astype(str).unique())
        
        # Create blue gradient color map (darker shades for later values in sorted order)
        import matplotlib.cm as cm
        sorted_values = sorted(all_cat_values)
        n_values = len(sorted_values)
        blue_cmap = cm.get_cmap('Blues')
        colors_map = {}
        for i, val in enumerate(sorted_values):
            # Range from 0.3 (light) to 0.9 (dark) for visibility
            shade = 0.3 + 0.6 * (i / max(1, n_values - 1))
            colors_map[val] = blue_cmap(shade)
        
        def autopct_format(values):
            def inner(pct):
                total = sum(values)
                val = int(round(pct * total / 100.0))
                return f"{pct:.1f}%\n({val})"
            return inner

        senior_citizen_counts = None
        for col in numeric_check_cols:
            if col == 'SeniorCitizen':
                senior_citizen_counts = df_vis[col].value_counts().sort_index()
                continue
            elif col == 'tenure':
                continue
            else:
                fig, ax = plt.subplots()
                sns.histplot(df_vis[col].dropna(), kde=False, ax=ax)
                ax.set_title(f"Distribution of {col}")
                if col in units:
                    ax.set_xlabel(units[col])
                st.pyplot(fig)
                mean = df_vis[col].mean()
                std = df_vis[col].std()
                outliers = df_vis[(df_vis[col] < mean - 3 * std) | (df_vis[col] > mean + 3 * std)]
                if not outliers.empty:
                    st.write(f"{len(outliers)} extreme values found in {col} (beyond 3σ)")
        # for non-numeric columns, show a pie chart of value counts
        firstset,secondset=st.columns(2)
        with firstset:

            for col in nonnum[:len(nonnum)//2]:
                counts = df_vis[col].value_counts()
                colors = [colors_map[str(val)] for val in counts.index]
                figp, axp = plt.subplots(figsize=(8,8))
                wedges, texts, autotexts = axp.pie(counts.values,
                        labels=counts.index,
                        autopct=autopct_format(counts.values),
                        colors=colors,
                        textprops={'color': 'white', 'weight': 'bold'})
                for text in texts:
                    text.set_color('black')
                axp.set_title(f"{col} distribution")
                st.pyplot(figp,clear_figure=True,width='stretch')
        with secondset:
            for col in nonnum[len(nonnum)//2:]:
                counts = df_vis[col].value_counts()
                colors = [colors_map[str(val)] for val in counts.index]
                figp, axp = plt.subplots(figsize=(8,8))
                wedges, texts, autotexts = axp.pie(counts.values,
                        labels=counts.index,
                        autopct=autopct_format(counts.values),
                        colors=colors,
                        textprops={'color': 'white', 'weight': 'bold'})
                for text in texts:
                    text.set_color('black')
                axp.set_title(f"{col} distribution")
                axp.axis('equal')
                st.pyplot(figp,clear_figure=True,width='stretch')
        # Add senior citizen distribution note at bottom
        st.markdown("---")
        st.markdown("#### Senior Citizen Distribution")
        if senior_citizen_counts is not None:
            non_senior = senior_citizen_counts.get(0, 0)
            senior = senior_citizen_counts.get(1, 0)
            total = non_senior + senior
            non_senior_pct = (non_senior / total * 100) if total > 0 else 0
            senior_pct = (senior / total * 100) if total > 0 else 0
            st.write(f"**Non-senior:** {non_senior:,} ({non_senior_pct:.1f}%) | **Senior:** {senior:,} ({senior_pct:.1f}%)")


elif page == "Prediction 🤖":
    st.subheader("Churn Prediction — Logistic Regression")

    # allow the user to tweak split and regularization
    st.sidebar.markdown("---")
    st.sidebar.write("### Model settings")
    test_size = st.sidebar.slider("Test set fraction", 0.1, 0.5, 0.2)
    C_val = st.sidebar.slider("Inverse regularization (C)", 0.01, 10.0, 1.0)
    solver = st.sidebar.selectbox("Solver", ["lbfgs", "liblinear"], index=0)

    # prepare data
    df2 = df.drop(columns=['customerID']).dropna().copy()
    categorical_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
                        'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                        'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
                        'PaperlessBilling', 'PaymentMethod']
    numeric_cols = ['SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges']

    le_churn = LabelEncoder()
    df2['Churn'] = le_churn.fit_transform(df2['Churn'])

    X = df2[numeric_cols + categorical_cols]
    y = df2['Churn']

    # build preprocessing pipeline
    

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
        ]
    )

    model = Pipeline(steps=[('pre', preprocessor),
                            ('clf', LogisticRegression(C=C_val, solver=solver, max_iter=1000))])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # Metrics
    st.markdown("### Model Performance")
    st.markdown(
        "**All of the following metrics range from 0 to 1 – higher is better.**\n\n"
        "- **Accuracy:** overall fraction of correct predictions.\n"
        "- **Precision:** of the customers predicted to churn, how many actually churned.\n"
        "- **Recall:** of the customers who truly churned, how many were correctly identified."
    )
    col1, col2, col3 = st.columns(3)
    col1.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.3f}")
    col2.metric("Precision", f"{precision_score(y_test, y_pred):.3f}")
    col3.metric("Recall", f"{recall_score(y_test, y_pred):.3f}")

    # Precision-Recall curve
    precision, recall, _ = precision_recall_curve(y_test, y_prob)

    fig_pr, ax_pr = plt.subplots()
    ax_pr.plot(recall, precision, label='Precision-Recall curve')
    ax_pr.set_xlabel('Recall')
    ax_pr.set_ylabel('Precision')
    ax_pr.set_title('Precision-Recall Curve')
    st.pyplot(fig_pr)

    # Confusion Matrix
    st.markdown("### Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    actual_labels = ['churned (left)' if c == 'Yes' else 'retained (stayed)' for c in le_churn.classes_]
    predicted_labels = ['churn (leave)', 'retain (stay)']
    fig_cm, ax_cm = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm,
                xticklabels=predicted_labels, yticklabels=actual_labels)
    ax_cm.set_xlabel('Predicted')
    ax_cm.set_ylabel('Actual')
    ax_cm.set_title('Churn Prediction Confusion Matrix')
    st.pyplot(fig_cm)

    # Feature Importance
    st.markdown("### Feature Importance")
    cat_names = model.named_steps['pre'].named_transformers_['cat'].get_feature_names_out(categorical_cols)
    feature_names = numeric_cols + list(cat_names)
    coefs = model.named_steps['clf'].coef_[0]
    importance = pd.DataFrame({'Feature': feature_names, 'Coefficient': coefs})
    importance = importance.sort_values('Coefficient', key=abs, ascending=False)

    fig_coef, ax_coef = plt.subplots(figsize=(10, 8))
    colors = ['#e74c3c' if c > 0 else '#2ecc71' for c in importance['Coefficient']]
    sns.barplot(data=importance, x='Coefficient', y='Feature', palette=colors, ax=ax_coef)
    ax_coef.set_title("Feature Importance (Logistic Regression Coefficients)", fontsize=16)
    ax_coef.set_xlabel("Coefficient Value")
    st.pyplot(fig_coef)

    # coefficient interpretation
    st.markdown("#### Coefficient Interpretation")
    for feat, coef in importance.head(5).itertuples(index=False):
        direction = 'increases' if coef > 0 else 'decreases'
        st.write(f"- {feat}: {direction} churn as its value rises ({coef:.3f}).")

    # single record prediction
    st.markdown("### Make a prediction on new customer data")
    units = {'tenure': 'months', 'MonthlyCharges': 'USD', 'TotalCharges': 'USD'}
    with st.form(key='single_pred'):
        entries = {}
        for col in numeric_cols:
            if col == 'SeniorCitizen':
                # Show as Yes/No dropdown instead of numeric input
                senior_choice = st.selectbox("SeniorCitizen", ["No", "Yes"])
                entries[col] = 1 if senior_choice == "Yes" else 0
            else:
                label = f"{col} ({units.get(col, '')})" if col in units else col
                entries[col] = st.number_input(label, value=float(df[col].mean()))
        for col in categorical_cols:
            options = df[col].unique().tolist()
            entries[col] = st.selectbox(col, options)
        submit = st.form_submit_button("Predict churn probability")
        if submit:
            new_df = pd.DataFrame([entries])
            prob = model.predict_proba(new_df)[0, 1]
            st.write(f"Predicted churn probability: {prob:.2%}")

elif page == "Insights and Recommendations 🧠":
    st.subheader("Insights and Recommendations")

    # compute main stats with confidence intervals
    churn_rate = (df['Churn'] == 'Yes').mean()
    n = len(df)
    ci = 1.96 * np.sqrt(churn_rate * (1 - churn_rate) / n)
    avg_tenure = df['tenure'].mean()
    avg_charges = df['MonthlyCharges'].mean()

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Churn Rate", f"{churn_rate*100:.1f}%", delta=f"±{ci*100:.2f}%")
        with st.expander("Details"):
            figc, axc = plt.subplots()
            sns.histplot(df['Churn'].map(lambda x: 1 if x=='Yes' else 0), bins=2, kde=False, ax=axc)
            axc.set_xticks([0,1]); axc.set_xticklabels(['Stayed','Churned'])
            axc.set_title("Churn count")
            st.pyplot(figc)
    with col2:
        st.metric("Avg Tenure", f"{avg_tenure:.0f} months")
        with st.expander("Tenure distribution"):
            figt, axt = plt.subplots()
            sns.histplot(df['tenure'], bins=30, ax=axt)
            axt.set_title("Tenure distribution")
            st.pyplot(figt)
    with col3:
        st.metric("Avg Monthly Charges", f"${avg_charges:.2f}")
        with st.expander("Charges distribution"):
            figm, axm = plt.subplots()
            sns.histplot(df['MonthlyCharges'], bins=30, ax=axm)
            axm.set_title("Monthly charges")
            st.pyplot(figm)

    # download buttons
    st.download_button("Download full dataset", df.to_csv(index=False), "telco_churn.csv", "text/csv")

    st.markdown("""
    ## Key Insights

    1. **Contract Type Drives Churn:** Month-to-month customers churn at significantly higher rates than those on 1-year or 2-year contracts.
    2. **Tenure Matters:** Customers with shorter tenure are far more likely to churn — the first 12 months are critical.
    3. **Fiber Optic Risk:** Fiber optic internet customers show higher churn rates, suggesting possible service quality or pricing issues.
    4. **Payment Method:** Electronic check users have notably higher churn compared to automatic payment methods.
    5. **Senior Citizens:** Higher churn rate compared to non-senior customers.

    ## Recommendations

    1. **Incentivize Long-Term Contracts:** Offer discounts or perks for customers switching from month-to-month to annual contracts.
    2. **Early Retention Programs:** Focus retention efforts on new customers within their first 12 months.
    3. **Investigate Fiber Optic Service:** Review fiber optic pricing, speed, and reliability to address higher churn.
    4. **Promote Auto-Pay:** Encourage customers to switch from electronic check to automatic bank transfer or credit card payments.
    5. **Senior-Specific Plans:** Create tailored plans and dedicated support for senior citizen customers.
    """)

    st.info("🎯 Targeting the right segments with proactive retention can reduce churn by 15-25%.")
