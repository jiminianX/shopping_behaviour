# 📡 Telco Customer Churn Analysis

This repository hosts a Streamlit dashboard focused on understanding and predicting customer churn for a telecom company. Its goal is to surface actionable insights and provide a simple predictive model to help marketing and retention teams act proactively.

Data comes from the popular [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) dataset. Churn has serious business impacts – replacing a lost customer can cost **3–5x more** than retaining one, and industry studies estimate that improving retention by just 5% can increase profits by up to 25–95%.

The app includes:

- Business context and data dictionary
- Interactive visualizations with filtering
- Logistic regression model with tuning interface
- Single-user prediction and model interpretation
- Insights & automated recommendations with export options


[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://blank-app-template.streamlit.app/)

### Getting started

1. Install the requirements:
   ```bash
   $ pip install -r requirements.txt
   ```
2. Launch the dashboard:
   ```bash
   $ streamlit run streamlit_app.py
   ```

### Development notes

- The dataset is read and cached; cleaning statistics are reported on the business page.
- Features are preprocessed with a `ColumnTransformer` and model built using `Pipeline`.
- The project is intentionally compact but can be extended with additional analyses or alternative models.

### Research & business case

- **Churn costs:** According to [Harvard Business Review](https://hbr.org/2014/10/the-value-of-keeping-the-right-customers), acquiring a new customer can be as much as 25-times more expensive than retaining an existing one.
- **Retention impact:** [McKinsey](https://www.mckinsey.com/business-functions/marketing-and-sales/our-insights/the-three-cs-of-customer-satisfaction-consistency-consistency-consistency) reports that a mere 1% improvement in customer retention can yield an 8% increase in profitability.

For more details, see the `Business Case` tab in the running app.
