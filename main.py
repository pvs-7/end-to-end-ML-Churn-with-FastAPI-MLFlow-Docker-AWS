from churn_ml.data.loader import load_data
from churn_ml.validation.validator import validate_data
from churn_ml.preprocessing.cleaning import clean_data
from churn_ml.features.engineering import engineer_features

df = load_data("data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")

validate_data(df)
clean_df = clean_data(df)
eng_df = engineer_features(clean_df)

