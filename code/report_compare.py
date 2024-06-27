import numpy as np
import pandas as pd
from ydata_profiling import ProfileReport

#put train data
train_df = pd.read_csv('stock_data.csv')


# Generate profile report for training data
train_report = ProfileReport(train_df, title="Train")

# put new synthetic data
test_df = pd.read_csv('CopulaGAN_Data_for_synthesis.csv')

# Generate profile report for test data
test_report = ProfileReport(test_df, title="Test")

# Compare training and test reports
comparison_report = train_report.compare(test_report)

# Save comparison report to HTML file
comparison_report.to_file("comparison_data_for_synthesis.html")