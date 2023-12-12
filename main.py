import pandas as pd

training_set = pd.read_parquet('Tutorial_training_set.parquet')
print(training_set.shape)
testing_set = pd.read_parquet('Tutorial_testing_set.parquet')
print(testing_set.shape)

print(training_set.head())


training_set['fa_sitemap'].value_counts().to_csv('fa_sitemap.csv')