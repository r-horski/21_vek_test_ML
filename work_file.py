import os
import pandas as pd

test_df = pd.read_csv('data/test.csv')
train_df = pd.read_csv('data/train.csv')

test_df.info()
print(test_df.describe())