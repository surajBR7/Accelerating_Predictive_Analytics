import time
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.ensemble import GradientBoostingClassifier

df = pd.read_csv('data.csv')

def detect_outlier(data):
    threshold = 3
    mean = np.mean(data)
    std = np.std(data)
    z_scores = (data - mean) / std
    return data[abs(z_scores) < threshold]

for col in df.columns:
    df[col] = detect_outlier(df[col].values)

features = ['author_id', 'book_rating', 'publish_year', 'text_lang']
target = 'book_genre'
X = df[features]
y = df[target]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
fs = SelectKBest(score_func=f_regression, k=4)
X_selected = fs.fit_transform(X_scaled, y)

s1 = time.time()
clf = GradientBoostingClassifier(n_estimators=100, random_state=0)
new_model=clf.fit(X_selected, y)
e1 = time.time()

print('Time taken to train using 6 cores: ', e1-s1)
