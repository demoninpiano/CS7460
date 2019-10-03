from sklearn.tree import DecisionTreeRegressor    # Import decision tree regression model
import pandas as pd
from sklearn import datasets

boston = datasets.load_boston()            # Load Boston Dataset
df = pd.DataFrame(boston.data[:, 12])      # Create DataFrame using only the LSAT feature
df.columns = ['LSTAT']
df['MEDV'] = boston.target                 # Create new column with the target MEDV


X = df[['LSTAT']].values                          # Assign matrix X
y = df['MEDV'].values                             # Assign vector y

# Sort X and y by ascending values of X
sort_idx = X.flatten().argsort()
X = X[sort_idx]
y = y[sort_idx]

tree = DecisionTreeRegressor(criterion='mse',     # Initialize and fit regressor
                             max_depth=3)
tree.fit(X, y)
