
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

df = pd.read_csv('C:\\Users\\david\\OneDrive\\Escritorio\\Tennis regression model\\tennis_stats.csv')
print(df.head())

feature_variables = df.columns.difference(['Player', 'Winnings'])
target_variable = df['Winnings']




scores = {}
for i in feature_variables:
    X = df[i].values.reshape(-1,1)
    x_train, x_test, y_train, y_test = train_test_split(X, target_variable, train_size = 0.8, test_size = 0.2)
    model = LinearRegression()
    model.fit(x_train, y_train)
    scores[i] = model.score(x_test, y_test)

best_scores = {key:value for key, value in scores.items() if value > 0.7}

print(best_scores)
feature_variables = list(best_scores.keys())




X = df[feature_variables]
y = df['Winnings']

x_train, x_test, y_train, y_test = train_test_split(X, y, train_size = 0.8, test_size = 0.2)

model = LinearRegression()
model.fit(x_train, y_train)
y_prediction = model.predict(x_test)
print(model.score(x_test, y_test))

plt.scatter(y_test, y_prediction, alpha = 0.4)
plt.xlabel('test')
plt.ylabel('prediction')
plt.show()













