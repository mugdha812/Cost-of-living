# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 15:46:51 2024

@author: Mugdha Shah
"""

#!pip install seaborn --upgrade

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import plot_tree
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.metrics import r2_score


df=pd.read_csv('finaldata.csv')

# Columns containing dollar signs
currency_columns = [
    'Total Cost of Living 1',
    'Total Disposable Income 2',
    'Annual Average Wage 1',
    'Median Home Price',
    'Monthly Mortgage Payment',
    'Rent Price',
    'Annual Healthcare Cost 1'
]

# Removing dollar signs and converting to numeric
for col in currency_columns:
    df[col] = df[col].replace('[\$,]', '', regex=True).astype(float)

# Save the cleaned dataset
df.to_csv('cleaned_file.csv', index=False)

#linear regression
y=df['Total Cost of Living 1']
x=df[['Total Disposable Income 2','Annual Average Wage 1','Average Transportation Cost 1','Median Home Price','Monthly Mortgage Payment','Rent Price','Median Monthly Housing Cost 1','Total Annual Food Cost 1','Annual Healthcare Cost 1','Average Annual Taxes']]

x_train, x_test, y_train, y_test= train_test_split(x,y,test_size=0.3, random_state=52)
lm=LinearRegression()
lm.fit(x_train,y_train)
lm.coef_
lm.intercept_
y_pred=lm.predict(x_test)
rmse=mean_squared_error(y_test, y_pred, squared=False)
rmse #4.357 ~ 4.36


df['Affordability Index'] = df['Total Disposable Income 2'] / df['Total Cost of Living 1']


affordable_states = df[['State', 'Affordability Index', 'Total Disposable Income 2', 'Total Cost of Living 1']].sort_values(by='Affordability Index', ascending=False)

# Display the top 10 states by Affordability Index
affordable_states.head(10)


# Filtering the dataset for the top 10 affordable states
affordable_states_top10 = affordable_states.head(10)

# Visualization for affordability
plt.figure(figsize=(12, 8))
sns.barplot(x='Affordability Index', y='State', data=affordable_states_top10, color='blue')
plt.title('Top 10 Affordable States by Affordability Index')
plt.xlabel('Affordability Index')
plt.ylabel('State')
plt.tight_layout()
plt.show()



top_10_col_states = df.sort_values(by='Total Cost of Living 1', ascending=False).head(10)

# Create a bar chart
plt.figure(figsize=(12, 8))
sns.barplot(x='Total Cost of Living 1', y='State', data=top_10_col_states, color='blue')
plt.title('Top 10 States by Total Cost of Living')
plt.xlabel('Total Cost of Living')
plt.ylabel('State')
plt.grid(axis='x')

plt.show()


bottom_10_col_states = df.sort_values(by='Total Cost of Living 1', ascending=False).tail(10)

# Create a bar chart
plt.figure(figsize=(12, 8))
sns.barplot(x='Total Cost of Living 1', y='State', data=bottom_10_col_states, color='blue')
plt.title('Top 10 States with least Total Cost of Living')
plt.xlabel('Total Cost of Living')
plt.ylabel('State')
plt.grid(axis='x')

plt.show()

#correlation matrix

correlation_matrix = df[['Total Cost of Living 1','Total Disposable Income 2','Average Transportation Cost 1','Median Home Price','Annual Healthcare Cost 1','Average Annual Taxes']].corr()


plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')#, fmt=".2f", linewidths=.5)
plt.title('Correlation Heatmap')
plt.show()


#decision tree


df_new = df.drop(columns='State')
y = df_new['Total Cost of Living 1']
x = df_new.drop(columns='Total Cost of Living 1')

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1) # 70% training and 30% testing


dt = DecisionTreeRegressor(random_state=1)
dt.fit(x_train, y_train)  


y_pred_train = dt.predict(x_train)
print("Training R^2 Score:", r2_score(y_train, y_pred_train)) #1


y_pred_test = dt.predict(x_test)
print("Testing R^2 Score:", r2_score(y_test, y_pred_test))  
# 0.97

plt.figure(figsize=(20,10))
plot_tree(dt)
plt.show()





# Max depth
print("Max Depth of the Tree:", dt.tree_.max_depth) #8
parameter_grid={'max_depth': range(1,8), 'min_samples_split': range(2,51)} 
#(1,8) because 1 is minimum depth and 8 is the depth of tree as above

from sklearn.model_selection import GridSearchCV
grid=GridSearchCV(dt,parameter_grid,verbose=3,scoring='r2' ,cv=5)
grid.fit(x_train,y_train)


#finding the best parameter
grid.best_params_ #{'max_depth': 4, 'min_samples_split': 6}

#building decision tree
dt=DecisionTreeRegressor(max_depth=4, min_samples_split=6,random_state=1)
dt.fit(x_train,y_train)



y_pred_train=dt.predict(x_train)
r2_score(y_train, y_pred_train)  #1.0 to 0.90


y_pred_test=dt.predict(x_test)
r2_score(y_test,y_pred_test)  #0.97 to 0.98

plt.figure(figsize=(20,10))
plot_tree(dt)
plt.show()



