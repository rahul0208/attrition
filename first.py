import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns
import warnings
from pylab import rcParams
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings("ignore")

#%matplotlib inline

# Loading the CSV with pandas
data = pd.read_csv('NWMHackathon-master/NWMHackathonAIDataset.csv')

sizes = data['Attrition'].value_counts(sort = True)
colors = ["red","green"] 
labels = ["left","right"] 
rcParams['figure.figsize'] = 5,5

# Plot
plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=270,)

plt.title('Percentage of Attrition in Dataset')
#plt.show()

#Data cleansions NaN

data.drop(['EmployeeNumber','Gender','EmployeeCount','DailyRate','MonthlyRate','HourlyRate','EducationField','JobRole','Over18','StandardHours'], axis=1, inplace=True)
data.drop(['BusinessTravel','Department','MaritalStatus'] , axis=1, inplace=True)

data["OverTime"] =data["OverTime"].eq("Yes").mul(1)
data["Attrition"] = data["Attrition"].eq("Yes").mul(1)
print (data.dtypes)

Y=data['Attrition'].values
X=data.drop(labels = ["Attrition"],axis = 1)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=101)

model = LogisticRegression()
result = model.fit(X_train, Y_train)

prediction_test = model.predict(X_test)
# Print the prediction accuracy
print (metrics.accuracy_score(Y_test, prediction_test))

weights = pd.Series(model.coef_[0],index=X.columns.values)
print (weights.sort_values(ascending = False))

