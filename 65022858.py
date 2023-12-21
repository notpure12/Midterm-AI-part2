import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier,plot_tree
import matplotlib.pyplot as plt
import seaborn as sns

encoder = LabelEncoder()

File_path = 'D:/'
File_name = 'car_data.csv'

df = pd.read_csv(File_path+File_name)
df = df.dropna()

cols = ['Gender','Purchased']
df[cols] = df[cols].apply(encoder.fit_transform)

##scaler = StandardScaler()
##scaler.fit(df[['AnnualSalary']])
##df[['AnnualSalary']] = scaler.transform(df[['AnnualSalary']])

df.drop(columns=['User ID'], inplace=True)
x = df[['Gender','Age','AnnualSalary']]
y = df['Purchased']

x_train, x_test, y_train, y_test = train_test_split(x,y,random_state=0)

model = DecisionTreeClassifier(criterion='entropy')
model.fit(x,y)
dummy_pre = ['Female','25','80000']
encoder.fit(dummy_pre)
e_dummy = encoder.transform(dummy_pre)
e_dummy_adj = np.array(e_dummy).reshape(-1,3)

y_pre = model.predict(e_dummy_adj)
print('Predict : ',y_pre[0])
score = model.score(x, y)
print('Accuracy : ','{:.2f}'.format(score))

feature = x.columns.tolist()
Data_Class = y.tolist()

plt.figure(figsize=(15,10)) 
_ = plot_tree(model,
              feature_names = feature,
              class_names = Data_Class,
              label='all')
plt.show()
