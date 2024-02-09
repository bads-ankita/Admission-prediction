import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle


df = pd.read_csv(r"C:\Users\ANKITA GHOSH\Downloads\Admission_Prediction.csv")
#Fill missing values
# df['GRE Score'].fillna(df['GRE Score'].mean(), inplace=True)
# df['TOEFL Score'].fillna(df['TOEFL Score'].mean(), inplace=True)
# df['University Rating'].fillna(df['University Rating'].mean(), inplace=True)

def fill_missing(column):
    df[column]=df[column].fillna(df[column].mean())
missing_column=['GRE Score','TOEFL Score','University Rating']
for column in missing_column:
    fill_missing(column)

df.drop(columns=['Serial No.'],inplace=True)
x=df.drop(columns=['Chance of Admit'])
y=df['Chance of Admit']

X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.18,random_state=45)
lr= LinearRegression()
lr.fit(X_train,y_train)
filename='lr_for_admission'
pickle.dump(lr,open(filename,'wb'))
loaded_model=pickle.load(open('lr_for_admission','rb'))