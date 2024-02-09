
import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
import pickle

st.title('Admission project')

Gre_Score=st.number_input('Gre_score')
TOEFL_Score=st.number_input('TOEFL_Score')
University_rating=st.number_input('University_rating')
SOP=st.number_input('SOP',min_value=0,max_value=5)
LOR=st.number_input('LOR')
CGPA=st.number_input('CGPA')
Resarch=st.number_input('Resarch')


df = pd.read_csv(r"C:\Users\ANKITA GHOSH\Downloads\Admission_Prediction.csv")
def fill_missing(column):
    df[column]=df[column].fillna(df[column].mean())
missing_column=['GRE Score','TOEFL Score','University Rating']
for column in missing_column:
    fill_missing(column)

df.drop(columns=['Serial No.'],inplace=True)
x=df.drop(columns=['Chance of Admit'])
y=df['Chance of Admit']

user_input=[[Gre_Score,TOEFL_Score,University_rating,SOP,LOR,CGPA,Resarch]]
scaler=StandardScaler()
scaler.fit(x)
scaled_user_input=scaler.transform(user_input)
st.write(user_input)
st.write(scaled_user_input)
loaded_model=pickle.load(open('lr_for_admission','rb'))
result=loaded_model.predict(scaled_user_input)
st.write("The result is",result)

if st.button("Predict"):
    result_percentage = result * 100
    st.header( " percentage of you getting admitted in university is " + str(result_percentage))