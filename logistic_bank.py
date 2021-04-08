import pandas as pd
import streamlit as st 
import numpy as np
import sklearn
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
import pickle
import joblib

loan = pd.read_csv("train.csv")

with open ('logg_pickle','rb') as f :
	classifier = pickle.load(f)
	
def prediction(Gender, MaritalStatus, Education,Self_Employed,Dependents,ApplicantIncome,CoApplicantIncome, LoanAmount, Loan_Amount_Term,Credit_History):

	if Gender == 'Male':
		Gender = 0
	else :
		Gender = 1
	if MaritalStatus == 'Unmarried':
		MaritalStatus = 0
	else :
		MaritalStatus = 1
	
	if Education == 'Not Graduate':
		Education = 0
	else :
		Education = 1
	
	if Self_Employed == 'No':
		Self_Employed = 0
	else :
		Self_Employed = 1
	
	if Credit_History == '0':
		Credit_History = 0
	else :
		Credit_History = 1	
	prediction1 = classifier.predict([[Gender, MaritalStatus, Education,Self_Employed,Dependents,ApplicantIncome,CoApplicantIncome, LoanAmount, Loan_Amount_Term,Credit_History]])
	if prediction1 == 0:
		prediction1 = 'Rejected'
	else:
		prediction1 = 'Approved'
	return prediction1
	
	
def main():
    # giving the webpage a title
      
    # here we define some of the front end elements of the web page like 
    # the font and background color, the padding and the text to be displayed
	html_temp = """
	<div style ="background-color:purple;padding:13px">
	<h1 style ="color:white;text-align:center;">Loan Status Prediction </h1>
	</div>
	"""
      
    # this line allows us to display the front end aspects we have 
    # defined in the above code
	st.markdown(html_temp, unsafe_allow_html = True)
	Gender = st.selectbox('Gender',("Male","Female"))
	MaritalStatus = st.selectbox('MaritalStatus',("Unmarried","Married"))
	Education = st.selectbox('Education',("Not Graduate","Graduate"))
	Self_Employed = st.selectbox('Self_Employed',("Yes","No"))
	Dependents = st.selectbox('Dependents',('0','1','2'))
	ApplicantIncome = st.number_input("Applicant Income")
	CoApplicantIncome = st.number_input("Co-Applicant Income")
	LoanAmount = st.number_input("Loan Amount")
	Loan_Amount_Term=st.number_input("Loan_Amount_Term")
	Credit_History = st.selectbox('Credit_History',('0','1'))
	result = ""
	
	if st.button("Predict"):
		result = prediction(Gender, MaritalStatus, Education,Self_Employed,Dependents,ApplicantIncome,CoApplicantIncome, LoanAmount, Loan_Amount_Term,Credit_History)
		st.success('Loan Status is {}'.format(result))
		print(LoanAmount)

		
if __name__=='__main__':
	main()