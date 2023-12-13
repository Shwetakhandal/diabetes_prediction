import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier  
from sklearn.model_selection import GridSearchCV  
from sklearn import tree
from sklearn import metrics
@st.cache()
def d_tree_pred(diabetes_df, glucose, bp, insulin, bmi, pedigree, age):
    # Split the train and test dataset. 
    feature_columns = list(diabetes_df.columns)

    # Remove the 'Pregnancies', Skin_Thickness' columns and the 'target' column from the feature columns
    feature_columns.remove('Skin_Thickness')
    feature_columns.remove('Pregnancies')
    feature_columns.remove('Outcome')

    X = diabetes_df[feature_columns]
    y = diabetes_df['Outcome']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 1)

    dtree_clf = DecisionTreeClassifier(criterion="entropy", max_depth=3)
    dtree_clf.fit(X_train, y_train) 
    y_train_pred = dtree_clf.predict(X_train)
    y_test_pred = dtree_clf.predict(X_test)
    # Predict diabetes using the 'predict()' function.
    prediction = dtree_clf.predict([[glucose, bp, insulin, bmi, pedigree, age]])
    prediction = prediction[0]

    score = round(metrics.accuracy_score(y_train, y_train_pred) * 100, 3)

    return prediction, score
def grid_tree_pred(diabetes_df, glucose, bp, insulin, bmi, pedigree, age):    
    feature_columns = list(diabetes_df.columns)
    # Remove the 'Pregnancies', 'Skin_Thickness' columns and the 'target' column from the feature columns
    feature_columns.remove('Pregnancies')
    feature_columns.remove('Skin_Thickness')
    feature_columns.remove('Outcome')
    X = diabetes_df[feature_columns]
    y = diabetes_df['Outcome']
    # Split the train and test dataset. 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 1)

    param_grid = {'criterion':['gini','entropy'], 'max_depth': np.arange(4,21), 'random_state': [42]}

    # Create a grid
    grid_tree = GridSearchCV(DecisionTreeClassifier(), param_grid, scoring = 'roc_auc', n_jobs = -1)

    # Training
    grid_tree.fit(X_train, y_train)
    best_tree = grid_tree.best_estimator_
    
    # Predict diabetes using the 'predict()' function.
    prediction = best_tree.predict([[glucose, bp, insulin, bmi, pedigree, age]])
    prediction = prediction[0]

    score = round(grid_tree.best_score_ * 100, 3)

    return prediction, score
def app(diabetes_df):
    st.markdown("<p style='color:red;font-size:25px'>This app uses <b>Decision Tree Classifier</b> for the Early Prediction of Diabetes.", unsafe_allow_html = True) 
    st.subheader("Select Values:")
    
    # Create six sliders with the respective minimum and maximum values of features. 
    # store them in variables 'glucose', 'bp', 'insulin, 'bmi', 'pedigree' and 'age'
    # Write your code here:
    glucose = st.slider("Glucose", float(diabetes_df['Glucose'].min()), float(diabetes_df['Glucose'].max()))
    bp = st.slider("Blood Pressure", float(diabetes_df['Blood_Pressure'].min()), float(diabetes_df['Blood_Pressure'].max()))
    insulin = st.slider("Insulin", float(diabetes_df['Insulin'].min()), float(diabetes_df['Insulin'].max()))
    bmi = st.slider("BMI", float(diabetes_df['BMI'].min()), float(diabetes_df['BMI'].max()))
    pedigree = st.slider("Pedigree Function", float(diabetes_df['Pedigree_Function'].min()), float(diabetes_df['Pedigree_Function'].max()))
    age = st.slider("Age", float(diabetes_df['Age'].min()), float(diabetes_df['Age'].max()))

    st.subheader("Model Selection")

    # Add a single select drop down menu with label 'Select the Classifier'
    predictor = st.selectbox("Select the Decision Tree Classifier",('Decision Tree Classifier', 'GridSearchCV Best Tree Classifier'))

    if predictor == 'Decision Tree Classifier':
        if st.button("Predict"):            
            prediction, score = d_tree_pred(diabetes_df, glucose, bp, insulin, bmi, pedigree, age)
            st.subheader("Decision Tree Prediction results:")
            if prediction == 1:
                st.info("The person either has diabetes or prone to get diabetes")
            else:
                st.info("The person is free from diabetes")
            st.write("The accuracy score of this model is", score, "%")


    elif predictor == 'GridSearchCV Best Tree Classifier':
        if st.button("Predict"):
            prediction, score = grid_tree_pred(diabetes_df, glucose, bp, insulin, bmi, pedigree, age)
            st.subheader("Optimised Decision Tree Prediction results:")
            if prediction == 1:
                st.info("The person either has diabetes or prone to get diabetes")
            else:
                st.info("The person is free from diabetes")
            st.write("The best score of this model is", score, "%")        