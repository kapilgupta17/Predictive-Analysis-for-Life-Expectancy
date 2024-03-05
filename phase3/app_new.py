#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 12:23:02 2023

@author: kapilgupta
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score,precision_score,recall_score,auc,roc_curve
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import label_binarize
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
import pickle


import seaborn as sns
from sklearn.utils import shuffle
from sklearn.model_selection import GridSearchCV
from itertools import cycle
from sklearn.impute import SimpleImputer

def regression_input():
    reg_cat=None
    inc_cat=None
    region=st.selectbox("Region",('Select Region','Sub-Saharan Africa', 'Europe & Central Asia',
       'Middle East & North Africa', 'Latin America & Caribbean',
       'East Asia & Pacific', 'South Asia', 'North America'))
    if(region=="Sub-Saharan Africa"):
        reg_cat=6
    elif(region=="Europe & Central Asia"):
        reg_cat=1
    elif(region=="Middle East & North Africa"):
        reg_cat=3
    elif(region=="Latin America & Caribbean"):
        reg_cat=2
    elif(region=="East Asia & Pacific"):
        reg_cat=0
    elif(region=="South Asia"):
        reg_cat=5
    elif(region=="North America"):
        reg_cat=4
    
    income_grp=st.selectbox("Income Group", ('Select Income Group','Lower middle income', 'Upper middle income', 'High income',
       'Low income'))
    if(income_grp=="Lower middle income"):
        inc_cat=2
    elif(income_grp=="Upper middle income"):
        inc_cat=3
    elif(income_grp=="High income"):
        inc_cat=0
    elif(income_grp=="Low income"):
        inc_cat=1
        
    preve=st.text_input("Prevelance of Undernourishment",key='preve')
    
    health = st.text_input("Health Expenditure(%)",key='health')
    unemp = st.text_input("Unemployment",key='unemp')
    inc_cap=st.text_input("Income per capita",key='inc_cap')
    adu=st.text_input("Adult Mortality (per 1000 people)",key='adu')
    edu=st.text_input("Education Expenditure(%)",key='edu')
    pop=st.text_input("Population",key='pop')
    gdp=st.text_input("GDP",key='gdp')
    inf_d=st.text_input("Infant Deaths (per 1000 live births)",key="inf_d")
    
    input_data=[reg_cat,inc_cat,float(preve),float(health),float(unemp),float(inc_cap),float(adu),float(edu),float(pop),float(gdp),float(inf_d)]
    columns = ['Region', 'IncomeGroup', 'Prevelance of Undernourishment', 'Health Expenditure %', 'Unemployment',
               'Income per capita', 'Adult Mortality(per 1000 people)', 'Education expenditure(%)', 'Population',
               'GDP', 'Infant deaths(per 1000 live births)']

    df_input = pd.DataFrame([input_data], columns=columns)
    return df_input

def classifier_input():
    reg_cat=None
    region=st.selectbox("Region",('Select Region','Sub-Saharan Africa', 'Europe & Central Asia',
       'Middle East & North Africa', 'Latin America & Caribbean',
       'East Asia & Pacific', 'South Asia', 'North America'))
    if(region=="Sub-Saharan Africa"):
        reg_cat=6
    elif(region=="Europe & Central Asia"):
        reg_cat=1
    elif(region=="Middle East & North Africa"):
        reg_cat=3
    elif(region=="Latin America & Caribbean"):
        reg_cat=2
    elif(region=="East Asia & Pacific"):
        reg_cat=0
    elif(region=="South Asia"):
        reg_cat=5
    elif(region=="North America"):
        reg_cat=4
    li_ex=st.text_input("Life Expectancy",key='li_ex')
    preve=st.text_input("Prevelance of Undernourishment",key='preve')
    health = st.text_input("Health Expenditure(%)",key='health')
    unemp = st.text_input("Unemployment",key='unemp')
    inc_cap=st.text_input("Income per capita",key='inc_cap')
    adu=st.text_input("Adult Mortality (per 1000 people)",key='adu')
    edu=st.text_input("Education Expenditure(%)",key='edu')
    pop=st.text_input("Population",key='pop')
    gdp=st.text_input("GDP",key='gdp')
    inf_d=st.text_input("Infant Deaths (per 1000 live births)",key="inf_d")
    
    input_data=[reg_cat,float(li_ex),float(preve),float(health),float(unemp),float(inc_cap),float(adu),float(edu),float(pop),float(gdp),float(inf_d)]
    columns = ['Region','Life Expectancy World Bank','Prevelance of Undernourishment','Health Expenditure %','Unemployment','Income per capita','Adult Mortality(per 1000 people)',
               'Education expenditure(%)','Population','GDP','Infant deaths(per 1000 live births)']

    df_input = pd.DataFrame([input_data], columns=columns)
    
    return df_input


def rf_analytics_input():
    
    main_df=pd.read_csv('Life Expectancy Dataset(cleaned_encoded_normalized).csv')
    main_df = pd.DataFrame(main_df)
    data = main_df.copy()
    data = data.dropna()
    random_forest_data = data.copy()
    data.drop(columns=['Country'], inplace=True)
    data.drop(columns=['Country Code'], inplace=True)
    df_shuffled = data.sample(frac=1, random_state=42)
    train_size = 0.8
    test_size = 0.2
    train_data, test_data = train_test_split(df_shuffled, test_size=test_size, random_state=42)
    
    target_column = 'Life Expectancy World Bank'

    X_train = train_data.drop(target_column, axis=1)
    y_train = train_data[target_column]

    X_test = test_data.drop(target_column, axis=1)
    y_test = test_data[target_column]
    
    imputer = SimpleImputer(strategy='mean')
    X_train_imputed = imputer.fit_transform(X_train)
    X_test_imputed = imputer.transform(X_test)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imputed)
    X_test_scaled = scaler.transform(X_test_imputed)
        
    
    st.title("Machine Learning Model Analysis on Life Expectancy")

    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train_scaled, y_train)

    # Make predictions
    y_pred_rf = rf_model.predict(X_test_scaled)

    # Display the results
    st.write("Mean Squared Error:", mean_squared_error(y_test, y_pred_rf))
    st.write("R-squared (R^2):", r2_score(y_test, y_pred_rf))

    # Feature Importance
    feature_importance_rf = rf_model.feature_importances_
    features_rf = X_test.columns

    # Display Feature Importance
    st.subheader("Feature Importance (Random Forest)")
    feature_importance_df_rf = pd.DataFrame({'Feature': features_rf, 'Importance': feature_importance_rf})
    feature_importance_df_rf = feature_importance_df_rf.sort_values(by='Importance', ascending=False)
    st.table(feature_importance_df_rf)

    # Bar Plot of Feature Importance
    st.subheader("Bar Plot of Feature Importance (Random Forest)")
    fig_bar = plt.figure(figsize=(10, 6))
    plt.barh(features_rf, feature_importance_rf, color='skyblue')
    plt.xlabel('Importance')
    plt.title('Feature Importance (Random Forest)')
    st.pyplot(fig_bar)

    # Scatter Plot between Actual vs Predicted values
    st.subheader("Scatter Plot between Actual vs Predicted values")
    fig_scatter = plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred_rf, c='green')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Scatter Plot of Predicted vs. Actual Values (Random Forest)')
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], linestyle='--', color='red', label='Perfect Prediction')
    plt.legend()
    st.pyplot(fig_scatter)
    return


def svr_analytics_input():
    
    main_df=pd.read_csv('Life Expectancy Dataset(cleaned_encoded_normalized).csv')
    main_df = pd.DataFrame(main_df)

    dfy2=main_df['Life Expectancy World Bank']
    dfx2=main_df.drop(['Life Expectancy World Bank','Country','Year','Country Code'],axis=1)
        
    dfx2_train,dfx2_test,dfy2_train,dfy2_test = train_test_split(dfx2,dfy2,test_size=0.2)
        
    
    st.title("Machine Learning Model Analysis on Life Expectancy")
    
    svr = SVR(kernel='rbf', C=1.0, epsilon=0.05)
    svr.fit(dfx2_train,dfy2_train)
    y_pred_svr=svr.predict(dfx2_test)

    # Display the results
    st.write("Mean Squared Error:", mean_squared_error(dfy2_test, y_pred_svr))
    st.write("R-squared (R^2):", r2_score(dfy2_test, y_pred_svr))

    # Scatter Plot between Actual vs Predicted values
    st.subheader("Scatter Plot between Actual vs Predicted values")
    fig_scatter_svr = plt.figure(figsize=(8, 6))
    plt.scatter(dfy2_test, y_pred_svr, c='blue')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Scatter Plot of Predicted vs. Actual Values (SVR)')
    plt.plot([min(dfy2_test), max(dfy2_test)], [min(dfy2_test), max(dfy2_test)], linestyle='--', color='red', label='Perfect Prediction')
    plt.legend()
    st.pyplot(fig_scatter_svr)

    # Error Distribution Plot
    st.subheader("Error Distribution Plot")
    fig_error_svr = plt.figure(figsize=(8, 6))
    residuals_svr = np.array(dfy2_test) - np.array(y_pred_svr)
    plt.hist(residuals_svr, bins=20, edgecolor='k')
    plt.xlabel('Residuals (Errors)')
    plt.ylabel('Frequency')
    plt.title('Error Distribution Plot (Residuals Histogram) - SVR')
    st.pyplot(fig_error_svr)
    return

def pr_analytics_input():
    
    main_df=pd.read_csv('Life Expectancy Dataset(cleaned_encoded_normalized).csv')
    main_df = pd.DataFrame(main_df)
    
    df=main_df.copy()
    df = shuffle(df, random_state=42)
    X_poly = df[['CO2','Health Expenditure %', 'Unemployment', 'Income per capita', 'Education expenditure(%)','GDP', 'Infant deaths(per 1000 live births)']]
    y_poly = df['Life Expectancy World Bank']
    X_train, X_test, y_train, y_test = train_test_split(X_poly, y_poly, test_size=0.2, random_state=42)
    
    st.title("Machine Learning Model Analysis on Life Expectancy")
    

    degree = 2
    poly = PolynomialFeatures(degree=degree)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)

    model = LinearRegression()
    model.fit(X_train_poly, y_train)
    y_pred = model.predict(X_test_poly)


    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f'Mean Squared Error: {mse:.2f}')
    print(f'R-squared (R2) Score: {r2:.2f}')

    # Display the results
    st.write(f'Mean Squared Error: {mean_squared_error(y_test, y_pred):.2f}')
    st.write(f'R-squared (R2) Score: {r2_score(y_test, y_pred):.2f}')

    # Residual plot
    st.subheader("Residual plot")
    fig_residual_poly = plt.figure(figsize=(8, 6))
    residuals_poly = y_test - y_pred
    plt.scatter(y_pred, residuals_poly)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title('Residual Plot (Polynomial Regression)')
    st.pyplot(fig_residual_poly)

    # Scatter plot between Actual vs Predicted Values
    st.subheader("Scatter plot between Actual vs Predicted Values")
    fig_scatter_poly = plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--', label='Perfect Predictions (Diagonal)')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Actual vs. Predicted Values (Polynomial Regression)')
    st.pyplot(fig_scatter_poly)
    return

def gbr_analytics_input():
    
    main_df=pd.read_csv('Life Expectancy Dataset(cleaned_encoded_normalized).csv')
    main_df = pd.DataFrame(main_df)
    
    df=main_df.copy()
    df = shuffle(df, random_state=42)
    
    X_gbr = df[['CO2','Health Expenditure %', 'Unemployment', 'Income per capita', 'Education expenditure(%)','GDP', 'Infant deaths(per 1000 live births)']]
    y_gbr = df['Life Expectancy World Bank']
    X_train, X_test, y_train, y_test = train_test_split(X_gbr, y_gbr, test_size=0.2, random_state=42)

    gbr_model = GradientBoostingRegressor(n_estimators=100,learning_rate=0.01, max_depth=5,min_samples_split=3,min_samples_leaf=3,random_state=41)

    gbr_model.fit(X_train, y_train)
    #k-fold cross-validation
    cv_scores_r2 = cross_val_score(gbr_model, X_gbr, y_gbr, cv=5, scoring='r2')
    print(f'Average Cross-Validated R-squared (R2) Score: {np.mean(cv_scores_r2):.2f}')
    for i, r2 in enumerate(cv_scores_r2, start=1):
        print(f'Fold {i} - R2: {r2:.2f}')
    
    st.title("Machine Learning Model Analysis on Life Expectancy")
    
    y_pred_gb = gbr_model.predict(X_test)

    # Display the results
    st.write("Mean Squared Error:", mean_squared_error(y_test, y_pred_gb))
    st.write("R-squared (R^2):", r2_score(y_test, y_pred_gb))

    # Scatter Plot between Actual vs Predicted values
    st.subheader("Scatter Plot between Actual vs Predicted values")
    fig_scatter_gb = plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred_gb, c='green')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Scatter Plot of Predicted vs. Actual Values (Gradient Boosting Regression)')
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], linestyle='--', color='red', label='Perfect Prediction')
    plt.legend()
    st.pyplot(fig_scatter_gb)

    # Feature Importance Plot
    st.subheader("Feature Importance Plot")
    feature_importance_gb = gbr_model.feature_importances_
    feature_names_gb = X_test.columns
    fig_feature_importance_gb = plt.figure(figsize=(10, 6))
    sns.barplot(x=feature_importance_gb, y=feature_names_gb)
    plt.title('Feature Importance Plot (Gradient Boosting Regression)')
    st.pyplot(fig_feature_importance_gb)

    return

def lr_analytics_input():
    
    main_df=pd.read_csv('Life Expectancy Dataset(cleaned_encoded_normalized).csv')
    main_df = pd.DataFrame(main_df)
    
    data3 = main_df.copy()
    data3.drop(["Country"], axis=1, inplace=True)
    data3.drop(["Country Code"], axis=1, inplace=True)
    data3.head()
    
    X = data3.drop("Life Expectancy World Bank", axis=1)
    y = data3["Life Expectancy World Bank"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    median_life_expectancy = data3["Life Expectancy World Bank"].median()
    y_train_binary = (y_train > median_life_expectancy).astype(int)
    y_test_binary = (y_test > median_life_expectancy).astype(int)

    model = LogisticRegression()

    model.fit(X_train, y_train_binary)

    y_pred = model.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test_binary, y_pred)
    st.write(f"Accuracy: {accuracy:.2f}")
    st.write("Classification Report:")
    st.text(classification_report(y_test_binary, y_pred))
    
    # Confusion Matrix Visualization
    st.subheader("Confusion Matrix Visualization")
    conf_matrix = confusion_matrix(y_test_binary, y_pred)
    fig_conf_matrix = plt.figure(figsize=(8, 6))
    plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    classes = ['Below Median', 'Above Median']
    tick_marks = [0.5, 1.5]
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    for i in range(len(classes)):
        for j in range(len(classes)):
            plt.text(j, i, str(conf_matrix[i, j]), horizontalalignment="center", color="white" if conf_matrix[i, j] > conf_matrix.max() / 2 else "black")
    st.pyplot(fig_conf_matrix)

    return


def knn_analytics_input():
    
    main_df=pd.read_csv('Life Expectancy Dataset(cleaned_encoded_normalized).csv')
    main_df = pd.DataFrame(main_df)

    # Selecting target column
    df_y = main_df['IncomeGroup']
    df_x = main_df.drop(['IncomeGroup', 'Country', 'Country Code', 'Year', 'Region'], axis=1)

    # Splitting the data
    dfx_train, dfx_test, dfy_train, dfy_test = train_test_split(df_x, df_y, test_size=0.2)

    # Creating the model
    knn = KNeighborsClassifier(n_neighbors=4)
    knn.fit(dfx_train, dfy_train)
    knn_pred = knn.predict(dfx_test)

    # Evaluate the model
    accuracy = accuracy_score(dfy_test, knn_pred)
    st.write(f"Accuracy: {accuracy:.2f}")
    st.write("Classification Report:")
    st.text(classification_report(dfy_test, knn_pred))

    # Confusion Matrix Visualization
    st.subheader("Confusion Matrix Visualization")
    labels = [0, 1, 2, 3]
    conf_matrix = confusion_matrix(dfy_test, knn_pred)
    fig_conf_matrix = plt.figure(figsize=(8, 6))
    conf = sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    conf.set_xticklabels(labels)
    conf.set_yticklabels(labels)
    conf.set_xlabel('Predicted Label')
    conf.set_ylabel('True Label')
    conf.set_title('Confusion Matrix')
    st.pyplot(fig_conf_matrix)

    return


def normalize_data(input_data,isregression):
    
    min_features = np.array([3.7, 2.5, 1.8, 0.14, 10.51, 0.8, -0.76, 17.93, 13.18, 2.3])
    max_features = np.array([84.36, 31.7, 13.28, 19.5, 36795.98, 15.56, 7.67, 61120128.0, 526014.0, 117.8])
    
    if isregression:
        min_features=min_features[1:]
        max_features=max_features[1:]
        features=['Prevelance of Undernourishment', 
       'Health Expenditure %', 'Unemployment', 'Income per capita',
       'Adult Mortality(per 1000 people)', 'Education expenditure(%)',
       'Population', 'GDP', 'Infant deaths(per 1000 live births)']
        input_data[features] = (input_data[features] - min_features) / (max_features - min_features)
    else:
        features=['Life Expectancy World Bank', 'Prevelance of Undernourishment', 
       'Health Expenditure %', 'Unemployment', 'Income per capita',
       'Adult Mortality(per 1000 people)', 'Education expenditure(%)',
       'Population', 'GDP', 'Infant deaths(per 1000 live births)']
        input_data[features] = (input_data[features] - min_features) / (max_features - min_features)



    #normalized_input=normalized_input.reshape(-1,1)
    return input_data

def random_forest(input_data):
    scaled_input = normalize_data(input_data, True)
    with open('random_forest_model.pkl', 'rb') as model_file:
        rf_model = pickle.load(model_file)
    
    pred=rf_model.predict(scaled_input)
    
    st.write("Life expectancy: ",str(pred*100))

def svr(input_data):
    scaled_input = normalize_data(input_data, True)
    with open('svr.pkl', 'rb') as model_file:
        svr_model = pickle.load(model_file)
    
    pred=svr_model.predict(scaled_input)
    
    st.write("Life expectancy: ",str(pred*100))

def poly_reg(input_data):
    scaled_input = normalize_data(input_data, True)
    with open('poly_reg_model.pkl', 'rb') as model_file:
        saved_model = pickle.load(model_file)

    poly = saved_model['poly']
    regression_model = saved_model['regression_model']


    scaled_input_poly = poly.transform(scaled_input)
    pred = regression_model.predict(scaled_input_poly)
    st.write("Life expectancy: ",str(pred*100))

def gb_reg(input_data):
    scaled_input = normalize_data(input_data, True)
    with open('gbr.pkl', 'rb') as model_file:
        gbr_model = pickle.load(model_file)
    
    pred=gbr_model.predict(scaled_input)
    st.write("Life expectancy: ",str(pred*100))

def log_reg(input_data):
    scaled_input = normalize_data(input_data, False)
    with open('logreg.pkl', 'rb') as model_file:
        logreg_model = pickle.load(model_file)
    
    pred=logreg_model.predict(scaled_input)
    pred_cat=None
    if(pred==0):
        pred_cat="High income"
    elif(pred==1):
        pred_cat="Low income"
    elif(pred==2):
        pred_cat="Lower middle income"
    elif(pred==3):
        pred_cat="Upper middle income"
    
    st.write("Income Group: ",pred_cat)


def knn_model(input_data):
    scaled_input = normalize_data(input_data, False)
    
    with open('knn.pkl', 'rb') as model_file:
        knn = pickle.load(model_file)
    
    pred=knn.predict(scaled_input)
    pred_cat=None
    if(pred==0):
        pred_cat="High income"
    elif(pred==1):
        pred_cat="Low income"
    elif(pred==2):
        pred_cat="Lower middle income"
    elif(pred==3):
        pred_cat="Upper middle income"
    
    st.write("Income Group: ",pred_cat)

st.title("Life expectancy Databank")
st.sidebar.header("User Input")


st.sidebar.subheader("Choose Model Type")

model_type = st.sidebar.selectbox("Select Model type",("Select","Classifier","Regression", "Model Analytics"))

if(model_type=="Regression"):
    selected_model = st.sidebar.selectbox("Select a regression model",
                                              ("Select","Random Forest", "Support Vector Regression", "Polynomial Regression", "Gradient Boosting Regression"),)
    clicked=st.sidebar.button("Predict")
    input_data=regression_input()
    
    if selected_model == "Random Forest":
        if(clicked):
            random_forest(input_data)
    elif selected_model == "Support Vector Regression":
        if(clicked):
            svr(input_data)
    elif selected_model == "Polynomial Regression":
        if(clicked):
            poly_reg(input_data)
    elif selected_model == "Gradient Boosting Regression":
        if(clicked):
            gb_reg(input_data)

elif(model_type=="Classifier"):
    selected_model = st.sidebar.selectbox("Select a classifier model",
                                              ("Select","K-Nearest Neighbors", "Logistic Regression"))
    clicked=st.sidebar.button("Predict")
    input_data=classifier_input()
    
    if selected_model == "K-Nearest Neighbors":
        if(clicked):
            knn_model(input_data)
    elif selected_model == "Logistic Regression":
        if(clicked):
            log_reg(input_data)
            
            
elif(model_type=="Model Analytics"):
    selected_model = st.sidebar.selectbox("Select a model to view analytics",
                                              ("Select", "Random Forest", "Support Vector Regression", "Polynomial Regression", "Gradient Boosting Regression", "K-Nearest Neighbors", "Logistic Regression"))
    clicked=st.sidebar.button("Show Analytics")
    # input_data=rf_analytics_input()
    
    # Linear Regression
    if selected_model == 'Random Forest':
        if(clicked):
            # knn_model(input_data)
            st.subheader("Model 1 - Random Forest Regression")
            rf_analytics_input()

    # Support Vector Regression
    elif selected_model == 'Support Vector Regression':
        if (clicked):
            st.subheader("Model 2 - Support Vector Regression")
            svr_analytics_input()

    # Polynomial Regression
    elif selected_model == 'Polynomial Regression':
        if (clicked):
            st.subheader("Model 3 - Polynomial Regression")
            pr_analytics_input()

    # Gradient Boosting Regression
    elif selected_model == 'Gradient Boosting Regression':
        if (clicked):
            st.subheader("Model 4 - Gradient Boosting Regression")
            gbr_analytics_input()

    # Logistic Regression
    elif selected_model == 'Logistic Regression':
        if (clicked):
            st.subheader("Model 5 - Logistic Regression")
            lr_analytics_input()
    
    # KNN
    elif selected_model == 'K-Nearest Neighbors':
        if (clicked):
            st.subheader("Model 6 - K-Nearest Neighbors")
            knn_analytics_input()



        
elif(model_type=="Select"):
    st.subheader("Welcome to Life expectancy project")



# Main content

