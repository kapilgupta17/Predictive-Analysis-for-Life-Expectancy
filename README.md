# Predictive Analysis for Life Expectancy using Machine Learning

This project aims to analyze the life expectancy of populations in different countries using machine learning techniques. The project includes several key components:

**Phase 1:** 
1. Data Cleaning: The dataset for this project is cleaned to ensure that it is suitable for analysis. This includes handling missing values, removing duplicates, and formatting data types.
2. Exploratory Data Analysis (EDA): EDA is conducted to gain insights into the dataset and understand the relationships between variables. This helps in identifying patterns and trends that can be used in the predictive models.
   
**Phase 2:**  

Machine Learning Models: Several machine learning models are used to predict life expectancy based on the available data. These models include:  
-> Logistic Regression  
-> Random Forest  
-> K Nearest Neighbors  
-> Support Vector Regression  
-> Polynomial Regression  
-> Gradient Boosting Regression  

**Phase 3:**

End-Product using Streamlit: The predictive models are integrated into an end-product using Streamlit, a popular Python library for building web applications. This allows users to interact with the models and visualize the results.

**Instructions to run the end-product:**

STEP 1 → Install Streamlit by running the command "pip install streamlit" in your terminal or command prompt.  

STEP 2 → Run the app.py file which has our streamlit code implementation by using the command “python -m streamlit run app_new.py” or “streamlit run app_new.py”.  

STEP 3 → Select the model type and then select the sub model type. An example entry is: Select “Regression” as model type and then select “Random Forest” as the sub model. We can now see the input values boxes appearing. Here for each of the following input type enter the corresponding values,  
Region = Latin America & Caribbean   
Income Group = Upper middle income   
Health Expenditure(%) = 8.37   
Unemployment = 17.32  
Income per capita = 7208.37  
Adult Mortality(per 1000 people) = 7.55  
Education Expenditure(%) = 4.83  
Population = 37275644.0  
GDP = 300421.0  
Infant Deaths(per 1000 live births) = 19.0  
Upon clicking the “predict” button we can now observe the predicted life expectancy.  

STEP 4 → Select the model type and then select the sub model type. An example entry is: Select “Classifier” as model type and then select “K- Nearest Neighbors” as the sub model. We can now see the input values boxes appearing. Here for each of the following input type enter the corresponding values,  
Region = Latin America & Caribbean  
Life Expectancy = 67.0  
Prevalance of Undernourishment = 3.0  
Health Expenditure(%) = 8.37  
Unemployment = 17.32  
Income per capita = 7208.37  
Adult Mortality(per 1000 people) = 7.55  
Education Expenditure(%) = 4.83  
Population = 37275644.0  
GDP = 300421.0  
Infant Deaths(per 1000 live births) = 19.0  
Upon clicking the “predict” button we can now observe the classified income group  

STEP 5 → Select “Model Analytics” to get the information about the error metrics, scatterplots and accuracy vs predicted plots for each of our models to understand the reliability of their predictions and classifications.  

STEP 6 → If we want to see the predictions of all the models with the same values then we can simply change the models in the select option and keep the input values as there are. No need to reenter.  

