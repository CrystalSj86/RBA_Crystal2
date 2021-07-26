import streamlit as st
import pandas as pd
import numpy as np
import math
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import seaborn as sns
import time

st.header("Welcome to my first data science project!")
st.text('In this project I would like to find out the variable that affect KL property pricing')

st.sidebar.write('Dataset:')
st.sidebar.write("<a href= 'https://www.kaggle.com/dragonduck/property-listings-in-kuala-lumpur/'> Property Listing in Kuala Lumpur </a>",unsafe_allow_html=True)

st.sidebar.write('Source of Dataset:')
st.sidebar.write("<a href='https://www.kaggle.com/'> Kaggle </a>", unsafe_allow_html=True)

st.sidebar.write ("For more info, please contact:")
st.sidebar.write("<a href='https://www.linkedin.com/in/CrystalNgSj/'>Crystal Ng </a>", unsafe_allow_html=True)

option = st.sidebar.selectbox(   
    'Select a mini project',
     ['Bar Chart & Dataset','Correlation Matrix Plots','Random Forest'])


if option=='Bar Chart & Dataset':
    #kl_property = pd.read_csv('/Users/crystalng/Desktop/Python/RBA/klproperty_EncodedData.csv')
    kl_property = pd.read_csv('klproperty_EncodedData.csv')
    
    st.subheader('KL Property Location Distribution')
    location_dist = pd.DataFrame(kl_property['Location'].value_counts()).head(50)
    st.bar_chart(location_dist)
    st.write(kl_property)
    
elif option=='Correlation Matrix Plots':
    FloatData = pd.read_csv('klproperty_FloatData.csv')
    #FloatData = pd.read_csv('/Users/crystalng/Desktop/Python/RBA/klproperty_FloatData.csv')
    
    st.subheader('Correlation Heatmap')
    
    corr = FloatData.corr()
    fig3 = plt.figure(figsize=(16, 6))
    heatmap =sns.heatmap(corr, vmin=-1, vmax=1, annot=True)
    st.pyplot(fig3)
    
    st.subheader('Sorting The Correlation Heatmap')
    corr_pairs = corr.unstack()
    st.write(corr_pairs)

elif option=='Random Forest':

    st.header('Train the model with Random Forest')
    st.text('Here you may choose the hyperparameters of the model and observe the performance change')
    
    sel_col, disp_col = st.beta_columns(2)
    max_depth = sel_col.slider('What should be the max_depth of the model?', 
                               min_value =10, max_value =100, value =20, step =10)
    
    n_estimators = sel_col.slider('How many trees should there be?', 
                                  min_value =10, max_value =100, value =10, step =10)
    
    input_feature = sel_col.multiselect(
        'Which feature should be used as the input feature? \n' 'You may select more than one feature', 
        options= ['Location', 'Bathrooms', 'CarParks', 'Furnishing', 'NoOfRoom', 'SimplifyPropertyType', 'Sqft', 'PricePerSqft'], default=['Location'])
    st.write('You selected:', input_feature)
    
    RandomForest = RandomForestRegressor(max_depth= max_depth, n_estimators= n_estimators)
    
    kl_property = pd.read_csv('klproperty_EncodedData.csv')
    #kl_property = pd.read_csv('/Users/crystalng/Desktop/Python/RBA/klproperty_EncodedData.csv')
    
    X = kl_property.loc[:,input_feature]
    y = kl_property.loc[:,'Price']
    X_train, X_test, y_train, y_test = train_test_split(X, y,random_state=1)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)    
    
    RandomForest.fit(X_train_scaled, y_train)
    y_pred = RandomForest.predict(X_test_scaled)
     
    disp_col.subheader('Mean squared error of the model is:')
    disp_col.write("%.2f" % mean_squared_error(y_test, y_pred))
    
    disp_col.subheader('Variance score of the model is:')
    disp_col.write("%.2f" % r2_score(y_test,y_pred))