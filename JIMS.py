#import required libraries
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
import plotly.graph_objects as go
from PIL import Image
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

st.set_page_config(layout='wide',initial_sidebar_state='collapsed')


model_data = pd.read_csv("Data/mod_csv_res.csv")
model_data.drop(['Unnamed: 0'],axis=1,inplace=True)
deps = pd.read_csv('Data/Deps_mod_res.csv')

###############################################################################################
model_excel_rating = st.container(border=True)
model_excel_rating.markdown('<div style="text-align: center; font-size: 24px">Predictive Model on Excellent Ratings</div>',unsafe_allow_html=True)
model_excel_rating.divider()
excel_class,excel_chart = model_excel_rating.columns([.35,.65]) 
excel_class_ = excel_class.container(border=True)
excel_chart_ = excel_chart.container(border=True)


#from sklearn.ensemble import RandomForestClassfier
scaler = StandardScaler()
smt = SMOTE()
selected_vars = excel_chart_.multiselect('Select Variables for the Model:',model_data.columns.tolist(),default=model_data.columns.tolist())
#training the model on selected features and then 
model_lr = LogisticRegression()
req_data = model_data[selected_vars]
X_Scaled = scaler.fit_transform(req_data)
X, y = smt.fit_resample(X_Scaled,deps['Excellent'])
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25)
model_lr.fit(X_train,y_train)
y_pred_rfc = model_lr.predict(X_test)


#visualisation of accracy
cm = confusion_matrix(y_test, y_pred_rfc)
# Create the heatmap plot using Plotly Express
con_mat = px.imshow(cm, labels=dict(x="Predicted", y="True"), x=['Not Followed', 'Followed'], y=['Not Followed', 'Followed'], color_continuous_scale='Blues',text_auto=True,title='Confusion Matrix Logitic Regression Excellent Ratings')
# Update the color axis to hide the scale
con_mat.update_coloraxes(showscale=False)
#creating a container for model_accuracy
excel_class_.plotly_chart(con_mat,use_container_width=True)
excel_class_.divider()
excel_class_.text(classification_report(y_test,y_pred_rfc))

#Faeture importance plot
# Extract coefficients and feature names
coefficients = model_lr.coef_[0]
feature_names = [f'{i}' for i in model_data[selected_vars].columns.to_list()]
# Create a DataFrame with feature names and coefficients
df_coefficients = pd.DataFrame({'Feature': feature_names, 'Coefficient Value': coefficients.round(2)})

# Sort the DataFrame by coefficient values
df_coefficients_sorted = df_coefficients.sort_values(by='Coefficient Value',ascending=False)

# Create the bar plot using Plotly Express
feat_importance = px.bar(df_coefficients_sorted, 
             y='Coefficient Value', 
             x='Feature', 
             orientation='v', 
             title='Feature Importance in Logistic Regression',
             labels={'Coefficient Value': 'Coefficient Value', 'Feature': 'Feature'},color='Feature',text='Coefficient Value')


# Show the plot
#model_feat_imp_.plotly_chart(feat_importance,use_container_width=True)
excel_chart_.plotly_chart(feat_importance,use_container_width=True)
