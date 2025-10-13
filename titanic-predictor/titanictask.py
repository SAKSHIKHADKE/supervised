###################################################################################################
# Required Libraries
###################################################################################################
import pandas as pd
import numpy as np
import os
import joblib   
import sys
from matplotlib.pyplot import figure,show
import matplotlib.pyplot as plt
import seaborn as sns
from seaborn import countplot
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score,confusion_matrix

###################################################################################################
# File path
###################################################################################################

line = "-" * 42
datapath = "MarvellousTitanicDataset.csv"
model_path =os.path.join("artifacts/titanic_predictor", "titanictask.joblib")
folder_path="artifacts/titanic_predictor"
file_path=folder_path+ "/" "titanic_predictor_report.txt"
###################################################################################################
# Function name =open_folder_file
# description = This function opens a folder and a file for writing.
# author = sakshi khadke
# date = 13-10-2025
###################################################################################################

def open_folder_file(file_path=file_path):
    os.makedirs(folder_path,exist_ok=True)
    file=open(file_path,"w")
    return file
###################################################################################################
# Dataset Headers
###################################################################################################
headers = ['Passengerid', 'Age', 'Fare', 'Sex', 'sibsp', 'Parch', 'zero', 'Pclass', 'Embarked', 'Survived']

###################################################################################################
# Function name = read_csv
# description = This function reads a CSV file and returns a DataFrame.
# author = sakshi khadke
# date = 13-10-2025
###################################################################################################
def read_csv(datapath):
    df=pd.read_csv(datapath)
    return df
###################################################################################################
# Function name = displayhead
# description = This function displays the head of the DataFrame.
# author = sakshi khadke
# date = 13-10-2025
###################################################################################################

def displayhead(df,file):
    file.write(f"{df.head().to_string()}\n")

###################################################################################################
# Function name = displayshape
# description = This function displays the shape of the DataFrame.
# author = sakshi khadke
# date = 13-10-2025
###################################################################################################  
def displayshape(df,file):
    file.write(str(df.shape))
###################################################################################################
# Function name = displayinfo
# description = This function displays the information of the DataFrame.
# author = sakshi khadke
# date = 13-10-2025
###################################################################################################  
def displayinfo(df):
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
    return df
###################################################################################################
# Function name = plotgraph1
# description = This function plots the graph for the DataFrame.
# author = sakshi khadke
# date = 13-10-2025
###################################################################################################  
def plotgraph1(df):
    figure()
    target="Survived"
    countplot(data=df,x=target).set_title("Survived vs nonSurvived")
    plt.savefig("Artifacts/titanic_Predictor/survived_vs_non_survived.png")
    plt.close()

###################################################################################################
# Function name = plotgender
# description = This function plots the gender distribution of the DataFrame.
# author = sakshi khadke
# date = 13-10-2025
###################################################################################################  
def plotgender(df):
    figure()
    target="Survived"
    countplot(data=df,x=target,hue='Sex').set_title("based on gender")
    plt.savefig("Artifacts/titanic_Predictor/gender.png")
    plt.close()

###################################################################################################
# Function name = plotpclass
# description = This function plots the passenger class distribution of the DataFrame.
# author = sakshi khadke
# date = 13-10-2025
###################################################################################################  
def plotpclass(df):
    figure()
    target="Survived"
    countplot(data=df,x=target,hue='Pclass').set_title("based on Pclass")
    plt.savefig("Artifacts/titanic_Predictor/pclass.png")
    plt.close()

###################################################################################################
# Function name = plotage
# description = This function plots the age distribution of the DataFrame.
# author = sakshi khadke
# date = 13-10-2025
###################################################################################################  
def plotage(df):
    figure()
    df['Age'].plot.hist().set_title("Age report") 
    plt.savefig("Artifacts/titanic_Predictor/age.png")
    plt.close()

###################################################################################################
# Function name = plotfare
# description = This function plots the fare distribution of the DataFrame.
# author = sakshi khadke
# date = 13-10-2025
###################################################################################################  
def plotfare(df):
    figure()
    df['Fare'].plot.hist().set_title("Fare report")
    plt.savefig("Artifacts/titanic_Predictor/fare.png")
    plt.close()

###################################################################################################
# Function name = plotheatmap
# description = This function plots the heatmap of the DataFrame.
# author = sakshi khadke
# date = 13-10-2025
###################################################################################################  
def plotheatmap(df):
    plt.figure(figsize=(10,6))
    sns.heatmap(df.corr(),annot=True,cmap='coolwarm')
    plt.title("feature correlation heatmap")
    plt.savefig("Artifacts/titanic_Predictor/feature_correlation_heatmap.png")
    plt.close()

###################################################################################################
# Function name = displayencode
# description = This function displays the encoded version of the DataFrame.
# author = sakshi khadke
# date = 13-10-2025
###################################################################################################  
def displayencode(df):
    df.drop(columns=['Passengerid','zero'],inplace=True)
    return df
###################################################################################################
# Function name = alter
# description = This function alters the features of the DataFrame.
# author = sakshi khadke
# date = 13-10-2025
###################################################################################################  
def alter(df):
    x= df.drop(columns=['Survived'])
    y=df['Survived']
    return x,y

###################################################################################################
# Function name =scaler
# description = This function scales the features of the DataFrame.
# author = sakshi khadke
# date = 13-10-2025
###################################################################################################  
def scaler(x):
    scaler=StandardScaler()
    x_scale=scaler.fit_transform(x)
    return x_scale
###################################################################################################
# Function name = split_data
# description = This function splits the data into training and testing sets.
# author = sakshi khadke
# date = 13-10-2025
###################################################################################################  
def split_data(x_scale,y):
    x_train,x_test,y_train,y_test=train_test_split(x_scale,y,test_size=0.2,random_state=42)
    return x_train,x_test,y_train,y_test

###################################################################################################
# Function name = fit_model
# description = This function fits the model to the training data.
# author = sakshi khadke
# date = 13-10-2025
###################################################################################################  
def fit_model(x_train, y_train):
    # Handle missing values in x_train
    imputer = SimpleImputer(strategy='mean')  # You can also use 'median' or 'most_frequent'
    x_train_imputed = imputer.fit_transform(x_train)

    # Fit the model
    model = LogisticRegression()
    model.fit(x_train_imputed, y_train)

    return model


###################################################################################################
# Function name =save model
# description = This function save the model after training.
# author = sakshi khadke
# date = 13-10-2025
###################################################################################################
def save_model(model,model_path=model_path):
    joblib.dump(model,model_path)
    print(f"model saved at {model_path}")

###################################################################################################
# Function name =load_model
# description = This function load the model for prediction.
# author = sakshi khadke
# date = 13-10-2025
###################################################################################################
def load_model(model_path=model_path):
    model=joblib.load(model_path)
    return model  

###################################################################################################
# Function name = write_accuracy
# description = This function writes the accuracy and confusion matrix to a file.
# author = sakshi khadke
# date = 13-10-2025
###################################################################################################  
def write_accuracy(y_test, y_pred, file):
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    file.write(f"Accuracy: {acc}\n")
    file.write(f"Confusion Matrix:\n{cm}\n")


###################################################################################################
# Function name = main
# description = this function from where execution start
# description = This function preprocesses the dataset and trains the model and calls the given functions.
# author = sakshi khadke
# date = 13-10-2025
###################################################################################################

def main():
    try :
        file=open_folder_file(file_path=file_path)

        df=read_csv(datapath)

        if sys.argv[1]=="--train":

            displayhead(df,file)

            displayshape(df,file)

            displayinfo(df)

            displayencode(df)
            x,y=alter(df)
            scale_x=scaler(x)
            x_train,x_test,y_train,y_test=split_data(scale_x,y)

            model=fit_model(x_train,y_train)

            model=save_model(model,model_path=model_path)

            model=load_model(model_path=model_path)

            y_pred=model.predict(x_test)

            write_accuracy(y_test, y_pred, file)
            acc = accuracy_score(y_test, y_pred)
            cm = confusion_matrix(y_test, y_pred)
            file.write(f"Accuracy: {acc}\n")
            file.write(f"Confusion Matrix:\n{cm}\n")



            plotheatmap(df)

            plotfare(df)

            plotage(df)

            plotpclass(df)

            plotgender(df)

            plotgraph1(df)


        elif sys.argv[1]=="--predict": 
            x,y=alter(df)
            scale_x=scaler(x)
            x_train,x_test,y_train,y_test=split_data(scale_x,y)

            model=fit_model(x_train,y_train)

            model=save_model(model,model_path=model_path)

            model=load_model(model_path=model_path)

            y_pred=model.predict(x_test)

            write_accuracy(y_test, y_pred, file)
            acc = accuracy_score(y_test, y_pred)
            cm = confusion_matrix(y_test, y_pred)
            file.write(f"Accuracy: {acc}\n")
            file.write(f"Confusion Matrix:\n{cm}\n")

        else: 
            print("valid arguments are --train and --predict")
            return

    except FileNotFoundError as e:
       print("Error occurred:", e)
       
if __name__=="__main__":
    main()    