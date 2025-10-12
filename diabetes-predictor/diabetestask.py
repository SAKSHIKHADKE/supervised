from sklearn.model_selection import train_test_split
import numpy as np 
import pandas as pd
from sklearn.linear_model import LogisticRegression
import os
import sys
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import joblib
from sklearn.metrics import accuracy_score
###################################################################################################
# File path
###################################################################################################

line = "-" * 42
datapath = "diabetes.csv"
model_path =os.path.join("artifacts/diabetes_predictor", "diabetestask.joblib")
folder_path="artifacts/diabetes_predictor"
file_path=folder_path+ "/" "diabetes_predictor_report.txt"

###################################################################################################
# Function name =open_folder_file
# description = This function opens a folder and a file for writing.
# author = sakshi khadke
# date = 12-10-2025
###################################################################################################

def open_folder_file(file_path=file_path):
    os.makedirs(folder_path,exist_ok=True)
    file=open(file_path,"w")
    return file
###################################################################################################
# Dataset Headers
###################################################################################################
headers = ['Whether', 'Temperature', 'Play']

###################################################################################################
# Function name = read_csv
# description = This function reads a CSV file and returns a DataFrame.
# author = sakshi khadke
# date = 12-10-2025
###################################################################################################
def read_csv(datapath):
    return pd.read_csv(datapath)

###################################################################################################
# Function name = displayhead
# description = This function displays the head of the DataFrame.
# author = sakshi khadke
# date = 12-10-2025
###################################################################################################

def displayhead(df,file,label="dataset sample is :"):
    file.write(f"{label}\n")
    file.write(f"{df.head().to_string()}\n")

###################################################################################################
# Function name = describe
# description = This function returns the descriptive statistics of the DataFrame.
# author = sakshi khadke
# date = 12-10-2025
###################################################################################################

def describe(datapath,file):
    file.write(line)
    file.write(datapath.describe().to_string())
    file.write("\n")

###################################################################################################
# Function name = columns
# description = This function returns the columns of the DataFrame.
# author = sakshi khadke
# date = 12-10-2025
###################################################################################################

def columns(datapath,file):
    file.write(line)
    file.write(datapath.columns.to_series().to_string())
    file.write("\n")

###################################################################################################
# Function name =datatypes
# description = This function returns the data types of the columns in the DataFrame.
# author = sakshi khadke
# date = 12-10-2025
###################################################################################################

def datatypes(datapath,file):
    file.write(line)
    file.write(datapath.dtypes.to_string())
    file.write("\n")    
###################################################################################################
# Function name = encode
# description = This function encode a CSV file and returns a encoded DataFrame.
# author = sakshi khadke
# date = 12-10-2025
###################################################################################################

def encode(df) :
    df.dropna(inplace=True)
    return df

###################################################################################################
# Function name = alter
# description = This function alters the DataFrame for model training.
# author = sakshi khadke
# date = 12-10-2025
###################################################################################################

def alter(df) :
    x=df.drop(columns=['target'])
    y=df['target']
    return x,y
###################################################################################################
# Function name = scaler
# description = This function scale the DataFrame for model training.
# author = sakshi khadke
# date = 12-10-2025
###################################################################################################

def scaler(x) :

    scaler=StandardScaler()
    x_scale=scaler.fit_transform(x)

    return x_scale

###################################################################################################
# Function name = split_data
# description = This function splits the DataFrame for model training.
# author = sakshi khadke
# date = 12-10-2025
###################################################################################################
def split_data(x,y) :
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
    return x_train,x_test,y_train,y_test

###################################################################################################
# Function name = fit_model
# description = This function fits the model fo training.
# author = sakshi khadke
# date = 12-10-2025
###################################################################################################

def fit_model(x_train,y_train):
    model=KNeighborsClassifier(n_neighbors=50)
    model.fit(x_train,y_train)
    return model

###################################################################################################
# Function name =save model
# description = This function save the model after training.
# author = sakshi khadke
# date = 12-10-2025
###################################################################################################
def save_model(model,model_path=model_path):
    joblib.dump(model,model_path)
    print(f"model saved at {model_path}")

###################################################################################################
# Function name =load_model
# description = This function load the model for prediction.
# author = sakshi khadke
# date =12-10-2025
###################################################################################################
def load_model(model_path=model_path):
    model=joblib.load(model_path)
    return model    
###################################################################################################
# Function name = find best k
# description = This function find best k for the DataFrame for model training.
# author = sakshi khadke
# date = 12-10-2025
###################################################################################################

def find_best_k(y_test,y_pred) :
    accuracy_scores=[]
    k_range=range(1,21)

    for k in k_range:
        accuracy=accuracy_score(y_test,y_pred)
        accuracy_scores.append(accuracy)

    return k_range,accuracy_scores

###################################################################################################
# Function name = plot
# description = This function plot the accuracy of the model.
# author = sakshi khadke
# date = 12-10-2025
###################################################################################################
def plot(k_range,accuracy_scores):

    plt.figure(figsize=(8,5))
    plt.plot(k_range,accuracy_scores,marker='o',linestyle='--')
    plt.title("accuracy of k value")
    plt.xlabel("value of k")
    plt.ylabel("accuracy of model")
    plt.grid(True)
    plt.xticks(k_range)
    plt.savefig("Artifacts/Diabetes_Predictor/K_value_vs_Accuracy.png")
    plt.close()
###################################################################################################
# Function name = displaycorrelation
# description = this function display correlation 
# description = This function preprocesses the dataset and trains the model and calls the given functions.
# author = sakshi kedari
# date = 12-10-2025
###################################################################################################
def DisplayCorrelation(df, file) :
        file.write("co-relation matrice : \n")
        file.write(df.corr().to_string())
        file.write("\n")

        plt.figure(figsize=(10,6))
        sns.heatmap(df.corr(), annot = True, cmap = 'coolwarm')
        plt.title("Diabetes Predictor")
        plt.savefig("Artifacts/Diabetes_Predictor/Correlation_plot.png")
        plt.close()
###################################################################################################
# Function name = displaypairplot
# description = this function display pairplot
# description = This function preprocesses the dataset and trains the model and calls the given functions.
# author = sakshi kedari
# date = 12-10-2025
#######################################################################################
def DisplayPairplot(df) :

    sns.pairplot(df)
    plt.suptitle("pairplot of feature", y  = 1.02)
    plt.savefig("Artifacts/Diabetes_Predictor/Pairplot_plot.png")
    plt.close()    
###################################################################################################
# Function name = main
# description = this function from where execution start
# description = This function preprocesses the dataset and trains the model and calls the given functions.
# author = sakshi kedari
# date = 12-10-2025
#######################################################################################

def main():
    try:
        file=open_folder_file(file_path)
        df=read_csv(datapath)

        if sys.argv[1]=="--train":
       
            displayhead(df,file)
            describe(df,file)
            columns(df,file)
            datatypes(df,file)
            df=encode(df)
            displayhead(df,file,label="after encoding dataset sample is :")
            x,y=alter(df)
            x=scaler(x)
            x_train,x_test,y_train,y_test=split_data(x,y)
            model=fit_model(x_train,y_train)
            save_model(model,model_path)
            model=load_model(model_path)
            y_pred=model.predict(x_test)
            accuracy=accuracy_score(y_test,y_pred)
            file.write(f"accuracy of model is : {accuracy}\n")

        elif sys.argv[1]=="--predict":
                x,y=alter(df)
                x=scaler(x)
                x_train,x_test,y_train,y_test=split_data(x,y)
                model=fit_model(x_train,y_train)
                y_pred=model.predict(x_test)
                accuracy=accuracy_score(y_test,y_pred)
                file.write(f"accuracy of model is : {accuracy}\n")

                k_range,accuracy_scores=find_best_k(y_test,y_pred)
                plot(k_range,accuracy_scores)
                DisplayCorrelation(df,file)
                DisplayPairplot(df)

        else: 
            print("valid arguments are --train and --predict")
            return

    except FileNotFoundError as e:
       print("Error occurred:", e)
       
   


if __name__=="__main__":
    main()    