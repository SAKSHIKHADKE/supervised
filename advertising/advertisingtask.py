import os
import sys
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor

import seaborn as sns
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
###################################################################################################
# File path
###################################################################################################

line = "-" * 42
datapath = "advertising.csv"
model_path =os.path.join("artifacts/advertising", "advertisingtask.joblib")
folder_path="artifacts/advertising"
file_path=folder_path+ "/" "advertising_report.txt"
###################################################################################################
# Function name =open_folder_file
# description = This function opens a folder and a file for writing.
# author = sakshi khadke
# date = 3-10-2025
###################################################################################################

def open_folder_file(file_path=file_path):
    os.makedirs(folder_path,exist_ok=True)
    file=open(file_path,"w")
    return file

###################################################################################################
# Dataset Headers
###################################################################################################
headers=['TV','radio','newspaper','sales']

###################################################################################################
# Function name =read_csv
# description = This function reads a CSV file and returns a DataFrame.
# author = sakshi khadke
# date = 4-10-2025
###################################################################################################

def read_csv(datapath):
    return pd.read_csv(datapath)

###################################################################################################
# Function name =displayhead
# description = This function displays the head of the DataFrame.
# author = sakshi khadke
# date = 4-10-2025
###################################################################################################

def displayhead(datapath,file):
    file.write(line)
    file.write("dataset sample is:")
    file.write(datapath.head().to_string())
    file.write("\n")

###################################################################################################
# Function name =displaydrop
# description = This function drops the specified column from the DataFrame.
# author = sakshi khadke
# date = 4-10-2025
###################################################################################################

def displaydrop(datapath,file):
    file.write(line)
    file.write("clean the dataset")
    datapath.drop(columns=['Unnamed: 0'],inplace=True) #true is necessary he will drop on the spot
    file.write(datapath.to_string())
    file.write("\n")

###################################################################################################
# Function name =displayupdated
# description = This function updates the specified column from the DataFrame.
# author = sakshi khadke
# date = 4-10-2025
###################################################################################################

def displayupdated(datapath,file):
    file.write(line)
    file.write("updated dataset is:")
    file.write(datapath.head().to_string())
    file.write("\n")

###################################################################################################
# Function name =displaynull
# description = This function displays the null values in the DataFrame.
# author = sakshi khadke
# date = 4-10-2025
###################################################################################################

def displaynull(datapath,file):
    file.write(line)
    file.write("missing values in each column:\n" + datapath.isnull().sum().to_string())
    file.write("\n")

###################################################################################################
# Function name =displaydescribe
# description = This function displays the statistical summary of the DataFrame.
# author = sakshi khadke
# date = 4-10-2025
###################################################################################################

def displaydescribe(datapath,file):
    file.write(line)
    file.write("statistical summary:")
    file.write(datapath.describe().to_string())
    file.write("\n")

###################################################################################################
# Function name =displaycorrelation
# description = This function displays the correlation matrix of the DataFrame.
# author = sakshi khadke
# date = 4-10-2025
###################################################################################################

def displaycorrelation(datapath,file):
    file.write(line)
    file.write("correlation matrix")
    file.write(datapath.corr().to_string())
    file.write("\n")

###################################################################################################
# Function name =columns
# description = This function displays the columns of the DataFrame.
# author = sakshi khadke
# date = 4-10-2025
###################################################################################################

def columns(datapath,file):
    file.write(line)
    file.write(datapath.columns.to_series().to_string())
    file.write("\n")    

###################################################################################################
# Function name =datatypes
# description = This function displays the data types of the columns in the DataFrame.
# author = sakshi khadke
# date = 4-10-2025
###################################################################################################

def datatypes(datapath,file):
    file.write("\n")
    file.write(datapath.dtypes.to_string())
    file.write("\n")

###################################################################################################
# Function name =encode
# description = This function encodes the specified column from the DataFrame.
# author = sakshi khadke
# date = 4-10-2025
###################################################################################################

def encode(df):
    x=df.dropna(inplace=True)
    return df

###################################################################################################
# Function name =alter
# description = This function alters the specified column from the DataFrame.
# author = sakshi khadke
# date = 4-10-2025
###################################################################################################
def alter(df):
    x=df.drop(columns=['radio'])
    y=df['radio']
    return x,y

###################################################################################################
# Function name =scaler
# description = This function scales the specified column from the DataFrame.
# author = sakshi khadke
# date = 4-10-2025
###################################################################################################
def scaler(x):

    scaler=StandardScaler()
    x_scale=scaler.fit_transform(x)
    return x_scale

###################################################################################################
# Function name =split_data
# description = This function splits the DataFrame into training and testing sets.
# author = sakshi khadke
# date = 4-10-2025
###################################################################################################
def split_data(x, y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    return x_train, y_train, x_test, y_test

###################################################################################################
# Function name =fit_model
# description = This function fit model the specified column from the DataFrame.
# author = sakshi khadke
# date = 4-10-2025
###################################################################################################
def fit_model(x_train,y_train): 
        model=KNeighborsRegressor(n_neighbors=5)
        model.fit(x_train,y_train)
        return model    
###################################################################################################
# Function name =save_model
# description = This function saves the trained model to a file.
# author = sakshi khadke
# date = 4-10-2025
###################################################################################################
def save_model(model,model_path=model_path):
    joblib.dump(model,model_path)
    print(f"model saved at {model_path}")

###################################################################################################
# Function name =load_model
# description = This function loads a trained model from a file.
# author = sakshi khadke
# date = 4-10-2025
###################################################################################################
def load_model(model_path=model_path):
    model=joblib.load(model_path)
    return model

###################################################################################################
# Function name =find_best_k
# description = This function finds the best k value for KNN.
# author = sakshi khadke
# date = 4-10-2025
###################################################################################################
def find_best_k(x_train, y_train, x_test, y_test):
    scores = []
    k_range = range(1, 21)

    for k in k_range:
        model = KNeighborsRegressor(n_neighbors=k)
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        score = metrics.r2_score(y_test, y_pred)  # or use negative RMSE
        scores.append(score)

    return k_range, scores

###################################################################################################
# Function name =plot(k_range,accuracy_scores)
# description = This function plots the accuracy scores for different k values.
# author = sakshi khadke
# date = 4-10-2025
###################################################################################################
def plot(k_range, accuracy_scores):

    plt.figure(figsize=(8,5))
    plt.plot(k_range, accuracy_scores, marker='o', linestyle='--')
    plt.title("accuracy so k value")
    plt.xlabel("value of k")
    plt.ylabel("accuracy of model")
    plt.grid(True)
    plt.xticks(k_range)
    plt.savefig("artifacts/advertising/k_value_vs_accuracy.png")
    plt.close()    

###################################################################################################
# Function name =correlation
# description = This function computes the correlation matrix for the DataFrame.
# author = sakshi khadke
# date = 4-10-2025
###################################################################################################
def correlation(df,file):
    file.write("co-relation matrice : \n")
    file.write(df.corr().to_string())
    file.write("\n")

    plt.figure(figsize=(10,5))
    sns.heatmap(df.corr(),annot=True,cmap='coolwarm')
    plt.title("marvellous correlation heatmap")
    plt.savefig("artifacts/advertising/correlation_plot.png")
    plt.close()

###################################################################################################
# Function name =pairplot
# description = This function creates a pairplot for the DataFrame.
# author = sakshi khadke
# date = 4-10-2025
###################################################################################################
def pairplot(df,file):   
    file.write("pairplot matrice : \n")
    file.write("\n")

    sns.pairplot(df)
    plt.suptitle("pairplot of features",y=1.02)
    plt.savefig("artifacts/advertising/pairplot_plot.png")
    plt.close()
        
###################################################################################################
# Function name =main
# description = This function form where execution start
# author = sakshi khadke
# date = 4-10-2025
###################################################################################################
def main():
    try:
        file=open_folder_file(file_path=file_path)

        df=read_csv(datapath)

        if sys.argv[1]=="--train":

            displayhead(df,file)

            displaydrop(df,file)

            displayupdated(df,file)

            displaynull(df,file)

            displaydescribe(df,file)

            displaycorrelation(df,file)

            columns(df,file)

            datatypes(df,file)

            df=encode(df)

            x, y = alter(df)

            x_train, y_train, x_test, y_test = split_data(x, y)  # ✅ Define x_train and y_train

            model=fit_model(x_train,y_train)

            correlation(df,file)

            pairplot(df,file)

            

        elif sys.argv[1] == "--predict":

            x, y = alter(df)
            scale_x = scaler(x)

            # Split the scaled data
            x_train, y_train, x_test, y_test = split_data(scale_x, y)

            # Train and save the model
            model = fit_model(x_train, y_train)
            save_model(model, model_path=model_path)
            file.write("model is trained and saved \n")

            # Load and use the model
            model = load_model(model_path=model_path)
            file.write("model is loaded \n")

            y_pred = model.predict(x_test)

            # Evaluate best k
            k_range, accuracy_scores = find_best_k(x_train, y_train, x_test, y_test)

            plot(k_range, accuracy_scores)

            best_k = k_range[accuracy_scores.index(max(accuracy_scores))]
            best_score = max(accuracy_scores)

            file.write("best accuracy is : ")
            file.write(f"{best_k} {best_score}\n")
        else: 
            print("valid arguments are --train and --predict")
            return

    except Exception as e:
        print(e)
        print("valid arguments are --train and --predict")          

if __name__=="__main__":
    main()    

