###################################################################################################
# Required Libraries
###################################################################################################
import os
import sys
import pandas as pd
import seaborn as sns
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score,confusion_matrix

###################################################################################################
# File path
###################################################################################################

line = "-" * 42
datapath = "WinePredictor.csv"
model_path =os.path.join("artifacts/wine_predictor", "winepredictorparametertunningtask.joblib")
folder_path="artifacts/wine_predictor"
file_path=folder_path+ "/" "wine_predictor_report.txt"
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
headers = ['Class,Alcohol','Malic acid','Ash','Alcalinity of ash','Magnesium','Total phenols','Flavanoids','Nonflavanoid phenols','Proanthocyanins','Color intensity,Hue','OD280/OD315 of diluted wines','Proline']

###################################################################################################
# Function name = read_csv
# description = This function reads a CSV file and returns a DataFrame.
# author = sakshi khadke
# date = 3-10-2025
###################################################################################################

def read_csv(datapath):
    return pd.read_csv(datapath)
###################################################################################################
# Function name = displayhead
# description = This function displays the head of the DataFrame.
# author = sakshi khadke
# date = 3-10-2025
###################################################################################################

def displayhead(df,file,label="dataset sample is :"):
    file.write(f"{label}\n")
    file.write(f"{df.head().to_string()}\n")

###################################################################################################
# Function name = describe
# description = This function returns the descriptive statistics of the DataFrame.
# author = sakshi khadke
# date = 3-10-2025
###################################################################################################

def describe(datapath,file):
    file.write(line)
    file.write(datapath.describe().to_string())
    file.write("\n")
###################################################################################################
# Function name = columns
# description = This function returns the columns of the DataFrame.
# author = sakshi khadke
# date = 3-10-2025
###################################################################################################

def columns(datapath,file):
    file.write(line)
    file.write(datapath.columns.to_series().to_string())
    file.write("\n")

###################################################################################################
# Function name =datatypes
# description = This function returns the data types of the columns in the DataFrame.
# author = sakshi khadke
# date = 3-10-2025
###################################################################################################

def datatypes(datapath,file):
    file.write(line)
    file.write(datapath.dtypes.to_string())
    file.write("\n")    
###################################################################################################
# Function name = encode
# description = This function encode a CSV file and returns a encoded DataFrame.
# author = sakshi khadke
# date = 3-10-2025
###################################################################################################

def encode(df) :
    df.dropna(inplace=True)
    return df

###################################################################################################
# Function name = alter
# description = This function alters the DataFrame for model training.
# author = sakshi khadke
# date =3-10-2025
###################################################################################################

def alter(df) :
    x=df.drop(columns=['Class'])
    y=df['Class']
    return x, y

###################################################################################################
# Function name = scaler
# description = This function scale the DataFrame for model training.
# author = sakshi khadke
# date = 3-10-2025
###################################################################################################

def scaler(x) :

    scaler=StandardScaler()
    x_scale=scaler.fit_transform(x)

    return x_scale

###################################################################################################
# Function name = split_data
# description = This function splits the DataFrame for model training.
# author = sakshi khadke
# date = 3-10-2025
###################################################################################################
def split_data(x,y) :
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
    return x_train,x_test,y_train,y_test

###################################################################################################
# Function name = fit_model
# description = This function fits the model fo training.
# author = sakshi khadke
# date =3-10-2025
###################################################################################################

def fit_model(x_train,y_train):
    model=KNeighborsClassifier(n_neighbors=50)
    model.fit(x_train,y_train)
    return model

###################################################################################################
# Function name =save model
# description = This function save the model after training.
# author = sakshi khadke
# date = 3-10-2025
###################################################################################################
def save_model(model,model_path=model_path):
    joblib.dump(model,model_path)
    print(f"model saved at {model_path}")

###################################################################################################
# Function name =load_model
# description = This function load the model for prediction.
# author = sakshi khadke
# date =3-10-2025
###################################################################################################
def load_model(model_path=model_path):
    model=joblib.load(model_path)
    return model    
###################################################################################################
# Function name = find best k
# description = This function find best k for the DataFrame for model training.
# author = sakshi khadke
# date = 3-10-2025
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
# date = 3-10-2025
###################################################################################################
def plot(k_range,accuracy_scores):

    plt.figure(figsize=(8,5))
    plt.plot(k_range,accuracy_scores,marker='o',linestyle='--')
    plt.title("accuracy of k value")
    plt.xlabel("value of k")
    plt.ylabel("accuracy of model")
    plt.grid(True)
    plt.xticks(k_range)
    plt.savefig("Artifacts/Wine_Predictor/K_value_vs_Accuracy.png")
    plt.close()
###################################################################################################
# Function name = displaycorrelation
# description = this function display correlation 
# description = This function preprocesses the dataset and trains the model and calls the given functions.
# author = sakshi khadke
# date = 3-10-2025
###################################################################################################
def DisplayCorrelation(df, file) :
        file.write("co-relation matrice : \n")
        file.write(df.corr().to_string())
        file.write("\n")

        plt.figure(figsize=(10,6))
        sns.heatmap(df.corr(), annot = True, cmap = 'coolwarm')
        plt.title("Wine Predictor")
        plt.savefig("Artifacts/Wine_Predictor/Correlation_plot.png")
        plt.close()
###################################################################################################
# Function name = displaypairplot
# description = this function display pairplot
# description = This function preprocesses the dataset and trains the model and calls the given functions.
# author = sakshi khadke
# date = 3-10-2025
#######################################################################################
def DisplayPairplot(df) :

    sns.pairplot(df)
    plt.suptitle("pairplot of feature", y  = 1.02)
    plt.savefig("Artifacts/Wine_Predictor/Pairplot_plot.png")
    plt.close()    
###################################################################################################
# Function name = main
# description = this function from where execution start
# description = This function preprocesses the dataset and trains the model and calls the given functions.
# author = sakshi khadke
# date = 3-10-2025
###################################################################################################

def main():
    try :
        file=open_folder_file(file_path=file_path)

        df=read_csv(datapath)

        if sys.argv[1]=="--train":
       
            displayhead(df,file)

            describe(df,file)

            columns(df,file)

            df=encode(df)

            DisplayCorrelation(df, file)

            DisplayPairplot(df)
            file.write("correlation and pairplot is saved \n")

            x,y= alter(df)
            
            scale_x=scaler(x)

            x_train,x_test,y_train,y_test=split_data(scale_x,y)

            model=fit_model(x_train,y_train)

            save_model(model,model_path=model_path)
            file.write("model is trained and saved \n")



        elif sys.argv[1]=="--predict":    

            x,y= alter(df)

            model=load_model(model_path=model_path)

            scale_x=scaler(x)

            x_train,x_test,y_train,y_test=split_data(scale_x,y)

            file.write("model is loaded \n")
            y_pred=model.predict(x_test)

            k_range,accuracy_scores=find_best_k(y_test,y_pred)

            plot(k_range,accuracy_scores)
            plt.plot(k_range,accuracy_scores)
            file.write("best accuracy is : ")
            file.write(f"{k_range[accuracy_scores.index(max(accuracy_scores))]} {max(accuracy_scores)}")
            file.write("\n")

        
        else: 
            print("valid arguments are --train and --predict")
            return

    except Exception as e:
        print(e)
        print("valid arguments are --train and --predict")          


if __name__=="__main__":

    main()    
