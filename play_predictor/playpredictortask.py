###################################################################################################
# Required Libraries
###################################################################################################
import os
import os
import sys
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

###################################################################################################
# Function name = read_csv
# description = This function reads a CSV file and returns a DataFrame.
# author = sakshi khadke
# date = 5-10-2025
###################################################################################################

def read_csv(datapath):
    return pd.read_csv(datapath)


###################################################################################################
# Dataset Headers
###################################################################################################
    df['Whether']=df['Whether'].map({'Sunny':0,'Overcast':1,'Rainy':2})
    df['Temperature']=df['Temperature'].map({'Hot':0,'Mild':1,'Cool':2})
    df['Play']=df['Play'].map({'Yes':0,'No':1})

###################################################################################################
# File path
###################################################################################################

line = "-" * 42
datapath = "PlayPredictor.csv"
model_path =os.path.join("artifacts/play_predictor", "playpredictortask.joblib")
folder_path="artifacts/play_predictor"
file_path=folder_path+ "/" "play_predictor_report.txt"

###################################################################################################
# Function name = open_folder_file
# description = This function creates a folder and opens a file in write mode.
# author = sakshi khadke
# date = 5-10-2025
###################################################################################################

def open_folder_file(file_path=file_path):
    os.makedirs(folder_path,exist_ok=True)
    file=open(file_path,"w")
    return file

###################################################################################################
# Function name = encode
# description = This function encodes categorical variables in the DataFrame.
# author = sakshi khadke
# date = 5-10-2025
###################################################################################################

def encode(df):
    df.dropna(inplace=True)
    if 'Whether' in df.columns:
        df['Whether'] = df['Whether'].map({'Sunny': 0, 'Overcast': 1, 'Rainy': 2})
    if 'Temperature' in df.columns:
        df['Temperature'] = df['Temperature'].map({'Hot': 0, 'Mild': 1, 'Cool': 2})
    if 'Play' in df.columns:
        df['Play'] = df['Play'].map({'Yes': 0, 'No': 1})
    return df

###################################################################################################
# Function name = displayhead
# description = This function displays the head of the DataFrame.
# author = sakshi khadke
# date = 5-10-2025
###################################################################################################

def displayhead(df,file,label="dataset sample is:"):
   file.write(f"{label}\n")
   file.write(f"{df.head().to_string()}\n")

###################################################################################################
# Function name = describe
# description = This function describes the DataFrame.
# author = sakshi khadke
# date = 5-10-2025
###################################################################################################

def describe(datapath,file):
    file.write(line)
    file.write(datapath.describe().to_string())
    file.write("\n")

###################################################################################################
# Function name = displayshape
# description = This function displays the shape of the DataFrame.
# author = sakshi khadke
# date = 5-10-2025
###################################################################################################

def displayshape(datapath,file):
    file.write(datapath.shape().to_string())

###################################################################################################
# Function name = columns
# description = This function displays the columns of the DataFrame.
# author = sakshi khadke
# date = 5-10-2025
###################################################################################################

def columns(datapath,file):
    file.write(line)
    file.write(datapath.columns.to_series().to_string())
    file.write("\n")

###################################################################################################
# Function name = datatypes
# description = This function displays the data types of the columns in the DataFrame.
# author = sakshi khadke
# date = 5-10-2025
###################################################################################################

def datatypes(datapath,file):
    file.write(line)
    file.write(datapath.dtypes.to_string())
    file.write("\n") 

###################################################################################################
# Function name = displaydrops
# description = This function displays the dropped columns of the DataFrame.
# author = sakshi khadke
# date = 5-10-2025
###################################################################################################

def displaydrop(df):
    x=df.drop(columns='Play')
    y=df['Play']
    return x,y

###################################################################################################
# Function name = scaler
# description = This function scales the features of the DataFrame.
# author = sakshi khadke
# date = 5-10-2025
###################################################################################################

def scaler(x):

    scaler=StandardScaler()
    x_scale=scaler.fit_transform(x)

    return x_scale

###################################################################################################
# Function name = split_data
# description = This function splits the data into training and testing sets.
# author = sakshi khadke
# date = 5-10-2025
###################################################################################################

def split_data(x,y):
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
    return x_train,x_test,y_train,y_test

###################################################################################################
# Function name = fit_model
# description = This function fits a KNN model to the training data.
# author = sakshi khadke
# date = 5-10-2025
###################################################################################################

def fit_model(x_train,y_train):
    model=KNeighborsClassifier(n_neighbors=5)
    model.fit(x_train,y_train)
    return model

###################################################################################################
# Function name = save_model
# description = This function saves the trained model to a file.
# author = sakshi khadke
# date = 5-10-2025
###################################################################################################

def save_model(model,model_path=model_path):
    joblib.dump(model,model_path)
    print(f"model saved at {model_path}")

###################################################################################################
# Function name = load_model
# description = This function loads a trained model from a file.
# author = sakshi khadke
# date = 5-10-2025
###################################################################################################

def load_model(model_path=model_path):
    model=joblib.load(model_path)
    return model    

###################################################################################################
# Function name = find_best_k
# description = This function finds the best k value for the KNN model.
# author = sakshi khadke
# date = 5-10-2025
###################################################################################################

def find_best_k(y_test,y_pred):
    accuracy_scores=[]
    k_range=range(1,21)

    for k in k_range:

        accuracy=accuracy_score(y_test,y_pred)
        accuracy_scores.append(accuracy)

    return k_range, accuracy_scores

###################################################################################################
# Function name = plot(k_range, accuracy_scores)
# description = This function plots the accuracy of the KNN model for different k values.
# author = sakshi khadke
# date = 5-10-2025
###################################################################################################

def plot(k_range, accuracy_scores):

    plt.figure(figsize=(8,5))
    plt.plot(k_range, accuracy_scores, marker='o', linestyle='--')
    plt.title("accuracy of k value")
    plt.xlabel("value of k")
    plt.ylabel("accuracy of model")
    plt.grid(True)
    plt.xticks(k_range)
    plt.savefig("Artifacts/play_predictor/K_value_vs_Accuracy.png")
    plt.close() 

###################################################################################################
# Function name = displayCorrelation
# description = This function displays the correlation matrix of the DataFrame.
# author = sakshi khadke
# date = 5-10-2025
###################################################################################################

def DisplayCorrelation(df, file) :
        file.write("co-relation matrice : \n")
        file.write(df.corr().to_string())
        file.write("\n")

        plt.figure(figsize=(10,6))
        sns.heatmap(df.corr(), annot = True, cmap = 'coolwarm')
        plt.title("play Predictor")
        plt.savefig("Artifacts/play_Predictor/Correlation_plot.png")
        plt.close()

###################################################################################################
# Function name = displaypairplot
# description = This function displays the pairplot of the DataFrame.
# author = sakshi khadke
# date = 5-10-2025
###################################################################################################

def DisplayPairplot(df) :

    sns.pairplot(df)
    plt.suptitle("pairplot of feature", y  = 1.02)
    plt.savefig("Artifacts/play_Predictor/Pairplot_plot.png")
    plt.close()    

###################################################################################################
# Function name = main
# description = This function preprocesses the dataset and train the model and calls the given functions.
# author = sakshi khadke
# date = 5-10-2025
###################################################################################################

def main():
    try :
        file=open_folder_file(file_path=file_path)

        df=read_csv(datapath)
        df=encode(df)

        if sys.argv[1]=="--train":
       
            displayhead(df,file)

            describe(df,file)

            columns(df,file)

            DisplayCorrelation(df, file)

            DisplayPairplot(df)
            file.write("correlation and pairplot is saved \n")

            x,y= displaydrop(df)
            
            scale_x=scaler(x)

            x_train,x_test,y_train,y_test=split_data(scale_x,y)

            model=fit_model(x_train,y_train)

            save_model(model,model_path=model_path)
            file.write("model is trained and saved \n")



        elif sys.argv[1]=="--predict":    

            x,y= displaydrop(df)

            scale_x=scaler(x)

            model=load_model(model_path=model_path)

            x_train,x_test,y_train,y_test=split_data(scale_x,y)

            file.write("model is loaded \n")
            y_pred=model.predict(x_test)

            k_range, accuracy_scores = find_best_k(y_test, y_pred)

            plot(k_range, accuracy_scores)
            plt.plot(k_range, accuracy_scores)
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
if __name__=="__main__":
    main()   