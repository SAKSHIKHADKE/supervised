import pandas as pd
import numpy as np
import os
import sys
import seaborn as sns
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,r2_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

###################################################################################################
# File path
###################################################################################################

line = "-" * 42
datapath = "MarvellousHeadBrain.csv"
model_path =os.path.join("artifacts/head_brain", "head-braintask.joblib")
folder_path="artifacts/head_brain"
file_path=folder_path+ "/" "head_brain_report.txt"
###################################################################################################
# Function name =open_folder_file
# description = This function opens a folder and a file for writing.
# author = sakshi khadke
# date = 19-10-2025
###################################################################################################

def open_folder_file(file_path=file_path):
    os.makedirs(folder_path,exist_ok=True)
    file=open(file_path,"w")
    return file
###################################################################################################
# Dataset Headers
###################################################################################################
headers=['Gender','Age','Range','Head Size(cm^3)','Brain Weight(grams)']

###################################################################################################
# Function name = read_csv
# description = This function reads a CSV file and returns a DataFrame.
# author = sakshi khadke
# date = 19-10-2025
###################################################################################################
def read_csv(datapath):
    return pd.read_csv(datapath)

####################################################################################################
# Function name = displayhead
# description = This function displays the head of the DataFrame.
# author = sakshi khadke
# date = 19-10-2025
###################################################################################################

def displayhead(df,file):
    file.write(df.head().to_string())

###################################################################################################
# Function name = describe
# description = This function describes the DataFrame.
# author = sakshi khadke
# date = 19-10-2025
###################################################################################################
def describe(datapath,file):
    file.write(line)
    file.write(datapath.describe().to_string())
    file.write("\n")

###################################################################################################
# Function name = columns
# description = This function returns the columns of the DataFrame.
# author = sakshi khadke
# date = 19-10-2025
###################################################################################################
def columns(datapath,file):
    file.write(line)
    file.write(datapath.columns.to_series().to_string())
    file.write("\n")

###################################################################################################
# Function name =datatypes
# description = This function returns the data types of the columns in the DataFrame.
# author = sakshi khadke
# date = 19-10-2025
###################################################################################################
def datatypes(datapath,file):
    file.write(line)
    file.write(datapath.dtypes.to_string())
    file.write("\n")   

###################################################################################################
# Function name =displayshape
# description = This function display the shapesof the columns in the DataFrame.
# author = sakshi khadke
# date = 19-10-2025
###################################################################################################
def displayshape(datapath,file):
    file.write(str(datapath.shape))
    file.write("\n")

###################################################################################################
# Function name = encode
# description = This function encode a CSV file and returns a encoded DataFrame.
# author = sakshi khadke
# date = 19-10-2025
###################################################################################################
def encode(df) :
    df.dropna(inplace=True)
    return df

###################################################################################################
# Function name = alter
# description = This function alters the DataFrame for model training.
# author = sakshi khadke
# date =19-10-2025
###################################################################################################
def alter(df):  
    x=df[['Head Size(cm^3)']]
    y=df[['Brain Weight(grams)']]
    return x,y
###################################################################################################
# Function name = scaler
# description = This function scale the DataFrame for model training.
# author = sakshi khadke
# date = 19-10-2025
###################################################################################################

def scaler(x) :

    scaler=StandardScaler()
    x_scale=scaler.fit_transform(x)

    return x_scale

###################################################################################################
# Function name = split_data
# description = This function splits the DataFrame for model training.
# author = sakshi khadke
# date = 19-10-2025
###################################################################################################
def split_data(x,y):
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
    return x_train,x_test,y_train,y_test

###################################################################################################
# Function name = fit_model
# description = This function fits the model fo training.
# author = sakshi khadke
# date =19-10-2025
###################################################################################################
def fit_model(x_train,y_train):
    model=LinearRegression()
    model.fit(x_train,y_train)
    return model

###################################################################################################
# Function name =save_model
# description = This function save the model after training.
# author = sakshi khadke
# date =19-10-2025
###################################################################################################
def save_model(model,model_path=model_path):
    joblib.dump(model,model_path)
    print(f"model saved at {model_path}")

###################################################################################################
# Function name =load_model
# description = This function load the model for prediction.
# author = sakshi khadke
# date =19-10-2025
###################################################################################################
def load_model(model_path=model_path):
    model=joblib.load(model_path)
    return model    

###################################################################################################
# Function name =displaymeanerror
# description = This function displays the meanerror of the model
# author = sakshi khadke
# date =19-10-2025
###################################################################################################
def displaymeanerror(y_test,y_pred):   
    mse=mean_squared_error(y_test,y_pred)
    rmse=np.sqrt(mse)
    r2=r2_score(y_test,y_pred)
    return mse,rmse,r2

###################################################################################################
# Function name = displaygraph
# description = This function displays the graph of the model.
# author = sakshi khadke
# date = 19-10-2025
###################################################################################################
def displaygraph(x_test,y_test,y_pred):
    plt.figure(figsize=(8,5))
    plt.scatter(x_test,y_test,color='blue',label='actual')
    plt.plot(x_test.values.flatten(),y_pred,color='red',linewidth=2,label="regressionline")
    plt.xlabel('Head Size(cm^3)')
    plt.ylabel('brain weight(grams)')
    plt.title("marvellous head brain regression")
    plt.legend()
    plt.grid(True)
    plt.savefig("Artifacts/head_brain/head_brain.png")
    plt.close()

###################################################################################################
# Function name = displaymetrics
# description = This function displays the mse,rmse,r2 of the model
# author = sakshi khadke
# date = 19-10-2025
###################################################################################################
def  displaymetrics(file,model,mse,rmse,r2):
    file.write("result of case study")
    file.write("slope of line(m): " + str(model.coef_[0]) + "\n")
    file.write(f"intercept(c): {model.intercept_}\n")
    file.write("Mean Squared Error: " + str(mse) + "\n")
    file.write(f"R2 score: {r2}\n")
    file.write(f"RMSE: {rmse}\n")
###################################################################################################
# Function name = displaypairplot
# description = this function display pairplot
# description = This function preprocesses the dataset and trains the model and calls the given functions.
# author = sakshi khadke
# date =19-10-2025
#######################################################################################
def DisplayPairplot(df) :

    sns.pairplot(df)
    plt.suptitle("pairplot of feature", y  = 1.02)
    plt.savefig("Artifacts/head_brain/pairplot.png")
    plt.close()    

###################################################################################################
# Function name = displaycorrelation
# description = this function display correlation 
# description = This function preprocesses the dataset and trains the model and calls the given functions.
# author = sakshi khadke
# date =19-10-2025
###################################################################################################
def DisplayCorrelation(df, file) :
        file.write("co-relation matrice : \n")
        file.write(df.corr().to_string())
        file.write("\n")

        plt.figure(figsize=(10,6))
        sns.heatmap(df.corr(), annot = True, cmap = 'coolwarm')
        plt.title("Head_Brain Predictor")
        plt.savefig("Artifacts/head_brain/correlation_plot.png")
        plt.close()
###################################################################################################
# Function name = main
# description = this function from where execution start
# description = This function preprocesses the dataset and trains the model and calls the given functions.
# author = sakshi khadke
# date =19-10-2025
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
            datatypes(df,file)
            displayshape(df,file)
            x,y=alter(df)
            scale_x=scaler(x)
            x_train,x_test,y_train,y_test=split_data(x,y)
            model=fit_model(x_train,y_train)
            y_pred = model.predict(x_test)

            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, y_pred)
            save_model(model, model_path=model_path)

            displaymeanerror(y_test, y_pred)
            displaymetrics(file, model, mse, rmse, r2)
            displaygraph(x_test,y_test,y_pred)
            DisplayPairplot(df)
            DisplayCorrelation(df, file)
            
        elif sys.argv[1]=="--predict":
            x,y=alter(df)
            scale_x=scaler(x)
            x_train,x_test,y_train,y_test=split_data(x,y)
            model=fit_model(x_train,y_train)
            y_pred = model.predict(x_test)

            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, y_pred)
            save_model(model, model_path=model_path)

            displaymeanerror(y_test, y_pred)
            displaymetrics(file, model, mse, rmse, r2)


        else: 
            print("valid arguments are --train and --predict")
            return

    except FileNotFoundError as e:
        print(e)
        print("valid arguments are --train and --predict") 

if __name__=="__main__":
    main()    