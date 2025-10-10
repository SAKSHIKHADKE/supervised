###################################################################################################
# Required Libraries
###################################################################################################
import pandas as pd
import seaborn as sns
import joblib
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import (
    StandardScaler, 
    LabelEncoder)
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    confusion_matrix, 
    ConfusionMatrixDisplay, 
    classification_report,
    roc_curve)

###################################################################################################
# File path
###################################################################################################
line = "-" * 42
datapath = "breast-cancer-wisconsin.csv"
model_path =os.path.join("artifacts/breast_cancer", "BreastCancerRandomforesttask.joblib")
folder_path="artifacts/breast_cancer"
file_path=folder_path+ "/" "breast_cancer_report.txt"

###################################################################################################
# Function name = open_folder_file
# description = This function open a folder and a file for writing.
# author = sakshi khadke
# date = 6-10-2025
###################################################################################################
def open_folder_file(file_path=file_path):
    os.makedirs(folder_path,exist_ok=True)
    file=open(file_path,"w")
    return file
###################################################################################################
# Dataset Headers
###################################################################################################
headers = ['CodeNumber','ClumpThickness','UniformityCellSize','UniformityCellShape','MarginalAdhesion','SingleEpithelialCellSize','BareNuclei','BlandChromatin','NormalNucleoli','Mitoses','CancerType']

###################################################################################################
# Function name = read_csv
# description = This function reads a CSV file and returns a DataFrame.
# author = sakshi khadke
# date = 6-10-2025
###################################################################################################

def read_csv(datapath) :
    return pd.read_csv(datapath)

###################################################################################################
# Function name = displayhead
# description = This function displays the head of the DataFrame.
# author = sakshi khadke
# date = 6-10-2025
###################################################################################################

def displayhead(df,file,label="dataset sample is :"):
    file.write(f"{label}\n")
    file.write(f"{df.head().to_string()}\n")

###################################################################################################
# Function name = describe
# description = This function returns the descriptive statistics of the DataFrame.
# author = sakshi khadke
# date = 6-10-2025
###################################################################################################

def describe(datapath,file):
    file.write(line)
    file.write(datapath.describe().to_string())
    file.write("\n")
###################################################################################################
# Function name = columns
# description = This function prints the columns of the dataset.
# author =sakshi khadke
# date =6-10-2025
###################################################################################################
    
def columns(datapath,file) :
    file.write(line)
    file.write(datapath.columns.to_series().to_string())
    
###################################################################################################
# Function name = datatypes
# description = This function prints the datatypes of the dataset.
# author = sakshi khadke
# date =6-10-2025
###################################################################################################
    
def datatypes(datapath,file) :
    file.write(line)
    file.write(datapath.dtypes.to_string())
    file.write("\n")

###################################################################################################
# Function name = encoding
# description = This function encodes categorical features in the dataset and returns the modified DataFrame.
# author = sakshi khadke
# date =6-10-2025
###################################################################################################
    
def encoding(datapath) :

    for col in datapath.select_dtypes(include='object') :
        datapath[col] = LabelEncoder().fit_transform(datapath[col])

    return datapath


def Fill_empty(df):
    df.replace('?', np.nan, inplace=True)
    df = df.fillna(df.mean(numeric_only=True))
    return df


###################################################################################################
# Function name = heatmap
# description = This function show heat map of dataset
# author = sakshi khadke
# date = 6-10-2025
###################################################################################################
    
def heatmap(datapath) :
    sns.heatmap(datapath.corr(), annot = True, cmap = "Purples")
    plt.title("heat map for breast cancer")
    plt.savefig("Artifacts/breast_cancer/heatmap.png")
    plt.close()
    
###################################################################################################
# Function name = alter
# description = This function alters the dataset by dropping unnecessary columns.
# author = sakshi khadke
# date = 6-10-2025
###################################################################################################
    
def alter(datapath) :
    a = datapath.drop(columns=['CodeNumber','CancerType'])
    b = datapath['CancerType']
    return a, b

###################################################################################################
# Function name = scaler
# description = This function scales the features of the dataset.
# author = sakshi khadke
# date = 6-10-2025
###################################################################################################

def scaler(x):
    data = StandardScaler()
    return data.fit_transform(x)

###################################################################################################
# Function name = trainModel
# description = This function splits the dataset into training and testing sets.
# author = sakshi khadke
# date = 6-10-2025
###################################################################################################

def split_data(x, y) :
    a, b, c, d = train_test_split(x, y, test_size=0.2, random_state=42)
    return a, b, c, d

###################################################################################################
# Function name = fit
# description = This function fits a logistic regression model to the training data.
# author = sakshi khadke
# date = 6-10-2025
###################################################################################################

def fit(x_train, y_train) :
    model = RandomForestClassifier(n_estimators=300)
    model.fit(x_train, y_train)
    return model

###################################################################################################
# Function name = save_model
# description = This function saves the trained model to a file.
# author = sakshi khadke
# date = 6-10-2025
###################################################################################################

def save_model(model_fit, path = model_path):
    joblib.dump(model_fit, path)
    print(f"Model saved to {path}\n")
    return path
###################################################################################################
# Function name = load_model
# description = This function loads a trained model from a file.
# author = sakshi khadke
# date = 6-10-2025
###################################################################################################

def load_model(path = model_path):
    model = joblib.load(path)
    print(f"Model loaded from {path}")
    return model

###################################################################################################
# Function name = accuracy
# description = This function calculates the accuracy of the model.
# author = sakshi khadke
# date = 6-10-2025
###################################################################################################

def accuracy(prediction, y_test) :
    return accuracy_score(y_test, prediction)

###################################################################################################
# Function name = classification
# description = This function return the classification report.
# author = sakshi khadke
# date = 6-10-2025
###################################################################################################

def classification(prediction, y_test) :
    return classification_report(y_test, prediction)

###################################################################################################
# Function name = confusion
# description = This function returns the confusion matrix.
# author = sakshi khadke
# date = 6-10-2025
###################################################################################################

def confusion(prediction, y_test) :
    return confusion_matrix(y_test, prediction) 

###################################################################################################
# Function name = matrixDisplay
# description = This function displays the confusion matrix.
# author = sakshi khadke
# date = 6-10-2025
###################################################################################################

def matrixDisplay(matrix, y_test):
    cmdk = ConfusionMatrixDisplay(matrix, display_labels=np.unique(y_test))
    cmdk.plot(cmap='magma')
    plt.title("confusion matrix")
    plt.savefig("Artifacts/breast_cancer/Confusion_Matrix.png")
    plt.close()

###################################################################################################
# Function name = pair_plot
# description = This function displays the pair plot.
# author = sakshi khadke
# date = 6-10-2025
###################################################################################################
    
def pair_plot(datapath) :
    df = read_csv(datapath)
    sns.pairplot(df)
    plt.title("Pair Plot")
    plt.savefig("Artifacts/breast_cancer/Pairplot_plot.png")
    plt.close()

###################################################################################################
# Function name = feature_importance
# description = This function displays the feature importance.
# author =  sakshi khadke
# date = 6-10-2025
###################################################################################################

def feature_importance(model, feature_names,file) :
    feature_names = list(feature_names)
    importance = np.array(model.feature_importances_).flatten()
    
    file.write("Number of feature names:", len(feature_names))
    file.write("Number of importances:", len(model.feature_importances_))

    if len(feature_names) != len(importance):
        raise ValueError("Length of feature_names and feature_importances_ must match.")

    feature_importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values(by='importance', ascending=True)
    
    plt.figure(figsize=(10, 6))
    plt.barh(feature_importance_df['feature'], feature_importance_df['importance'], color='skyblue')
    plt.title("Feature Importance")
    plt.xlabel("Importance")
    plt.ylabel("Features")
    plt.tight_layout()
    plt.savefig("Artifacts/breast_cancer/Feature_Importance.png")
    plt.close()

###################################################################################################
# Function name = auc_score
# description = This function calculates the AUC score.
# author =  sakshi khadke
# date = 6-10-2025
###################################################################################################

def auc_score(y_test, prediction):
    auc = roc_auc_score(y_test, prediction)
    return auc

###################################################################################################
# Function name = roc_graph
# description = This function displays the ROC curve.
# author =  sakshi khadke
# date = 6-10-2025
###################################################################################################

def roc_graph(y_test, prediction):
    lr, tlr, _ = roc_curve(y_test, prediction, pos_label=4)

    plt.plot(lr, tlr, color='blue')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve of logistic regression')
    plt.grid()
    plt.savefig("Artifacts/breast_cancer/ROC_Curve.png")
    plt.close()

###################################################################################################
# Function name = main
# description = this function from where execution start
# description = This function preprocesses the dataset and trains the model and calls the given functions.
# author =  sakshi khadke
# date = 6-10-2025
###################################################################################################

def main() :
    try:
        file=open_folder_file(file_path=file_path)

        # 1)read dataset
        df = read_csv(datapath)

        df = Fill_empty(df) # add this function

        df = encoding(df)

        x, y = alter(df)

        # 9)Scale the features
        x_scale = scaler(x)

        if sys.argv[1]=="--train":
            # 3)Describe the dataset information
            describe(df,file)
            
            # 4)Display information of column
            columns(df,file)
            
            # 5)display datatypes of columns
            datatypes(df,file)

            # 7)Display heatmap of the dataset
            #heatmap(df)
            
            # 10)Train the model and print its size
            x_train, x_test, y_train, y_test = split_data(x_scale, y)
            
            file.write('size of x_train :')
            file.write(f"{x_train.shape}")
            file.write('size of x_test :')
            file.write(f"{x_test.shape}")
            file.write('size of y_train :')
            file.write(f"{y_train.shape}")
            file.write('size of y_test :')
            file.write(f"{y_test.shape}")

            # 11)Fit the model in algorithm
            model_fit = fit(x_train, y_train)

            # 12)save the model
            save_model(model_fit)
            
            # 13) load the model
            model = load_model()

            # 14) make predictions
            prediction = model.predict(x_test)

            # 15) calculate training and testing accuracy and print that accuracy
            
            accuracy_Training = accuracy(y_train, model.predict(x_train))
            file.write(f"training accuracy is : {accuracy_Training}")

            accuracy_testing = accuracy(prediction, y_test)
            file.write(f"Testing accuracy is : {accuracy_testing}")

            # 16) generate classification report and print report
            classif = classification(prediction, y_test)
            file.write(f"classification report is :\n{classif}")

            # 17) generate confusion matrix and display that matrix
            con_mat =confusion(prediction, y_test)
            file.write(f"confusion matrix is :\n{con_mat}")

            #18) display confusion matrix in visual format
            matrixDisplay(confusion_matrix(y_test, prediction), y_test)
            
            # 19) display pair plot
            pair_plot(datapath)

            # 20) display feature importance
            #feature_importance(model, feature_names,file)

            # 21) display auc score 
            AUC_score =auc_score(y_test, prediction)
            file.write(f"AUC Score: {AUC_score:.4f}")

            # 22) display ROC curve
            roc_graph(y_test, prediction)

        elif sys.argv[1]=="--predict":

             # Load the model
            model=load_model(path = model_path)

             # 10)Train the model and print its size
            x_train, x_test, y_train, y_test = split_data(x_scale, y)
            
            file.write('size of x_train :')
            file.write(f"{x_train.shape}")
            file.write('size of x_test :')
            file.write(f"{x_test.shape}")
            file.write('size of y_train :')
            file.write(f"{y_train.shape}")
            file.write('size of y_test :')
            file.write(f"{y_test.shape}")

            # Make predictions
            prediction = model.predict(x_test) # change x_scale to x_test

            # Display the predictions
            file.write(f"Predictions are: {prediction}")

            accuracy_Training = accuracy(y_train, model.predict(x_train))
            file.write(f"training accuracy is : {accuracy_Training}")
            
            accuracy_testing = accuracy(prediction, y_test)
            file.write(f"Testing accuracy is : {accuracy_testing}")

            #  generate classification report and print report
            classif = classification(df)
            file.write(f"classification report is :\n{classif}")

            # generate confusion matrix and display that matrix
            con_mat = confusion(prediction, y_test)
            file.write(f"confusion matrix is :\n{con_mat}")

            # display auc score 
            AUC_score = auc_score(y_test, prediction)
            file.write(f"AUC Score: {AUC_score:.4f}")

        else: 
            print("valid arguments are --train and --predict")
            return
    
    except FileNotFoundError as e:
            print("Exception occured:",e)

    ###################################################################################################
    # application starter
    ###################################################################################################

if __name__ == "__main__" :
    main()