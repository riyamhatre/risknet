Statement of Purpose
Banks are an integral part of society for obvious reasons and credit score is an important metric that can determine the trajectory of one’s life. This score is used to determine whether to offer you loans, mortgage, etc.; sometimes this score is used to choose people for jobs! This team aims to predict credit default with the aid of machine learning and data processing techniques. Our methodology includes performing an ablation study to evaluate the effectiveness, as assessed by AUC, of a machine learning model trained on a dataset with processed features in comparison to two benchmarks: the identical model architecture trained on unprocessed data and a traditional credit score approach. This comparative analysis aims to elucidate the impact of data processing techniques on improving the predictive capabilities of machine learning models in predicting credit defaults. Our objective is to contribute to the comprehension of how advanced data processing may result in more precise risk assessments within the realm of financial lending.
Data and EDA 
Refer to my jupyter nb and talk about freddie mac 

File Information
To ensure reproducibility, the team created a pip-installable package called “risknet”, which contains all the necessary packages to successfully run the code on any computer. 
Loading the Data
For this project, the team used the Freddie Mac Single-Family Loan-Level dataset due to its accessibility and quality. It is the one of the largest and most high-quality credit performance datasets for mortgages. 
Parquet.py
To expedite the model’s run time, the team decided to convert the data, which was in .txt format, to parquet. parquet.py has one method, whose sole purpose is to take in the file path to the .txt file and convert it to a parquet file, which is then stored in a specified location. This method is called in the main py file before the main processes begin. 
Label_prep.py
Before running the model, it is important to clean the data being used. This involves changing the data type in certain columns, dropping irrelevant data, and computing extra calculations. 

Reducer.py
WIP
Encoder.py
This file contains methods that are utilized to encode columns. Initially, the columns are sorted into two categories: numerical and categorical. Based on the column type, the appropriate methods are called. As stated in the exploratory data analysis section, the dataset has many null values. To combat this, the team decided to encode the null values with either a certain number in the numerical columns, or a certain string value in the categorical columns.  

Model.py
This file contains the code for the model and defines the hyperparameters and returns the evaluation metrics. 

Pipeline.py
All the code that we have written comes together in this file. Initially, we specify the file path and the files that we are going to utilize in the model. Subsequently,  we call various functions from .model, .reducer, .encoder, etc.
