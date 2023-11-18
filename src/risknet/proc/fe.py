'''Uses featuretools (API) to engineer new features'''

import featuretools as ft
import pickle
import pandas as pd

import logging
logger = logging.getLogger("freelunch")

#Local imports
import encoder

'''
Uses featuretools's DFS method to generate new features for df

input: df = DataFrame pulled from df.pkl that contains all cleaned/formatted features
output: returns DataFrame with multiple generated features
'''
def fe(df):
    #Get dataframe with nulls defined ("nd") but not categorically encoded (yet)

    #Create an EntitySet that stores dataframe
    es = ft.EntitySet(id="data")
    es = es.add_dataframe(
        dataframe_name="data",
        dataframe=df,
        index="loan_sequence_number",
        time_index="first_payment_date",
    )
    
    #This returns feature_matrix (new features generated off of df) and defines features
    feature_matrix, feature_defs = ft.dfs(entityset=es, target_dataframe_name="data")

    #Combine new features + old feature table (df)
    combo = feature_matrix.merge(
            df, on="loan_sequence_number", how="inner")
    
    #Do not want to create new features based on pred
    combo = combo.drop(['flag_x', 'flag_y', 'default_x', 'default_y', 'undefaulted_progress_x', 'undefaulted_progress_y'], axis=1) 
    
    logger.info(combo.columns)
    logger.info('flag' in combo.columns)
    
    return combo