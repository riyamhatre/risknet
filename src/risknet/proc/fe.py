'''Uses featuretools (API) to engineer new features'''

import featuretools as ft

'''
Uses featuretools's DFS method to generate new features for df

input: df = DataFrame pulled from df.pkl that contains all cleaned/formatted features
output: returns DataFrame with multiple generated features
'''
def fe(df):
    #Create an EntitySet that stores dataframe
    es = ft.EntitySet(id="cleaned_df")
    es = es.add_dataframe(
        dataframe_name="cleaned_df",
        dataframe=df,
        index="loan_sequence_number",
        time_index="first_payment_date",
    )