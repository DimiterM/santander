import time
import pandas as pd
import numpy as np

from dataset import *



def load_max_trainset(max_month=0, attr_cols=["t", "sex", "age", "seniority", "is_primary", "is_domestic", "income"], remove_non_buyers=False):
    df = pd.read_csv(trainset_filename)
    
    if max_month > 0 and max_month < MAX_SEQUENCE_LENGTH:
        print("max_month: " + str(max_month))
        df = df.loc[df["t"] <= max_month]
    
    df.loc[df["seniority"] < 0, "seniority"] = 0
    z_score_stats = get_z_score_stats(df)
    df = normalize_cols(df, z_score_stats)
    
    if remove_non_buyers:
        df = df_remove_non_buyers(df)
    
    # find last month for each id
    df_maxs = df[["id", "t"]].groupby("id").max()
    df_maxs["id"] = df_maxs.index
    df_maxs.columns = ["t_max", "id"]
    df = pd.merge(df_maxs, df, how="outer", on=["id"])
    ys = df.columns.tolist()[-NUM_CLASSES:]
    
    # take last ys for each id for y array and last attrs for X array
    df_last = df.loc[df["t"] == df["t_max"]]
    dfy = df_last[ys]
    dfa = df_last.iloc[:, :-NUM_CLASSES]
    
    # take previous ys for all purchases array
    df = df.loc[df["t"] < df["t_max"]]
    # take all purchases for X array
    df_maxed = df[["id"]+ys].groupby("id", as_index=False).max()
    
    # take penultimate ys for each id for X array
    df_maxs2 = df[["id", "t"]].groupby("id").max()
    df_maxs2["id"] = df_maxs2.index
    df_maxs2.columns = ["t_max2", "id"]
    df = pd.merge(df_maxs2, df, how="outer", on=["id"])
    df = df.loc[df["t"] == df["t_max2"]]
    df = df[["id"]+ys]
    
    X = pd.merge(dfa, pd.merge(df, df_maxed, how="inner", on=["id"]), how="inner", on=["id"])
    y = dfy
    i = X[["id"]]
    X = X[attr_cols + X.columns.tolist()[-2*NUM_CLASSES:]]
    
    return X.as_matrix(), y.as_matrix(), i.as_matrix()



def load_max_testset(last_month=17, next_month=18, attr_cols=["t", "sex", "age", "seniority", "is_primary", "is_domestic", "income"]):
    pass



###


def load_concat_trainset(max_month=0, attr_cols=["t", "sex", "age", "seniority", "is_primary", "is_domestic", "income"], 
    lags=[1, 2, 3, 4, 5, 6, 9, 12], remove_non_buyers=False):
    
    df = pd.read_csv(trainset_filename)
    
    if max_month > 0 and max_month < MAX_SEQUENCE_LENGTH:
        print("max_month: " + str(max_month))
        df = df.loc[df["t"] <= max_month]
    
    df.loc[df["seniority"] < 0, "seniority"] = 0
    z_score_stats = get_z_score_stats(df)
    df = normalize_cols(df, z_score_stats)
    
    if remove_non_buyers:
        df = df_remove_non_buyers(df)
    
    df_maxs = df[["id", "t"]].groupby("id").max()
    df_maxs["id"] = df_maxs.index
    df_maxs.columns = ["t_max", "id"]
    df = pd.merge(df_maxs, df, how="outer", on=["id"])
    
    ys = df.columns.tolist()[-NUM_CLASSES:]
    
    # take last month as train X.attrs & test y
    X = df.loc[df["t"] == df["t_max"]][["id"] + attr_cols]
    y = df.loc[df["t"] == df["t_max"]][ys]
    
    # take lags of ys for train X
    for lag in lags:
        dfm = df.loc[df["t"] == df["t_max"] - lag][["id"] + ys]
        dfm.columns = ["id"] + [c + "_T-" + str(lag) for c in dfm.columns.tolist()[-NUM_CLASSES:]]
        X = pd.merge(X, dfm, how="outer", on=["id"])
    
    i = X[["id"]]
    X = X[attr_cols + X.columns.tolist()[-len(lags)*NUM_CLASSES:]].fillna(value=0)
    return X.as_matrix(), y.as_matrix(), i.as_matrix()



def load_concat_testset(last_month=17, next_month=18, attr_cols=["t", "sex", "age", "seniority", "is_primary", "is_domestic", "income"], 
    lags=[1, 2, 3, 4, 5, 6, 9, 12]):
    pass


#