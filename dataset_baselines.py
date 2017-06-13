import time
import pandas as pd
import numpy as np

from dataset import *



def build_max_dataset(df, attr_cols):
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
    
    X = pd.merge(dfa, pd.merge(df, df_maxed, how="inner", on=["id"]), how="outer", on=["id"]).fillna(value=0)
    y = dfy
    i = X[["id"]]
    X = X[attr_cols + X.columns.tolist()[-2*NUM_CLASSES:]]
    
    return X.as_matrix(), y.as_matrix(), i.as_matrix()



def load_max_trainset(max_month=17, attr_cols=["t", "sex", "age", "seniority", "is_primary", "is_domestic", "income"], remove_non_buyers=False, 
    scale_time_dim=False, include_time_dim_in_X=True):
    
    df = pd.read_csv(trainset_filename)
    
    if max_month > 0 and max_month < MAX_SEQUENCE_LENGTH:
        print("max_month: " + str(max_month))
        df = df.loc[df["t"] <= max_month]
    
    df.loc[df["seniority"] < 0, "seniority"] = 0
    z_score_stats = get_z_score_stats(df)
    df = normalize_cols(df, z_score_stats)
    
    if remove_non_buyers:
        df = df_remove_non_buyers(df)
    
    if scale_time_dim:
        df = scale_df_time_dim(df, max_month)
    
    return build_max_dataset(df, attr_cols)



def load_max_testset(train_month=17, test_month=18, attr_cols=["t", "sex", "age", "seniority", "is_primary", "is_domestic", "income"], 
    scale_time_dim=False, include_time_dim_in_X=True):
    
    testdf = pd.DataFrame()
    if test_month > MAX_SEQUENCE_LENGTH:
        print("testset loaded")
        testdf = pd.read_csv(testset_filename)
    else:
        print("month " + str(test_month) + " testset loaded")
        testdf = pd.read_csv(trainset_filename)
        testdf = testdf.loc[testdf["t"] == test_month]
    
    df = pd.DataFrame()
    z_score_stats = dict()
    last_month = test_month - 1
    if last_month <= 0 or last_month >= MAX_SEQUENCE_LENGTH:
        print("trainset loaded")
        df = pd.read_csv(trainset_filename)
        z_score_stats = get_z_score_stats(df.loc[df["t"] <= train_month])
        df = df.loc[df["id"].isin(testdf["id"])]
    else:
        print("month " + str(last_month) + " trainset loaded")
        df = pd.read_csv(trainset_filename)
        df = df.loc[df["t"] <= last_month]
        z_score_stats = get_z_score_stats(df.loc[df["t"] <= train_month])
        df = df.loc[df["id"].isin(testdf["id"])]
    
    testdf.loc[testdf["seniority"] < 0, "seniority"] = 0
    testdf = normalize_cols(testdf, z_score_stats)
    
    if scale_time_dim:
        df = scale_df_time_dim(df, last_month)
        testdf = scale_df_time_dim(testdf, last_month)
    
    ys = df.columns.tolist()[-NUM_CLASSES:]
    df = df[['id'] + attr_cols + ys]
    df_cols = df.columns.tolist()
    df = pd.concat([df, testdf[['id'] + attr_cols + ([] if test_month > MAX_SEQUENCE_LENGTH else ys)]], ignore_index=True, copy=False)
    df = df[df_cols] # bug fix: concat unwantedly sorts DataFrame column names if they differ #4588
    
    return build_max_dataset(df, attr_cols)



###


def build_concat_dataset(df, attr_cols, lags):
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



def load_concat_trainset(max_month=17, attr_cols=["t", "sex", "age", "seniority", "is_primary", "is_domestic", "income"], 
    lags=[1, 2, 3, 4, 5, 6, 9, 12], remove_non_buyers=False, scale_time_dim=False, include_time_dim_in_X=True):
    
    df = pd.read_csv(trainset_filename)
    
    if max_month > 0 and max_month < MAX_SEQUENCE_LENGTH:
        print("max_month: " + str(max_month))
        df = df.loc[df["t"] <= max_month]
    
    df.loc[df["seniority"] < 0, "seniority"] = 0
    z_score_stats = get_z_score_stats(df)
    df = normalize_cols(df, z_score_stats)
    
    if remove_non_buyers:
        df = df_remove_non_buyers(df)
    
    if scale_time_dim:
        df = scale_df_time_dim(df, max_month)
    
    return build_concat_dataset(df, attr_cols, lags)



def load_concat_testset(train_month=17, test_month=18, attr_cols=["t", "sex", "age", "seniority", "is_primary", "is_domestic", "income"], 
    lags=[1, 2, 3, 4, 5, 6, 9, 12], scale_time_dim=False, include_time_dim_in_X=True):
    testdf = pd.DataFrame()
    if test_month > MAX_SEQUENCE_LENGTH:
        print("testset loaded")
        testdf = pd.read_csv(testset_filename)
    else:
        print("month " + str(test_month) + " testset loaded")
        testdf = pd.read_csv(trainset_filename)
        testdf = testdf.loc[testdf["t"] == test_month]
    
    df = pd.DataFrame()
    z_score_stats = dict()
    last_month = test_month - 1
    if last_month <= 0 or last_month >= MAX_SEQUENCE_LENGTH:
        print("trainset loaded")
        df = pd.read_csv(trainset_filename)
        z_score_stats = get_z_score_stats(df.loc[df["t"] <= train_month])
        df = df.loc[df["id"].isin(testdf["id"])]
    else:
        print("month " + str(last_month) + " trainset loaded")
        df = pd.read_csv(trainset_filename)
        df = df.loc[df["t"] <= last_month]
        z_score_stats = get_z_score_stats(df.loc[df["t"] <= train_month])
        df = df.loc[df["id"].isin(testdf["id"])]
    
    testdf.loc[testdf["seniority"] < 0, "seniority"] = 0
    testdf = normalize_cols(testdf, z_score_stats)
    
    if scale_time_dim:
        df = scale_df_time_dim(df, last_month)
        testdf = scale_df_time_dim(testdf, last_month)
    
    ys = df.columns.tolist()[-NUM_CLASSES:]
    df = df[['id'] + attr_cols + ys]
    df_cols = df.columns.tolist()
    df = pd.concat([df, testdf[['id'] + attr_cols + ([] if test_month > MAX_SEQUENCE_LENGTH else ys)]], ignore_index=True, copy=False)
    df = df[df_cols] # bug fix: concat unwantedly sorts DataFrame column names if they differ #4588
    
    return build_concat_dataset(df, attr_cols, lags)
    

#