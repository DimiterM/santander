import time
import pandas as pd
import numpy as np

NUM_CLASSES = 24
MAX_SEQUENCE_LENGTH = 17

trainset_filename = "./catdf.csv"
testset_filename = "./testcatdf.csv"
print(trainset_filename, testset_filename)



"""
for each row of the dataframe 
prepend the number of rows for this id 
and the maximum month for this id
"""
def df_merge_counts_and_maxs(dfa, dfx):
    df_counts = dfx[["id", "t"]].groupby("id").count()
    df_counts["id"] = df_counts.index
    df_counts.columns = ["t_count", "id"]
    
    df_maxs = dfx[["id", "t"]].groupby("id").max()
    df_maxs["id"] = df_maxs.index
    df_maxs.columns = ["t_max", "id"]
    
    df_counts_and_maxs = pd.merge(df_counts, df_maxs, how="outer", on=["id"])
    df_maxs.columns = ["t", "id"]
    return {
        "attrs": pd.merge(df_counts_and_maxs, pd.merge(df_maxs, dfa, how="inner", on=["id", "t"]), how="outer", on=["id"]), 
        "xs": pd.merge(df_counts_and_maxs, dfx, how="outer", on=["id"])
    }



"""
build list of numpy arrays (buckets) for all sequence lengths
"""
def make_buckets_dataset(df_tmax_groups, df_attr_groups, ys):
    ids_buckets = np.array([]).reshape(0, 1)
    for i in range(len(df_tmax_groups)):
        if not df_tmax_groups[i].empty:
            ids = df_tmax_groups[i].iloc[:, 1:2]["id"].unique()
            ids_buckets = np.concatenate((ids_buckets, ids.reshape(ids.size, 1)), axis=0)
            df_tmax_groups[i] = df_tmax_groups[i].loc[:, ['t', 't_month']+ys].as_matrix()
            df_tmax_groups[i] = df_tmax_groups[i].reshape(df_tmax_groups[i].shape[0]//(i+2), i+2, len(['t', 't_month']+ys))
            df_attr_groups[i] = df_attr_groups[i].as_matrix()
    
    X_buckets = []
    y_buckets = []
    A_buckets = []
    for g, a in zip(df_tmax_groups, df_attr_groups):
        if g.size > 0:
            X_buckets.append(g[:, :-1, :])
            y_buckets.append(g[:, -1:, 2:].reshape(g.shape[0], g.shape[2] - 2))
            A_buckets.append(a)
    
    df_tmax_groups, df_attr_groups = None, None
    
    return A_buckets, X_buckets, y_buckets, ids_buckets



"""
remove all rows for all ids that never purchased anything
"""
def df_remove_non_buyers(df):
    print("non-buyers removed")
    ys = df.columns.tolist()[-24:]
    df["y_sum"] = df[ys].sum(axis=1)
    df_sums = df[["id","y_sum"]].groupby("id", as_index=False).sum()
    df_sums = df_sums.loc[df_sums["y_sum"] > 0]
    df.drop(["y_sum"], axis=1, inplace=True)
    return pd.merge(df_sums, df, how="inner", on=["id"])



"""
normalize attributes with huge values
"""
def get_z_score_stats(df, cols_to_norm=["age", "seniority", "income"]):
    stats = df[cols_to_norm].describe()
    return {"mean": stats.loc["mean",:], "std": stats.loc["std",:]}

def normalize_cols(df, z_score_stats, cols_to_norm=["age", "seniority", "income"]):
    for col in cols_to_norm:
        df[col] = (df[col] - z_score_stats["mean"][col]) / z_score_stats["std"][col]
    return df



def load_trainset(max_month=0, attr_cols=["t", "sex", "age", "seniority", "is_primary", "is_domestic", "income"], remove_non_buyers=False):
    df = pd.read_csv(trainset_filename)
    
    if max_month > 0 and max_month < MAX_SEQUENCE_LENGTH:
        print("max_month: " + str(max_month))
        df = df.loc[df["t"] <= max_month]
    
    df.loc[df["seniority"] < 0, "seniority"] = 0
    z_score_stats = get_z_score_stats(df)
    df = normalize_cols(df, z_score_stats)
    
    if remove_non_buyers:
        df = df_remove_non_buyers(df)
    
    df = df_merge_counts_and_maxs(df[df.columns.tolist()[:-NUM_CLASSES]], df[['id', 't', 't_month']+df.columns.tolist()[-NUM_CLASSES:]])
    # print("dfxs", df["xs"].columns.tolist())
    # print("dfattrs", df["attrs"].columns.tolist())
    
    df_tmax_groups = []
    for i in range(2, MAX_SEQUENCE_LENGTH + 1):
        df_tmax_groups.append(df["xs"].loc[df["xs"]["t_count"] == i].sort_values(["id", "t"]))
    
    df_attr_groups = []
    # print(df["xs"].notnull().values.all())
    # print(df["attrs"].notnull().values.all())
    for i in range(2, MAX_SEQUENCE_LENGTH + 1):
        df_attr_groups.append(df["attrs"].loc[df["attrs"]["t_count"] == i].sort_values(["id"])[attr_cols])
    
    ys = df["xs"].columns.tolist()[-NUM_CLASSES:]
    df = None
    
    A_buckets, X_buckets, y_buckets, _ = make_buckets_dataset(df_tmax_groups, df_attr_groups, ys)
    return A_buckets, X_buckets, y_buckets



def load_testset(last_month=17, next_month=18, attr_cols=["t", "sex", "age", "seniority", "is_primary", "is_domestic", "income"]):
    testdf = pd.DataFrame()
    if next_month > MAX_SEQUENCE_LENGTH:
        print("testset loaded")
        testdf = pd.read_csv(testset_filename)
    else:
        print("month " + str(next_month) + " testset loaded")
        testdf = pd.read_csv(trainset_filename)
        testdf = testdf.loc[testdf["t"] == next_month]
    
    df = pd.DataFrame()
    z_score_stats = dict()
    if last_month <= 0 or last_month >= MAX_SEQUENCE_LENGTH:
        print("trainset loaded")
        df = pd.read_csv(trainset_filename)
        z_score_stats = get_z_score_stats(df)
        df = df.loc[df["id"].isin(testdf["id"])]
    else:
        print("month " + str(last_month) + " trainset loaded")
        df = pd.read_csv(trainset_filename)
        df = df.loc[df["t"] <= last_month]
        z_score_stats = get_z_score_stats(df)
        df = df.loc[df["id"].isin(testdf["id"])]
    
    testdf.loc[testdf["seniority"] < 0, "seniority"] = 0
    testdf = normalize_cols(testdf, z_score_stats)
    
    ys = df.columns.tolist()[-NUM_CLASSES:]
    df = df[['id', 't', 't_month']+ys]
    df_cols = df.columns.tolist()
    df = pd.concat([df, testdf[['id', 't', 't_month']+([] if next_month > MAX_SEQUENCE_LENGTH else ys)]], ignore_index=True, copy=False)
    df = df[df_cols] # bug fix: concat unwantedly sorts DataFrame column names if they differ #4588
    
    testdf = df_merge_counts_and_maxs(testdf, df)
    # print("testdfxs", testdf["xs"].columns.tolist())
    # print("testdfattrs", testdf["attrs"].columns.tolist())
    
    testdf_tmax_groups = []
    for i in range(2, MAX_SEQUENCE_LENGTH + 2):
        testdf_tmax_groups.append(testdf["xs"].loc[testdf["xs"]["t_count"] == i].sort_values(["id", "t"]))
    
    testdf_attr_groups = []
    # print(testdf["xs"].isnull().sum())
    # print(testdf["attrs"].isnull().sum())
    for i in range(2, MAX_SEQUENCE_LENGTH + 2):
        testdf_attr_groups.append(testdf["attrs"].loc[testdf["attrs"]["t_count"] == i].sort_values(["id"])[attr_cols])
    
    df, testdf = None, None
    
    return make_buckets_dataset(testdf_tmax_groups, testdf_attr_groups, ys)


