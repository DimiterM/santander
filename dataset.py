import time
import pandas as pd
import numpy as np

NUM_CLASSES = 24
MAX_SEQUENCE_LENGTH = 17

trainset_filename = "./df.csv"
testset_filename = "./testdf.csv"
print(trainset_filename, testset_filename)



"""
for each row of the dataframe 
prepend the number of rows for this id 
and the maximum month for this id
"""
def df_merge_counts_and_maxs(df):
    df_counts = df[["id", "t"]].groupby("id").count()
    df_counts["id"] = df_counts.index
    df_counts.columns = ["t_count", "id"]
    
    df_maxs = df[["id", "t"]].groupby("id").max()
    df_maxs["id"] = df_maxs.index
    df_maxs.columns = ["t_max", "id"]
    
    return pd.merge(df_counts, df_maxs, how="outer", on=["id"]).merge(df, how="outer", on=["id"])



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



def load_trainset(max_month=0, cols=["t", "sex", "age", "seniority", "is_primary", "is_domestic", "income"], remove_non_buyers=False):
    df = pd.read_csv(trainset_filename)
    
    if max_month > 0 and max_month < MAX_SEQUENCE_LENGTH:
        print("max_month: " + str(max_month))
        df = df.loc[df["t"] <= max_month]
    
    if remove_non_buyers:
        df = df_remove_non_buyers(df)
    
    df = df_merge_counts_and_maxs(df)
    
    df_tmax_groups = []
    for i in range(2, MAX_SEQUENCE_LENGTH + 1):
        df_tmax_groups.append(df.loc[df["t_count"] == i].sort_values(["id", "t"]))
    
    df_attr_groups = []
    print(df.notnull().values.all())
    for i in range(2, MAX_SEQUENCE_LENGTH + 1):
        df_attr_groups.append(df.loc[(df["t_count"] == i) & (df["t_max"] == df["t"])].sort_values(["id"])[cols])
    
    ys = df.columns.tolist()[-NUM_CLASSES:]
    df = None
    
    A_buckets, X_buckets, y_buckets, _ = make_buckets_dataset(df_tmax_groups, df_attr_groups, ys)
    return A_buckets, X_buckets, y_buckets



def load_testset(month=18, cols=["t", "sex", "age", "seniority", "is_primary", "is_domestic", "income"]):
    testdf = pd.DataFrame()
    df = pd.DataFrame()
    if month > MAX_SEQUENCE_LENGTH:
        print("testset loaded")
        testdf = pd.read_csv(testset_filename)
        df = pd.read_csv(trainset_filename)
        df = df.loc[df["id"].isin(testdf["id"])]
    else:
        print("month " + str(month) + " testset loaded")
        testdf = pd.read_csv(trainset_filename)
        testdf = testdf.loc[testdf["t"] == month]
        df = pd.read_csv(trainset_filename)
        df = df.loc[(df["id"].isin(testdf["id"])) & (df["t"] < month)]

    print(df.isnull().any())
    print(testdf.isnull().any())
    testdf = pd.concat([df, testdf], ignore_index=True, copy=False)

    testdf = df_merge_counts_and_maxs(testdf)

    testdf_tmax_groups = []
    for i in range(2, MAX_SEQUENCE_LENGTH + 2):
        testdf_tmax_groups.append(testdf.loc[testdf["t_count"] == i].sort_values(["id", "t"]))

    testdf_attr_groups = []
    print(testdf.notnull().values.all())
    print(df.isnull().any())
    print(testdf.isnull().any())
    for i in range(2, MAX_SEQUENCE_LENGTH + 2):
        testdf_attr_groups.append(testdf.loc[(testdf["t_count"] == i) & (testdf["t"] == month)].sort_values(["id"])[cols])

    ys = df.columns.tolist()[-NUM_CLASSES:]
    df, testdf = None, None
    
    return make_buckets_dataset(testdf_tmax_groups, testdf_attr_groups, ys)


