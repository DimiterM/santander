import time
import pandas as pd
import numpy as np
# TODO: less code duplication


MAX_SEQUENCE_LENGTH = 17

def load_trainset():
    df = pd.read_csv("./df.csv")
    
    df_counts = df[["id", "t"]].groupby("id").count()
    df_counts["id"] = df_counts.index
    df_counts.columns = ["t_count", "id"]
    
    df_maxs = df[["id", "t"]].groupby("id").max()
    df_maxs["id"] = df_maxs.index
    df_maxs.columns = ["t_max", "id"]
    
    df = pd.merge(df_counts, df_maxs, how="outer", on=["id"]).merge(df, how="outer", on=["id"])
    
    df_tmax_groups = []
    for i in range(2, MAX_SEQUENCE_LENGTH + 1):
        df_tmax_groups.append(df.loc[df["t_count"] == i].sort_values(["id", "t"]))
    
    df_attr_groups = []
    for i in range(2, MAX_SEQUENCE_LENGTH + 1):    # TODO: transform attrs...
        df_attr_groups.append(df.loc[(df["t_count"] == i) & (df["t_max"] == df["t"])].sort_values(["id"])[ \
            ["t", "sex", "age", "seniority", "is_primary", "is_domestic", "income"]] \
            .fillna(value=0))
    
    ys = df.columns.tolist()[-24:]
    df = None
    
    for i in range(len(df_tmax_groups)):
        df_tmax_groups[i] = df_tmax_groups[i].loc[:, ['t', 't_month']+ys].as_matrix()
        df_tmax_groups[i] = df_tmax_groups[i].reshape(df_tmax_groups[i].shape[0]//(i+2), i+2, len(['t', 't_month']+ys))
        df_attr_groups[i] = df_attr_groups[i].as_matrix()
    
    X_buckets = []
    y_buckets = []
    for g in df_tmax_groups:
        X_buckets.append(g[:, :-1, :])
        y_buckets.append(g[:, -1:, 2:].reshape(g.shape[0], g.shape[2] - 2))
    
    df_tmax_groups = None
    A_buckets = df_attr_groups
    df_attr_groups = None
    
    for a, x, y in zip(A_buckets, X_buckets, y_buckets):
        print(a.shape, x.shape, y.shape)
    
    return A_buckets, X_buckets, y_buckets



def load_testset():
    testdf = pd.read_csv("./testdf.csv")
    df = pd.read_csv("./df.csv")
    df = df.loc[df["id"].isin(testdf["id"])]
    testdf = pd.concat([df, testdf], ignore_index=True, copy=False)

    testdf_counts = testdf[["id", "t"]].groupby("id").count()
    testdf_counts["id"] = testdf_counts.index
    testdf_counts.columns = ["t_count", "id"]

    testdf_maxs = df[["id", "t"]].groupby("id").max()
    testdf_maxs["id"] = testdf_maxs.index
    testdf_maxs.columns = ["t_max", "id"]

    testdf = pd.merge(testdf_counts, testdf_maxs, how="outer", on=["id"]).merge(testdf, how="outer", on=["id"])

    testdf_tmax_groups = []
    for i in range(2, MAX_SEQUENCE_LENGTH + 2):
        testdf_tmax_groups.append(testdf.loc[testdf["t_count"] == i].sort_values(["id", "t"]))

    testdf_attr_groups = []
    for i in range(2, MAX_SEQUENCE_LENGTH + 2):
        testdf_attr_groups.append(testdf.loc[(testdf["t_count"] == i) & (testdf["t"] == 18)].sort_values(["id"])[ \
            ["t", "sex", "age", "seniority", "is_primary", "is_domestic", "income"]] \
            .fillna(value=0))

    ys = df.columns.tolist()[-24:]
    testdf = None

    ids_test_buckets = np.array([]).reshape(0, 1)
    for i in range(len(testdf_tmax_groups)):
        if not testdf_tmax_groups[i].empty:
            ids = testdf_tmax_groups[i].iloc[:, 1:2]["id"].unique()
            ids_test_buckets = np.concatenate((ids_test_buckets, ids.reshape(ids.size, 1)), axis=0)
            testdf_tmax_groups[i] = testdf_tmax_groups[i].loc[:, ['t', 't_month']+ys].as_matrix()
            testdf_tmax_groups[i] = testdf_tmax_groups[i].reshape(testdf_tmax_groups[i].shape[0]//(i+2), i+2, len(['t', 't_month']+ys))
            testdf_attr_groups[i] = testdf_attr_groups[i].as_matrix()

    X_test_buckets = []
    y_test_buckets = []
    for g in testdf_tmax_groups:
        if g.size > 0:
            X_test_buckets.append(g[:, :-1, :])
            y_test_buckets.append(g[:, -1:, 2:].reshape(g.shape[0], g.shape[2] - 2))

    testdf_tmax_groups = None
    A_test_buckets = testdf_attr_groups
    testdf_attr_groups = None

    for a, x, y in zip(A_test_buckets, X_test_buckets, y_test_buckets):
        print(a.shape if a.size > 0 else "--", x.shape if x.size > 0 else "--", y.shape if y.size > 0 else "--")

    print(ids_test_buckets.shape)
    
    return A_test_buckets, X_test_buckets, y_test_buckets, ids_test_buckets


