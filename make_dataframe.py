"""
make_dataframe.py
Transforms the original Kaggle dataset (csv) to input dataset (csv) for the model
Params:
IMPUTE - True if the script should impute missing values
ISTEST - True for the test set, False for the train set
"""

import pandas as pd
import numpy as np
IMPUTE = True
ISTEST = False
df = pd.DataFrame()
tr = pd.read_csv("./"+("test" if ISTEST else "train")+"_ver2.csv", 
    dtype={"age":str, "antiguedad":str, "indrel_1mes":str, "conyuemp":str})

df["t"] = 12 * (pd.DatetimeIndex(pd.to_datetime(tr["fecha_dato"],format="%Y-%m-%d")).year - 2015) + pd.DatetimeIndex(pd.to_datetime(tr["fecha_dato"],format="%Y-%m-%d")).month
df["t_month"] = pd.DatetimeIndex(pd.to_datetime(tr["fecha_dato"],format="%Y-%m-%d")).month
tr.drop(["fecha_dato"], axis=1, inplace=True)

df["id"] = tr["ncodpers"]
tr.drop(["ncodpers"], axis=1, inplace=True)

# df["employee"] = tr["ind_empleado"]
# df.loc[df["employee"] == "S", "employee"] = "N"
# tr.drop(["ind_empleado"], axis=1, inplace=True)
# df["is_spouse"] = tr["conyuemp"].map({'S': 1, 'N': 0})
# df.loc[df["is_spouse"] == 1, "employee"] = "M"
# tr.drop(["conyuemp"], axis=1, inplace=True)
# df.drop(["is_spouse"], axis=1, inplace=True)
df["employee"] = tr["ind_empleado"]
df.loc[(df["employee"] == "S") | (df["employee"].isnull()), "employee"] = "N"
tr.drop(["ind_empleado", "conyuemp"], axis=1, inplace=True)

df["country"] = tr["pais_residencia"]
tr.drop(["pais_residencia"], axis=1, inplace=True)

df["sex"] = tr["sexo"].map({'H': 1, 'V': 0})
tr.drop(["sexo"], axis=1, inplace=True)

df["age"] = pd.to_numeric(tr["age"], downcast="integer", errors="coerce") ## floats ?!
tr.drop(["age"], axis=1, inplace=True)

df["seniority_new"] = pd.to_numeric(tr["ind_nuevo"], downcast="integer", errors="coerce") ## floats ?!
tr.drop(["ind_nuevo"], axis=1, inplace=True)
df["seniority"] = pd.to_numeric(tr["antiguedad"], downcast="integer", errors="coerce") ## floats ?!
tr.drop(["antiguedad", "fecha_alta"], axis=1, inplace=True)

df["is_primary"] = tr["indrel"]
df.loc[df["is_primary"] == 99, "is_primary"] = 0
# df["last_day_primary"] = pd.DatetimeIndex(pd.to_datetime(tr["ult_fec_cli_1t"],format="%Y-%m-%d")).day
tr.drop(["indrel", "ult_fec_cli_1t"], axis=1, inplace=True)

df["customer_type"] = tr["indrel_1mes"].str.replace('.0', '').replace('P', '0')
tr.drop(["indrel_1mes"], axis=1, inplace=True)

df["customer_rel"] = tr["tiprel_1mes"]
df.loc[df["customer_rel"] == "N", "customer_rel"] = "A"
tr.drop(["tiprel_1mes"], axis=1, inplace=True)

df["is_domestic"] = tr["indresi"].map({'S': 1, 'N': 0})
tr.drop(["indresi"], axis=1, inplace=True)
df["is_foreigner"] = tr["indext"].map({'S': 1, 'N': 0})
tr.drop(["indext"], axis=1, inplace=True)
df["is_dead"] = tr["indfall"].map({'S': 1, 'N': 0})
tr.drop(["indfall"], axis=1, inplace=True)

# df["channel"] = tr["canal_entrada"]
# tr.drop(["canal_entrada"], axis=1, inplace=True)
tr["canal_entrada_0"] = tr["canal_entrada"].str.slice(0,1)
tr["canal_entrada_1"] = tr["canal_entrada"].str.slice(1,2)
tr["canal_entrada_2"] = tr["canal_entrada"].str.slice(2,3)
tr["channel_0"] = 0
tr["channel_1"] = 0
tr["channel_2"] = 0
tr.loc[tr["canal_entrada"] == "RED", "channel_0"] = 2
tr.loc[~(tr["canal_entrada"].isnull()) & (tr["canal_entrada"].str.startswith("0")), "channel_0"] = 1
tr.loc[~(tr["canal_entrada"].isnull()) & (tr["canal_entrada"].str.startswith("0")), "channel_1"] = tr["canal_entrada"].map({'004': 0, '007': 1, '013': 2, '025': 3})
tr.loc[(tr["canal_entrada_0"] == "K") & (tr["canal_entrada_1"] != "0"), "channel_1"] = tr["canal_entrada_1"].map(
    {c: int(c, 36) - 9 for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"})
tr.loc[(tr["canal_entrada_0"] == "K") & (tr["canal_entrada_2"] != "0"), "channel_2"] = tr["canal_entrada_2"].map(
    {c: int(c, 36) - 9 for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"})
tr.loc[tr["canal_entrada"].isnull(), "channel_0"] = np.nan
tr.loc[tr["canal_entrada"].isnull(), "channel_1"] = np.nan
tr.loc[tr["canal_entrada"].isnull(), "channel_2"] = np.nan
df["channel_0"] = tr["channel_0"]
df["channel_1"] = tr["channel_1"]
df["channel_2"] = tr["channel_2"]
tr.drop(["canal_entrada", "canal_entrada_0", "canal_entrada_1", "canal_entrada_2", "channel_0", "channel_1", "channel_2"], axis=1, inplace=True)

df["province"] = pd.to_numeric(tr["cod_prov"], downcast="integer", errors="coerce") ## floats ?!
tr.drop(["cod_prov", "nomprov", "tipodom"], axis=1, inplace=True)

df["is_active"] = pd.to_numeric(tr["ind_actividad_cliente"], downcast="integer", errors="coerce") ## floats ?!
tr.drop(["ind_actividad_cliente"], axis=1, inplace=True)

df["income"] = tr["renta"]
tr.drop(["renta"], axis=1, inplace=True)
if ISTEST:
    df["income"] = pd.to_numeric(df["income"].str.replace('NA', '').replace(' ', ''), downcast="float", errors="coerce") ## for testset

df["segment"] = pd.to_numeric(tr["segmento"].str.slice(1,2), downcast="integer", errors="coerce") ## floats ?!
tr.drop(["segmento"], axis=1, inplace=True)

if not ISTEST:
    tr.columns = [
    "y_1_saving", "y_2_guarantees", "y_3_current", "y_4_derivate", "y_5_payroll", "y_6_junior", 
    "y_7_particular_M", "y_8_particular", "y_9_particular_P", "y_10_deposit_S", "y_11_deposit_M", 
    "y_12_deposit_L", "y_13_eacc", "y_14_funds", "y_15_mortgage", "y_16_pensions", "y_17_loans", 
    "y_18_taxes", "y_19_creditcard", "y_20_securities", "y_21_homeacc", 
    "y_22_payroll_2", "y_23_pensions_2", "y_24_direct"]
    tr.fillna(value=0, inplace=True)
    df = pd.concat([df, tr], axis=1)


if IMPUTE:
    df.isnull().any()
    for c in [
        'employee', 'country', 'sex', 'seniority_new', 'is_primary', 'customer_type', 'customer_rel', 
        'is_domestic', 'is_foreigner', 'is_dead', 'channel_0', 'channel_1', 'channel_2', 
        'province', 'is_active', 'segment']:
        df[c].fillna(value=df[c].mode().iloc[0], inplace=True)
    
    for c in ['age', 'seniority']:
        df[c].fillna(value=df[c].mean(), inplace=True)
    
    df['income'].fillna(value=df['income'].mean(), inplace=True)
    df.isnull().any()


df.to_csv(path_or_buf="./"+("test" if ISTEST else "")+"df.csv", index=False)





#df = pd.read_csv("./"+("test" if ISTEST else "")+"df.csv")

catdf = pd.DataFrame()
catdf["id"] = df["id"]

catdf["employee"] = df["employee"]
catdf["employee_bit_notN"] = 0
catdf.loc[catdf["employee"] != 'N', "employee_bit_notN"] = 1
catdf.drop(["employee"], axis=1, inplace=True)

catdf["country"] = df["country"]
catdf["country_num"] = 0.0
catdf.loc[catdf["country"] == 'ES', "country_num"] = 1.0
catdf.loc[catdf["country"].isin(['AT', 'BE', 'BG', 'CY', 'CZ', 'DK', 'EE', 'FI', 'FR', 'DE', 'GR', 'HU', 'IE', 'IT', 'LV', 'LT', 'LU', 'MT', 'NL', 'PL', 'PT', 'RO', 'SK', 'SI', 'SE', 'GB', 'HR']), "country_num"] = 0.5
catdf.drop(["country"], axis=1, inplace=True)

catdf["customer_type"] = df["customer_type"]
catdf["customer_type_bit_not1"] = 0
catdf.loc[catdf["customer_type"] != 1, "customer_type_bit_not1"] = 1
catdf.drop(["customer_type"], axis=1, inplace=True)

catdf["customer_rel"] = df["customer_rel"]
catdf["customer_rel_I"] = 0
catdf["customer_rel_A"] = 0
catdf.loc[(catdf["customer_rel"] == 'I') | (catdf["customer_rel"] == 'P'), "customer_rel_I"] = 1
catdf.loc[(catdf["customer_rel"] == 'A') | (catdf["customer_rel"] == 'R'), "customer_rel_A"] = 1
catdf.drop(["customer_rel"], axis=1, inplace=True)

dummies = pd.get_dummies(df["channel_0"], prefix="channel_0")
dummies.columns = [cn.replace(".0", "") for cn in dummies.columns.tolist()]
catdf = pd.concat([catdf, dummies], axis=1)

dummies = pd.get_dummies(df["channel_1"], prefix="channel_1")
dummies.columns = [cn.replace(".0", "") for cn in dummies.columns.tolist()]
catdf = pd.concat([catdf, dummies], axis=1)

# dummies = pd.get_dummies(df["channel_2"], prefix="channel_2")
# dummies.columns = [cn.replace(".0", "") for cn in dummies.columns.tolist()]
# catdf = pd.concat([catdf, dummies], axis=1)
# 
# dummies = pd.get_dummies(df["province"], prefix="province")
# dummies.columns = [cn.replace(".0", "") for cn in dummies.columns.tolist()]
# catdf = pd.concat([catdf, dummies], axis=1)
catdf["province_code"] = df["province"].map({
    51: 'AFR', 52: 'AFR', 
    4: 'AND', 11: 'AND', 14: 'AND', 18: 'AND', 21: 'AND', 23: 'AND', 29: 'AND', 41: 'AND', 
    22: 'ARA', 44: 'ARA', 50: 'ARA', 33: 'AST',  7: 'BAL',  1: 'BAS', 48: 'BAS', 20: 'BAS', 
    35: 'CAN', 38: 'CAN',  5: 'CAS',  9: 'CAS', 24: 'CAS', 34: 'CAS', 37: 'CAS', 40: 'CAS', 42: 'CAS', 47: 'CAS', 49: 'CAS', 
    8: 'CAT', 17: 'CAT', 25: 'CAT', 43: 'CAT', 39: 'CNB',  6: 'EXT', 10: 'EXT', 
    15: 'GAL', 27: 'GAL', 32: 'GAL', 36: 'GAL', 28: 'MAD',  2: 'MAN', 13: 'MAN', 16: 'MAN', 19: 'MAN', 45: 'MAN', 
    30: 'MUR', 31: 'NAV', 26: 'RIO',  3: 'VAL', 12: 'VAL', 46: 'VAL'})
dummies = pd.get_dummies(catdf["province_code"], prefix="province")
catdf = pd.concat([catdf, dummies], axis=1)
catdf.drop(["province_code"], axis=1, inplace=True)
catdf["province_pop"] = df["province"].map({
    52.0  : 0.0128, 51.0  : 0.0129, 42.0  : 0.0143, 44.0  : 0.0218, 40.0  : 0.0248,  5.0  : 0.0259, 
    34.0  : 0.0260, 49.0  : 0.0289, 16.0  : 0.0326, 22.0  : 0.0348, 19.0  : 0.0396,  1.0  : 0.0494, 
    26.0  : 0.0495, 32.0  : 0.0502, 37.0  : 0.0531, 27.0  : 0.0532,  9.0  : 0.0571,  2.0  : 0.0615, 10.0  : 0.0631, 
    25.0  : 0.0678, 24.0  : 0.0753, 21.0  : 0.0801, 13.0  : 0.0808, 47.0  : 0.0819, 39.0 : 0.0911, 
    12.0  : 0.0926, 31.0  : 0.0992, 23.0  : 0.1023,  6.0  : 0.1068,  4.0  : 0.1076, 45.0 : 0.1087, 
    20.0  : 0.1098, 17.0  : 0.1172, 14.0  : 0.1235, 43.0  : 0.1247, 18.0  : 0.1415, 36.0 : 0.1470, 
    50.0  : 0.1506, 38.0  : 0.1562, 33.0  : 0.1644, 35.0  : 0.1699,  7.0  : 0.1711, 15.0 : 0.1752, 
    48.0  : 0.1780, 11.0  : 0.1906, 30.0  : 0.2266, 29.0  : 0.2544, 41.0  : 0.2989,  3.0  : 0.2995, 46.0  : 0.3951,  8.0  : 0.8530, 28.0  : 1.0000})


dummies = pd.get_dummies(df["segment"], prefix="segment")
dummies.columns = [cn.replace(".0", "") for cn in dummies.columns.tolist()]
catdf = pd.concat([catdf, dummies], axis=1)



df = df[["t", "t_month", "id", "sex", "age", "seniority_new", "seniority", "is_primary", "is_domestic", "is_foreigner", "is_dead", "is_active", "income"] + (["y_1_saving", "y_2_guarantees", "y_3_current", "y_4_derivate", "y_5_payroll", "y_6_junior", "y_7_particular_M", "y_8_particular", "y_9_particular_P", "y_10_deposit_S", "y_11_deposit_M", "y_12_deposit_L", "y_13_eacc", "y_14_funds", "y_15_mortgage", "y_16_pensions", "y_17_loans", "y_18_taxes", "y_19_creditcard", "y_20_securities", "y_21_homeacc", "y_22_payroll_2", "y_23_pensions_2", "y_24_direct"] if not ISTEST else [])]
catdf_cols = catdf.columns.tolist()[1:]
catdf = catdf[catdf_cols]
catdf = pd.concat([catdf, df], axis=1) # catdf = pd.merge(catdf, df, how="outer", on=["id"])
catdf = catdf[["t", "t_month", "id", "sex", "age", "seniority_new", "seniority", "is_primary", "is_domestic", "is_foreigner", "is_dead", "is_active", "income"] + catdf_cols + (["y_1_saving", "y_2_guarantees", "y_3_current", "y_4_derivate", "y_5_payroll", "y_6_junior", "y_7_particular_M", "y_8_particular", "y_9_particular_P", "y_10_deposit_S", "y_11_deposit_M", "y_12_deposit_L", "y_13_eacc", "y_14_funds", "y_15_mortgage", "y_16_pensions", "y_17_loans", "y_18_taxes", "y_19_creditcard", "y_20_securities", "y_21_homeacc", "y_22_payroll_2", "y_23_pensions_2", "y_24_direct"] if not ISTEST else [])]
catdf.to_csv(path_or_buf="./"+("test" if ISTEST else "")+"catdf.csv", index=False)

