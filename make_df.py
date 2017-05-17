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
