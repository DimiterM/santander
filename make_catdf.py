import pandas as pd
import numpy as np
ISTEST = False
df = pd.read_csv("./"+("test" if ISTEST else "")+"df.csv")

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

dummies = pd.get_dummies(df["channel_2"], prefix="channel_2")
dummies.columns = [cn.replace(".0", "") for cn in dummies.columns.tolist()]
catdf = pd.concat([catdf, dummies], axis=1)

dummies = pd.get_dummies(df["province"], prefix="province")
dummies.columns = [cn.replace(".0", "") for cn in dummies.columns.tolist()]
catdf = pd.concat([catdf, dummies], axis=1)

dummies = pd.get_dummies(df["segment"], prefix="segment")
dummies.columns = [cn.replace(".0", "") for cn in dummies.columns.tolist()]
catdf = pd.concat([catdf, dummies], axis=1)



df = df[["t", "t_month", "id", "sex", "age", "seniority_new", "seniority", "is_primary", "is_domestic", "is_foreigner", "is_dead", "is_active", "income"] + (["y_1_saving", "y_2_guarantees", "y_3_current", "y_4_derivate", "y_5_payroll", "y_6_junior", "y_7_particular_M", "y_8_particular", "y_9_particular_P", "y_10_deposit_S", "y_11_deposit_M", "y_12_deposit_L", "y_13_eacc", "y_14_funds", "y_15_mortgage", "y_16_pensions", "y_17_loans", "y_18_taxes", "y_19_creditcard", "y_20_securities", "y_21_homeacc", "y_22_payroll_2", "y_23_pensions_2", "y_24_direct"] if not ISTEST else [])]
catdf_cols = catdf.columns.tolist()[1:]
catdf = catdf[catdf_cols]
catdf = pd.concat([catdf, df], axis=1) # catdf = pd.merge(catdf, df, how="outer", on=["id"])
catdf = catdf[["t", "t_month", "id", "sex", "age", "seniority_new", "seniority", "is_primary", "is_domestic", "is_foreigner", "is_dead", "is_active", "income"] + catdf_cols + (["y_1_saving", "y_2_guarantees", "y_3_current", "y_4_derivate", "y_5_payroll", "y_6_junior", "y_7_particular_M", "y_8_particular", "y_9_particular_P", "y_10_deposit_S", "y_11_deposit_M", "y_12_deposit_L", "y_13_eacc", "y_14_funds", "y_15_mortgage", "y_16_pensions", "y_17_loans", "y_18_taxes", "y_19_creditcard", "y_20_securities", "y_21_homeacc", "y_22_payroll_2", "y_23_pensions_2", "y_24_direct"] if not ISTEST else [])]
catdf.to_csv(path_or_buf="./"+("test" if ISTEST else "")+"catdf.csv", index=False)
