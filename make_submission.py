import time
import numpy as np
import pandas as pd
RESULTS_FILENAME = "./res_man_test_6.csv"
SUBMIT_FILENAME  = "./submission_man_test_6.csv"
print(time.strftime("%H:%M:%S", time.localtime()))
res = pd.read_csv(RESULTS_FILENAME, header=None)
# row_multiplier = 0.5 / (res.mean().as_matrix()[1:] + (2 * res.std().as_matrix()[1:]))
# row_multiplier = 0.5 / (res.sum(0) / (res > 0).sum(0)).as_matrix()[1:]
ROW_THRESHOLD = 0.000001 # 0.007 # 0.000001
PRODUCT_LABELS = [
    "ind_ahor_fin_ult1","ind_aval_fin_ult1","ind_cco_fin_ult1","ind_cder_fin_ult1","ind_cno_fin_ult1","ind_ctju_fin_ult1",
    "ind_ctma_fin_ult1","ind_ctop_fin_ult1","ind_ctpp_fin_ult1","ind_deco_fin_ult1","ind_deme_fin_ult1","ind_dela_fin_ult1",
    "ind_ecue_fin_ult1","ind_fond_fin_ult1","ind_hip_fin_ult1","ind_plan_fin_ult1","ind_pres_fin_ult1","ind_reca_fin_ult1",
    "ind_tjcr_fin_ult1","ind_valo_fin_ult1","ind_viv_fin_ult1","ind_nomina_ult1","ind_nom_pens_ult1","ind_recibo_ult1"]

NUM_CLASSES = len(PRODUCT_LABELS)
K_VAL = 7
iter_csv = pd.read_csv("./df.csv", iterator=True, chunksize=1000000)
lastdf = pd.concat([chunk[chunk["t"] == 17] for chunk in iter_csv])
lastdf = pd.concat([lastdf["id"], lastdf.iloc[:,-NUM_CLASSES:]], axis=1)

res = res.merge(lastdf, left_on=0, right_on="id", how="left", copy=False).as_matrix()
print("y_all_sum = ", (res[:,1:NUM_CLASSES+1] >= ROW_THRESHOLD).sum(0))
print("y_new_sum = ", ((res[:,1:NUM_CLASSES+1] >= ROW_THRESHOLD) & (res[:,NUM_CLASSES+2:] == 0)).sum(0))

y_recom_sum = [0] * NUM_CLASSES
ixs_recommendation_rows = []
for row in res:
    row_id = int(row[0])
    row_prev = row[NUM_CLASSES+2:]
    row = row[1:NUM_CLASSES+1]
    # row = row * row_multiplier
    row_boolean = (row >= ROW_THRESHOLD)
    row_boolean_ixs = np.where(row_boolean)[0]
    row_previous_ixs = np.where(row_prev >= 1.0)[0]
    row_boolean_ixs = np.setdiff1d(row_boolean_ixs, row_previous_ixs)
    row_boolean_vals = row[row_boolean_ixs]
    row_recommendations_ixs = row_boolean_ixs[np.argsort(row_boolean_vals)[::-1]]
    for c in row_recommendations_ixs.tolist()[:K_VAL]:
        y_recom_sum[c] += 1
    ixs_recommendation_rows.append( str(row_id) + "," + ' '.join( list(map(lambda x: PRODUCT_LABELS[x], row_recommendations_ixs.tolist())) ) )

print("y_recom_sum = ", y_recom_sum)
with open(SUBMIT_FILENAME, 'w', newline='') as f:
    f.write( '\r\n'.join(["ncodpers,added_products"] + ixs_recommendation_rows) )

print(time.strftime("%H:%M:%S", time.localtime()))
