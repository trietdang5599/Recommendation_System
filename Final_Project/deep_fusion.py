from config import args
from data_process import *
from utils import CreateAndWriteCSV, merge_csv_columns
from review_rating_merge import z_review, z_item



#============================ Calulate U/I deep ===============================

def Calculate_Deep(v, z, start):
    list_sum = {}
    i = start
    for name, z_i in z.items():
        if i < len(v):  # Đảm bảo vẫn còn phần tử trong danh sách v
            v_i = v[i]
            sum_v_z = v_i * z_i
            sum_v2_z2 = (v_i**2) * (z_i**2)
            result = (1 / 2) * ((sum_v_z)**2 - sum_v2_z2)
            list_sum[name] = result
            i += 1
    return list_sum

#==============================================================================




# device = torch.device(args.device)
dataset = get_dataset(args.dataset_name, args.data_feature)
# model = get_model(dataset).to(device)
model = get_model(dataset)
v_list = np.array(model.embedding.embedding.weight.data.tolist())
# print(v_list)
u_deep = Calculate_Deep(v_list, z_review, 0)
i_deep = Calculate_Deep(v_list, z_item, len(z_review))
CreateAndWriteCSV("u_deep", u_deep)
CreateAndWriteCSV("i_deep", i_deep)
TransformLabel_Deep(pd.read_csv("feature/u_deep.csv", sep=',', engine='c', header='infer').to_numpy()[:, :3], "feature/transformed_udeep.csv")
TransformLabel_Deep(pd.read_csv("feature/i_deep.csv", sep=',', engine='c', header='infer').to_numpy()[:, :3], "feature/transformed_ideep.csv")


# Sử dụng function merge_csv_columns
merge_csv_columns(args.data_feature, 'reviewerID', 'feature/u_deep.csv', 'Key', 'Array', 'Udeep')
merge_csv_columns(args.data_feature, 'itemID', 'feature/i_deep.csv', 'Key', 'Array', 'Ideep')
merge_csv_columns(args.data_feature, 'reviewerID', 'feature/transformed_udeep.csv', 'ID', 'Array', 'Udeep')
merge_csv_columns(args.data_feature, 'itemID', 'feature/transformed_ideep.csv', 'ID', 'Array', 'Ideep')
#endregion