import matplotlib.pyplot as plt
import numpy as np

outliers = []
sample= [15, 101, 18, 7, 13, 16, 11, 21, 5, 15, 10, 9]
# plt.boxplot(sample, vert=False)
# plt.title("Detecting outliers using Boxplot")
# plt.xlabel('Sample')
# plt.show()



# def detect_outliers_zscore(data):
#   thres = 3
#   mean = np.mean(data)
#   std = np.std(data)
#   # print(mean, std)
#   for i in data:
#       z_score = (i-mean)/std
#       if (np.abs(z_score) > thres):
#           outliers.append(i)
#   return outliers# Driver code
# sample_outliers = detect_outliers_zscore(sample)
# print("Outliers from Z-scores method: ", sample_outliers)


def detect_outliers_iqr(data):
  data = sorted(data)
  q1 = np.percentile(data, 25)
  q3 = np.percentile(data, 75)
  # print(q1, q3)
  IQR = q3-q1
  lwr_bound = q1-(1.5*IQR)
  upr_bound = q3+(1.5*IQR)
  # print(lwr_bound, upr_bound)
  for i in data: 
      if (i<lwr_bound or i>upr_bound):
          outliers.append(i)
  return outliers# Driver code
sample_outliers = detect_outliers_iqr(sample)
print("Outliers from IQR method: ", sample_outliers)

detect_outliers_iqr(sample)

i = outliers[0]
# Trimming for i in sample_outliers:     
# Tìm vị trí các phần tử trong sample có giá trị nằm trong i
indices_to_delete = np.where(np.isin(sample, i))

# Loại bỏ các phần tử tương ứng từ mảng sample
a = np.delete(sample, indices_to_delete)

# In kết quả
print(a)
print(len(sample), len(a))

median = np.median(sample)
# Replace with median 
for i in sample_outliers:     
  c = np.where(sample==i, 14, sample) 
  print("Sample: ", sample) 
  print("New array: ",c) # print(x.dtype)