import numpy as np
import pandas as pd

#u.data文件中包含了完整数据集。
u_data_path="ml-100k\\"
header = ['user_id', 'item_id', 'rating', 'timestamp']
df = pd.read_csv(u_data_path+'u.data', sep='\t', names=header)
print(df.head(5))
print(len(df))
#观察数据前两行。接下来，让我们统计其中的用户和电影总数。
n_users = df.user_id.unique().shape[0]  #unique()为去重.shape[0]行个数
n_items = df.item_id.unique().shape[0]
print ('Number of users = ' + str(n_users) + ' | Number of movies = ' + str(n_items))
from sklearn import model_selection as cv
train_data, test_data = cv.train_test_split(df, test_size=0.25)
#
# Create two user-item matrices, one for training and another for testing
# 差别在于train_data与test_data
train_data_matrix = np.zeros((n_users, n_items))
print(train_data_matrix.shape)
for line in train_data.itertuples():
    train_data_matrix[line[1]-1, line[2]-1] = line[3]
test_data_matrix = np.zeros((n_users, n_items))
for line in test_data.itertuples():
    test_data_matrix[line[1]-1, line[2]-1] = line[3]
# 你可以使用 sklearn 的pairwise_distances函数来计算余弦相似性。注意，因为评价都为正值输出取值应为0到1.
from sklearn.metrics.pairwise import pairwise_distances

user_similarity = pairwise_distances(train_data_matrix, metric='cosine')
#矩阵的转置实现主题的相似度
item_similarity = pairwise_distances(train_data_matrix.T, metric='cosine')
#print(user_similarity[0:5])

def predict(ratings, similarity, type='user'):
    if type == 'user':
        mean_user_rating = ratings.mean(axis=1)
        #print("user rating" + str(ratings))
        #print(mean_user_rating[0:5, np.newaxis])
        #You use np.newaxis so that mean_user_rating has same format as ratings
        ratings_diff = (ratings - mean_user_rating[:, np.newaxis]) #每个user对每个item的评分减去该item的得分平均值
        #print(ratings_diff[0:5])
        pred = mean_user_rating[:, np.newaxis] + similarity.dot(ratings_diff) / np.array([np.abs(similarity).sum(axis=1)]).T
    elif type == 'item':
        pred = ratings.dot(similarity) / np.array([np.abs(similarity).sum(axis=1)])
    return pred
item_prediction = predict(train_data_matrix, item_similarity, type='item')
user_prediction = predict(train_data_matrix, user_similarity, type='user')

#有许多的评价指标，但是用于评估预测精度最流行的指标之一是Root Mean Squared Error (RMSE)。
from sklearn.metrics import mean_squared_error
from math import sqrt
def rmse(prediction, ground_truth):
    prediction = prediction[ground_truth.nonzero()].flatten() #nonzero(a)返回数组a中值不为零的元素的下标,相当于对稀疏矩阵进行提取
    ground_truth = ground_truth[ground_truth.nonzero()].flatten()
    return sqrt(mean_squared_error(prediction, ground_truth))

print ('User-based CF RMSE: ' + str(rmse(user_prediction, test_data_matrix)))
print ('Item-based CF RMSE: ' + str(rmse(item_prediction, test_data_matrix)))