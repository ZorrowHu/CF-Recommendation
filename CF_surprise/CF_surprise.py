import io
from surprise import KNNBaseline
from surprise import Dataset, Reader
import pandas as pd

def read_iter_names():
    # u.item格式：编号|电影名字|评分|url
    # 获取电影名到id和id到电影名的映射
    item_file = 'ml-100k/u.item'
    rid_2_name = {}
    name_2_rid = {}
    with io.open(item_file, 'r', encoding='ISO-8859-1') as f:
        for line in f:
            line = line.split('|')
            rid_2_name[line[0]] = line[1]
            name_2_rid[line[1]] = line[0]
    return rid_2_name, name_2_rid


# u.data数据格式为 user item rating timestamp；
reader = Reader(line_format='user item rating timestamp', sep='\t')
file_path = 'ml-100k'
data = Dataset.load_from_file(file_path=file_path + '/u.data', reader=reader)
train_set = data.build_full_trainset()

# #基于物品的推荐
# sim_options = {'name': 'pearson_baseline', 'user_based': False}
# item_baesd = KNNBaseline(sim_options=sim_options)
# #item_baesd.train(train_set) 过时的写法
# item_baesd.fit(train_set)

# 获取id对应的电影名列表，由于中途涉及一个id转换，所以要双向
rid_2_name, name_2_rid = read_iter_names()


# # raw-id映射到内部id
# toy_story_raw_id = name_2_rid['Toy Story (1995)']
# toy_story_inner_id = item_baesd.trainset.to_inner_iid(toy_story_raw_id)

# # 获取toy story对应的内部id 并由此取得其对应的k个近邻 k个近邻对应的也是内部id
# toy_story_neighbors = item_baesd.get_neighbors(toy_story_inner_id, k=10)

# # 近邻内部id转换为对应的名字
# toy_story_neighbors = (item_baesd.trainset.to_raw_iid(inner_id)
#                        for inner_id in toy_story_neighbors)
# toy_story_neighbors = (rid_2_name[rid] for rid in toy_story_neighbors)

# print('基于皮尔逊相似计算得到与toy story相近的十个电影为：\n')
# for moives in toy_story_neighbors:
#     print(moives)



#基于用户的推荐
uid = '1'   #为标号uid的用户推荐n部电影
n = 10

# u_data_path="ml-100k\\"
# header = ['user_id', 'item_id', 'rating', 'timestamp']
# data_df = pd.read_csv(u_data_path+'u.data', sep='\t', names=header)
file_path = 'ml-100k/u.data'
data_df = pd.read_csv(file_path, sep='\t', header=None, names=['user','item','rating','timestamp'])
data_df = data_df.astype(str)

sim_options = {'name': 'pearson_baseline', 'user_based': True}
user_based = KNNBaseline(sim_options=sim_options)
#user_based.train(train_set) 过时的写法
user_based.fit(train_set)

#将原始id转换为内部id
inner_id = user_based.trainset.to_inner_uid(uid)
#使用get_neighbors方法得到10个最相似用户
neighbors = user_based.get_neighbors(inner_id, k=10)
# print('inner_id:')
# print(neighbors)
neighbors = ([user_based.trainset.to_raw_uid(x) for x in neighbors])
# print('raw_id:')
# print(neighbors)
# print('end')
recommendations = []

#把评分为5的电影加入推荐列表
for user in neighbors:
    if len(recommendations) > n:
        break
    item = data_df[data_df['user']==user]
    item = item[item['rating']=='5']['item']
    for i in item:
        recommendations.append(rid_2_name[i])

print('Recommendations for user are:')
for i, j in enumerate(recommendations):
    if i >= 10:
        break
    print(j)
# # raw-id映射到内部id
# toy_story_raw_id = name_2_rid['Shanghai Triad (Yao a yao yao dao waipo qiao) (1995)']
# print(toy_story_raw_id)
# toy_story_inner_id = user_based.trainset.to_inner_iid(toy_story_raw_id)
# print(toy_story_inner_id)
# # 获取toy story对应的内部id 并由此取得其对应的k个近邻 k个近邻对应的也是内部id
# toy_story_neighbors = user_based.get_neighbors(toy_story_inner_id, k=10)

# # 近邻内部id转换为对应的名字
# toy_story_neighbors = (user_based.trainset.to_raw_iid(inner_id)
#                        for inner_id in toy_story_neighbors)
# toy_story_neighbors = (rid_2_name[rid] for rid in toy_story_neighbors)

# print('\n 基于用户的推荐（皮尔逊相似计算得到）与toy story相近的十个电影为：\n')
# for moives in toy_story_neighbors:
#     print(moives)