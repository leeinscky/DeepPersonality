
#show_pkl.py

import pickle

# label里有5种personality： 'extraversion', 'neuroticism', 'agreeableness', 'conscientiousness', 'interview', 'openness' 分别对应：外向性，神经质，宜人性，尽责性，面试，开放性

# pkl文件所在路径
test_path='/Users/lizejian/cambridge/mphil_project/learn/udiva/DeepPersonality/datasets/ChaLearn2016_tiny/annotation/annotation_test.pkl' 
training_path='/Users/lizejian/cambridge/mphil_project/learn/udiva/DeepPersonality/datasets/ChaLearn2016_tiny/annotation/annotation_training.pkl' 
validation_path='/Users/lizejian/cambridge/mphil_project/learn/udiva/DeepPersonality/datasets/ChaLearn2016_tiny/annotation/annotation_validation.pkl'



# label of training data： 一共6种pesonality, 每种有6000个视频
path = training_path
f=open(path,'rb')
data=pickle.load(f, encoding='latin1')
print(len(data)) #6
# print(data['extraversion'])
print(len(data['extraversion'])) # 6000
print(len(data['neuroticism'])) # 6000
print(len(data['agreeableness'])) # 6000
print(len(data['conscientiousness'])) # 6000
print(len(data['interview'])) # 6000
print(len(data['openness'])) # 6000

# label of validation data: 一共有2000个视频的标签
path = validation_path
f=open(path,'rb')
data=pickle.load(f, encoding='latin1')
print(len(data)) #6
# print(data['extraversion'])
print(len(data['extraversion'])) # 2000
print(len(data['neuroticism'])) # 2000
print(len(data['agreeableness'])) # 2000
print(len(data['conscientiousness'])) # 2000
print(len(data['interview'])) # 2000
print(len(data['openness'])) # 2000

# label of test data: 一共有2000个视频的label
path = test_path
f=open(path,'rb')
data=pickle.load(f, encoding='latin1')
print(data.keys()) #ict_keys(['extraversion', 'neuroticism', 'agreeableness', 'conscientiousness', 'interview', 'openness'])
print(len(data)) #6
# print(data['extraversion'])
print(len(data['extraversion'])) # 2000
print(len(data['neuroticism'])) # 2000
print(len(data['agreeableness'])) # 2000
print(len(data['conscientiousness'])) # 2000
print(len(data['interview'])) # 2000
print(len(data['openness'])) # 2000


# 校验ChalearnFirstImpressionV2 官网上的注释文件文件: 一共有2000个视频的label
test_path_ChalearnFirstImpressionV2='/Users/lizejian/cambridge/mphil_project/relation_recognition/test/ChalearnFirstImpressionV2/annotation_test.pkl'
path = test_path_ChalearnFirstImpressionV2
f=open(path,'rb')
data=pickle.load(f, encoding='latin1')
print(data.keys()) #ict_keys(['extraversion', 'neuroticism', 'agreeableness', 'conscientiousness', 'interview', 'openness'])
print(len(data)) #6
print(data['extraversion'])
print(len(data['extraversion'])) # 2000
print(len(data['neuroticism'])) # 2000
print(len(data['agreeableness'])) # 2000
print(len(data['conscientiousness'])) # 2000
print(len(data['interview'])) # 2000
print(len(data['openness'])) # 2000

# 经过对比，发现 以下2个pkl文件完全一致，第一个文件是DeepPersonality里提供的ChaLearn2016_tiny里的，第二个文件是ChalearnFirstImpressionV2 官网的,所以官网也是直接给的pkkl文件
# /Users/lizejian/cambridge/mphil_project/learn/udiva/DeepPersonality/lzjnote/pkl_note/ChalearnFirstImpressionV2/annotation_test.pkl
# /Users/lizejian/cambridge/mphil_project/relation_recognition/test/ChalearnFirstImpressionV2/annotation_test.pkl
# print结果分别放在 annotation_test_pkl_extraversion 和 ChalearnFirstImpressionV2_annotation_test_pkl_extraversionv 里