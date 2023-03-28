# Extract ID and KNOWN colum from csv file, transfer to dict and save as pkl file

import argparse
import os
import pandas as pd
import pickle
import json
valid_id_relation_dict = {}

def parse_args():
    parser = argparse.ArgumentParser(description='Extract ID and KNOWN colum from csv file, transfer to dict and save as pkl file')
    parser.add_argument('--csv_path', type=str, default=None, help='path to csv file')
    parser.add_argument('--output_path', type=str, default=None, help='path to output pkl file')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    # 获取csv文件的路径
    csv_file = args.csv_path
    # 获取输出文件的路径
    output_file = args.output_path
    
    # N/A 在pandas.read_csv中会被转换为NaN，所以需要keep_default_na=False，keep_default_na 参数用来控制是否要将被判定的缺失值转换为NaN这一过程，默认为True。当keep_default_na=False时，源文件中出现的什么值，DataFrame中就是什么值。
    df = pd.read_csv(csv_file, converters={'NewSessionID': str}, keep_default_na=False)
    
    # relationship level: N/A(stranger) < Acquaintances < Friends < Very good friends  (程度: N/A陌生人 < 认识的人 < 朋友 < 很好的朋友)
    # notice: session中有部分User1.level 和 User2.level 不同，以程度较高的为准，例如：User1.level = Acquaintances, User2.level = Friends, 则认为label是Friends
    
    relation_type = ['N/A', 'Acquaintances', 'Friends', 'Very good friends'] # 四分类
    # 如果df['User1.level'] 和 df['User2.level'] 相同，则index为df['User1.level']在relation_type中的索引，然后将relation_label中对应的index设为1，其他设为0; 如果df['User1.level'] 和 df['User2.level'] 不同，则index为df['User1.level']和df['User2.level']在relation_type中的索引最大值，然后将relation_label中对应的index设为1，其他设为0
    
    id_relation_dict = {}
    for index, row in df.iterrows():
        relation_label = [0] * len(relation_type)
        # print('\ndf.iterrows, index:', index, 'User1.level:', row['User1.level'], 'User2.level:', row['User2.level'])
        user1_level_index = relation_type.index(row['User1.level'])
        user2_level_index = relation_type.index(row['User2.level'])
        if row['User1.level'] == row['User2.level']:
            # print('1-:', row['User1.level'], ',', row['User2.level'], 'index:', user1_level_index)
            relation_label[user1_level_index] = 1
        else:
            relation_label[max(user1_level_index, user2_level_index)] = 1
            # print('2-:', row['User1.level'], ',', row['User2.level'], ',index1:', user1_level_index, ',index2:', user2_level_index, ', max index:', max(user1_level_index, user2_level_index))
        # print('relation_label:', relation_label)
        # 将 csv中NewSessionID那一列的值作为 key，上面得到的relation_label 作为 value 追加到字典中
        id_relation_dict[row['NewSessionID']] = relation_label
    
    with open(output_file, 'wb') as f:
        pickle.dump(id_relation_dict, f)
    f.close()

    # 验证是否正确 id_relation_dict 转为json
    # print('id_relation_dict:', json.dumps(id_relation_dict))
    for k, v in id_relation_dict.items():
        print('NewSessionID:', k, 'label:', v)

if __name__ == '__main__':
    main()

'''
# 打印结果 符合预期！
NewSessionID: 1 label: [0, 0, 1, 0]
NewSessionID: 2 label: [0, 1, 0, 0]
NewSessionID: 3 label: [0, 1, 0, 0]
NewSessionID: 4 label: [0, 0, 1, 0]
NewSessionID: 5 label: [0, 0, 1, 0]
NewSessionID: 6 label: [0, 1, 0, 0]
NewSessionID: 7 label: [1, 0, 0, 0]
NewSessionID: 8 label: [1, 0, 0, 0]
NewSessionID: 9 label: [0, 0, 1, 0]
NewSessionID: 10 label: [0, 0, 1, 0]
NewSessionID: 11 label: [1, 0, 0, 0]
NewSessionID: 12 label: [1, 0, 0, 0]
NewSessionID: 13 label: [0, 0, 1, 0]
NewSessionID: 14 label: [0, 1, 0, 0]
NewSessionID: 15 label: [0, 0, 0, 1]
NewSessionID: 16 label: [1, 0, 0, 0]
NewSessionID: 17 label: [1, 0, 0, 0]
NewSessionID: 18 label: [1, 0, 0, 0]
NewSessionID: 19 label: [0, 1, 0, 0]
NewSessionID: 20 label: [1, 0, 0, 0]
NewSessionID: 21 label: [0, 1, 0, 0]
NewSessionID: 22 label: [1, 0, 0, 0]
NewSessionID: 23 label: [1, 0, 0, 0]
NewSessionID: 24 label: [1, 0, 0, 0]
NewSessionID: 25 label: [1, 0, 0, 0]
NewSessionID: 26 label: [1, 0, 0, 0]
NewSessionID: 27 label: [1, 0, 0, 0]
NewSessionID: 28 label: [1, 0, 0, 0]
NewSessionID: 29 label: [0, 1, 0, 0]
NewSessionID: 30 label: [1, 0, 0, 0]
NewSessionID: 31 label: [1, 0, 0, 0]
NewSessionID: 32 label: [0, 1, 0, 0]
NewSessionID: 33 label: [1, 0, 0, 0]
NewSessionID: 34 label: [1, 0, 0, 0]
NewSessionID: 35 label: [1, 0, 0, 0]
NewSessionID: 36 label: [1, 0, 0, 0]
NewSessionID: 37 label: [1, 0, 0, 0]
NewSessionID: 38 label: [1, 0, 0, 0]
NewSessionID: 39 label: [1, 0, 0, 0]
NewSessionID: 40 label: [1, 0, 0, 0]
NewSessionID: 41 label: [1, 0, 0, 0]
NewSessionID: 42 label: [1, 0, 0, 0]
NewSessionID: 43 label: [1, 0, 0, 0]
NewSessionID: 44 label: [1, 0, 0, 0]
NewSessionID: 45 label: [0, 1, 0, 0]
NewSessionID: 46 label: [1, 0, 0, 0]
NewSessionID: 47 label: [1, 0, 0, 0]
NewSessionID: 48 label: [1, 0, 0, 0]
NewSessionID: 49 label: [1, 0, 0, 0]
NewSessionID: 50 label: [0, 0, 0, 1]
NewSessionID: 51 label: [0, 0, 0, 1]
NewSessionID: 52 label: [0, 0, 0, 1]
NewSessionID: 53 label: [0, 0, 0, 1]
NewSessionID: 54 label: [0, 0, 0, 1]
NewSessionID: 55 label: [0, 1, 0, 0]
NewSessionID: 56 label: [0, 1, 0, 0]
NewSessionID: 57 label: [0, 1, 0, 0]
NewSessionID: 58 label: [1, 0, 0, 0]
NewSessionID: 59 label: [0, 1, 0, 0]
NewSessionID: 60 label: [0, 1, 0, 0]
NewSessionID: 61 label: [0, 1, 0, 0]
NewSessionID: 62 label: [0, 1, 0, 0]
NewSessionID: 63 label: [0, 0, 1, 0]
NewSessionID: 64 label: [0, 1, 0, 0]
NewSessionID: 65 label: [0, 0, 0, 1]
NewSessionID: 66 label: [0, 1, 0, 0]
NewSessionID: 67 label: [0, 0, 0, 1]
NewSessionID: 68 label: [0, 0, 0, 1]
NewSessionID: 69 label: [0, 1, 0, 0]
NewSessionID: 70 label: [1, 0, 0, 0]
NewSessionID: 71 label: [0, 1, 0, 0]
NewSessionID: 72 label: [0, 1, 0, 0]
NewSessionID: 73 label: [0, 1, 0, 0]
NewSessionID: 74 label: [0, 1, 0, 0]
NewSessionID: 75 label: [0, 1, 0, 0]
NewSessionID: 76 label: [0, 0, 0, 1]
NewSessionID: 77 label: [0, 0, 0, 1]
NewSessionID: 78 label: [1, 0, 0, 0]
NewSessionID: 79 label: [0, 0, 1, 0]
NewSessionID: 80 label: [0, 1, 0, 0]
NewSessionID: 81 label: [0, 1, 0, 0]
NewSessionID: 82 label: [0, 0, 1, 0]
NewSessionID: 83 label: [0, 0, 1, 0]
NewSessionID: 84 label: [0, 0, 0, 1]

'''