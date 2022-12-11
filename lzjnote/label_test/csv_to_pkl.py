# Extract ID and KNOWN colum from csv file, transfer to dict and save as pkl file

import argparse
import os
import pandas as pd
import pickle

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
    df = pd.read_csv(csv_file, converters={'ID': str})
    
    # if KNOWN column is True, then set 1, else set 0
    known_label = df['KNOWN'].apply(lambda x: 1 if x else 0)
    # if KNOWN column is True, then set 0, else set 1
    not_known_label = df['KNOWN'].apply(lambda x: 0 if x else 1)
    # put both known_label and not_known_label into a list
    label_list = list(zip(known_label, not_known_label))
    # construct a dict, key is ID, value is label_list
    
    id_known_dict = dict(zip(df['ID'], label_list)) # zip 用于将可迭代的对象作为参数，将对象中对应的元素打包成一个个元组，然后返回由这些元组组成的对象
    print('id_known_dict:', id_known_dict)
    with open(output_file, 'wb') as f:
        pickle.dump(id_known_dict, f)

if __name__ == '__main__':
    main()

'''
# 打印结果
{
    128129: True,
    127128: True,
    127129: True,
    55128: True,
    59134: False,
    118125: True,
    55125: False,
    58110: False,
    58059: True,
    110136: True,
    9106: False,
    92107: False,
    139140: True,
    15126: True,
    15133: False,
    112132: False,
    113132: False,
    112113: True,
    41083: True,
    42084: True,
    41084: True,
    42092: False,
    79142: True,
    97098: True,
    98126: False,
    97142: False,
    76122: False,
    122144: False,
    13041: False,
    5013: False,
    3005: True,
    2003: True,
    5134: False,
    20150: True,
    17150: True,
    20149: False,
    17149: False,
    51076: True,
    44156: False,
    43152: False,
    152153: True,
    6007: True,
    6153: False,
    20025: False,
    18025: False,
    18020: True,
    27076: False,
    27113: False,
    4115: False,
    4096: False,
    92096: True,
    106108: False,
    92108: True,
    76143: True,
    20090: False,
    40090: True,
    35040: False,
    34133: False,
    34035: True,
    25157: True,
    25044: False,
    44157: False,
    148151: True,
    43079: False,
    106148: False,
    43143: False,
    118154: False,
    27154: False,
    27118: False,
    17027: False,
    10011: True,
    10034: False,
    11034: False,
    34121: False,
    35166: False,
    114166: True,
    35114: False,
    9167: False,
    167168: True,
    92168: False,
    82174: True,
    30082: False,
    30078: False,
    78156: False,
    136175: False,
    115175: False,
    171172: True,
    173176: True,
    144176: False,
    102176: False,
    102173: False,
    106173: False,
    173179: True,
    144169: False,
    156169: True,
    169171: False,
    144171: False,
    184185: True,
    127184: False,
    172185: False,
    100116: True,
    51100: False,
    30079: False,
    140170: False,
    188189: False,
    139189: False,
    101116: False,
    101139: False,
    79123: False,
    123188: False,
    23102: False,
    23191: False,
    191192: True,
    164165: True,
    151164: True,
    151165: True
}
'''