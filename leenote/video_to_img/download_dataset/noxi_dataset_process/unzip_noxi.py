
import os 
import zipfile

""" 遍历/home/zl525/rds/hpc-work/datasets/noxi_full/img_zip 下的所有.zip结尾的文件，将其解压到/home/zl525/rds/hpc-work/datasets/noxi_full/img/目录下 """

zip_path = '/home/zl525/rds/hpc-work/datasets/noxi_full/img_zip'
img_path = '/home/zl525/rds/hpc-work/datasets/noxi_full/img/'

# 遍历/home/zl525/rds/hpc-work/datasets/noxi_full/img_zip 下的所有.zip结尾的文件，将其解压到/home/zl525/rds/hpc-work/datasets/noxi_full/img/目录下
# for file_name in os.listdir(zip_path):
#     if file_name.endswith('.zip'):
#         print('Processing: ' + file_name)
#         file_path = os.path.join(zip_path, file_name)
#         # 解压缩 .zip 文件
#         with zipfile.ZipFile(file_path, 'r') as zip_ref:
#             zip_ref.extractall(img_path)


# 遍历/home/zl525/rds/hpc-work/datasets/noxi_full/img/ 下的所有文件夹，如果文件夹名字含有下划线，则将文件夹重命名为第一个下划线左边的 session_id
for folder_name in sorted(os.listdir(img_path)):
    folder_path = os.path.join(img_path, folder_name)
    if os.path.isdir(folder_path):
        # 如果文件夹名字含有下划线，则将文件夹重命名为第一个下划线左边的 session_id
        if '_' in folder_name:
            new_folder_name = folder_name.split('_')[0]
            new_folder_path = os.path.join(img_path, new_folder_name)
            print('Renaming: ' + folder_path + ' to ' + new_folder_path)
            os.rename(folder_path, new_folder_path)




""" log
Renaming: /home/zl525/rds/hpc-work/datasets/noxi_full/img/001_2016-03-17_Paris to /home/zl525/rds/hpc-work/datasets/noxi_full/img/001
Renaming: /home/zl525/rds/hpc-work/datasets/noxi_full/img/002_2016-03-17_Paris to /home/zl525/rds/hpc-work/datasets/noxi_full/img/002
Renaming: /home/zl525/rds/hpc-work/datasets/noxi_full/img/003_2016-03-17_Paris to /home/zl525/rds/hpc-work/datasets/noxi_full/img/003
Renaming: /home/zl525/rds/hpc-work/datasets/noxi_full/img/004_2016-03-18_Paris to /home/zl525/rds/hpc-work/datasets/noxi_full/img/004
Renaming: /home/zl525/rds/hpc-work/datasets/noxi_full/img/005_2016-03-18_Paris to /home/zl525/rds/hpc-work/datasets/noxi_full/img/005
Renaming: /home/zl525/rds/hpc-work/datasets/noxi_full/img/006_2016-03-18_Paris to /home/zl525/rds/hpc-work/datasets/noxi_full/img/006
Renaming: /home/zl525/rds/hpc-work/datasets/noxi_full/img/007_2016-03-21_Paris to /home/zl525/rds/hpc-work/datasets/noxi_full/img/007
Renaming: /home/zl525/rds/hpc-work/datasets/noxi_full/img/008_2016-03-23_Paris to /home/zl525/rds/hpc-work/datasets/noxi_full/img/008
Renaming: /home/zl525/rds/hpc-work/datasets/noxi_full/img/009_2016-03-25_Paris to /home/zl525/rds/hpc-work/datasets/noxi_full/img/009
Renaming: /home/zl525/rds/hpc-work/datasets/noxi_full/img/010_2016-03-25_Paris to /home/zl525/rds/hpc-work/datasets/noxi_full/img/010
Renaming: /home/zl525/rds/hpc-work/datasets/noxi_full/img/011_2016-03-25_Paris to /home/zl525/rds/hpc-work/datasets/noxi_full/img/011
Renaming: /home/zl525/rds/hpc-work/datasets/noxi_full/img/012_2016-03-25_Paris to /home/zl525/rds/hpc-work/datasets/noxi_full/img/012
Renaming: /home/zl525/rds/hpc-work/datasets/noxi_full/img/013_2016-03-30_Paris to /home/zl525/rds/hpc-work/datasets/noxi_full/img/013
Renaming: /home/zl525/rds/hpc-work/datasets/noxi_full/img/014_2016-04-01_Paris to /home/zl525/rds/hpc-work/datasets/noxi_full/img/014
Renaming: /home/zl525/rds/hpc-work/datasets/noxi_full/img/015_2016-04-05_Paris to /home/zl525/rds/hpc-work/datasets/noxi_full/img/015
Renaming: /home/zl525/rds/hpc-work/datasets/noxi_full/img/016_2016-04-05_Paris to /home/zl525/rds/hpc-work/datasets/noxi_full/img/016
Renaming: /home/zl525/rds/hpc-work/datasets/noxi_full/img/017_2016-04-05_Paris to /home/zl525/rds/hpc-work/datasets/noxi_full/img/017
Renaming: /home/zl525/rds/hpc-work/datasets/noxi_full/img/018_2016-04-18_Paris to /home/zl525/rds/hpc-work/datasets/noxi_full/img/018
Renaming: /home/zl525/rds/hpc-work/datasets/noxi_full/img/019_2016-04-20_Paris to /home/zl525/rds/hpc-work/datasets/noxi_full/img/019
Renaming: /home/zl525/rds/hpc-work/datasets/noxi_full/img/020_2016-04-21_Paris to /home/zl525/rds/hpc-work/datasets/noxi_full/img/020
Renaming: /home/zl525/rds/hpc-work/datasets/noxi_full/img/021_2016-04-25_Paris to /home/zl525/rds/hpc-work/datasets/noxi_full/img/021
Renaming: /home/zl525/rds/hpc-work/datasets/noxi_full/img/022_2016-04-25_Paris to /home/zl525/rds/hpc-work/datasets/noxi_full/img/022
Renaming: /home/zl525/rds/hpc-work/datasets/noxi_full/img/023_2016-04-25_Paris to /home/zl525/rds/hpc-work/datasets/noxi_full/img/023
Renaming: /home/zl525/rds/hpc-work/datasets/noxi_full/img/024_2016-04-27_Paris to /home/zl525/rds/hpc-work/datasets/noxi_full/img/024
Renaming: /home/zl525/rds/hpc-work/datasets/noxi_full/img/025_2016-04-27_Paris to /home/zl525/rds/hpc-work/datasets/noxi_full/img/025
Renaming: /home/zl525/rds/hpc-work/datasets/noxi_full/img/026_2016-04-06_Nottingham to /home/zl525/rds/hpc-work/datasets/noxi_full/img/026
Renaming: /home/zl525/rds/hpc-work/datasets/noxi_full/img/027_2016-04-06_Nottingham to /home/zl525/rds/hpc-work/datasets/noxi_full/img/027
Renaming: /home/zl525/rds/hpc-work/datasets/noxi_full/img/028_2016-04-06_Nottingham to /home/zl525/rds/hpc-work/datasets/noxi_full/img/028
Renaming: /home/zl525/rds/hpc-work/datasets/noxi_full/img/029_2016-04-06_Nottingham to /home/zl525/rds/hpc-work/datasets/noxi_full/img/029
Renaming: /home/zl525/rds/hpc-work/datasets/noxi_full/img/030_2016-04-06_Nottingham to /home/zl525/rds/hpc-work/datasets/noxi_full/img/030
Renaming: /home/zl525/rds/hpc-work/datasets/noxi_full/img/031_2016-04-06_Nottingham to /home/zl525/rds/hpc-work/datasets/noxi_full/img/031
Renaming: /home/zl525/rds/hpc-work/datasets/noxi_full/img/032_2016-04-07_Nottingham to /home/zl525/rds/hpc-work/datasets/noxi_full/img/032
Renaming: /home/zl525/rds/hpc-work/datasets/noxi_full/img/033_2016-04-07_Nottingham to /home/zl525/rds/hpc-work/datasets/noxi_full/img/033
Renaming: /home/zl525/rds/hpc-work/datasets/noxi_full/img/034_2016-04-07_Nottingham to /home/zl525/rds/hpc-work/datasets/noxi_full/img/034
Renaming: /home/zl525/rds/hpc-work/datasets/noxi_full/img/035_2016-04-07_Nottingham to /home/zl525/rds/hpc-work/datasets/noxi_full/img/035
Renaming: /home/zl525/rds/hpc-work/datasets/noxi_full/img/036_2016-04-07_Nottingham to /home/zl525/rds/hpc-work/datasets/noxi_full/img/036
Renaming: /home/zl525/rds/hpc-work/datasets/noxi_full/img/037_2016-04-07_Nottingham to /home/zl525/rds/hpc-work/datasets/noxi_full/img/037
Renaming: /home/zl525/rds/hpc-work/datasets/noxi_full/img/038_2016-04-07_Nottingham to /home/zl525/rds/hpc-work/datasets/noxi_full/img/038
Renaming: /home/zl525/rds/hpc-work/datasets/noxi_full/img/039_2016-04-07_Nottingham to /home/zl525/rds/hpc-work/datasets/noxi_full/img/039
Renaming: /home/zl525/rds/hpc-work/datasets/noxi_full/img/040_2016-04-07_Nottingham to /home/zl525/rds/hpc-work/datasets/noxi_full/img/040
Renaming: /home/zl525/rds/hpc-work/datasets/noxi_full/img/041_2016-04-07_Nottingham to /home/zl525/rds/hpc-work/datasets/noxi_full/img/041
Renaming: /home/zl525/rds/hpc-work/datasets/noxi_full/img/042_2016-04-07_Nottingham to /home/zl525/rds/hpc-work/datasets/noxi_full/img/042
Renaming: /home/zl525/rds/hpc-work/datasets/noxi_full/img/043_2016-04-07_Nottingham to /home/zl525/rds/hpc-work/datasets/noxi_full/img/043
Renaming: /home/zl525/rds/hpc-work/datasets/noxi_full/img/044_2016-04-07_Nottingham to /home/zl525/rds/hpc-work/datasets/noxi_full/img/044
Renaming: /home/zl525/rds/hpc-work/datasets/noxi_full/img/045_2016-04-07_Nottingham to /home/zl525/rds/hpc-work/datasets/noxi_full/img/045
Renaming: /home/zl525/rds/hpc-work/datasets/noxi_full/img/046_2016-04-07_Nottingham to /home/zl525/rds/hpc-work/datasets/noxi_full/img/046
Renaming: /home/zl525/rds/hpc-work/datasets/noxi_full/img/047_2016-04-07_Nottingham to /home/zl525/rds/hpc-work/datasets/noxi_full/img/047
Renaming: /home/zl525/rds/hpc-work/datasets/noxi_full/img/048_2016-04-12_Nottingham to /home/zl525/rds/hpc-work/datasets/noxi_full/img/048
Renaming: /home/zl525/rds/hpc-work/datasets/noxi_full/img/049_2016-04-12_Nottingham to /home/zl525/rds/hpc-work/datasets/noxi_full/img/049
Renaming: /home/zl525/rds/hpc-work/datasets/noxi_full/img/050_2016-04-12_Nottingham to /home/zl525/rds/hpc-work/datasets/noxi_full/img/050
Renaming: /home/zl525/rds/hpc-work/datasets/noxi_full/img/051_2016-04-12_Nottingham to /home/zl525/rds/hpc-work/datasets/noxi_full/img/051
Renaming: /home/zl525/rds/hpc-work/datasets/noxi_full/img/052_2016-04-12_Nottingham to /home/zl525/rds/hpc-work/datasets/noxi_full/img/052
Renaming: /home/zl525/rds/hpc-work/datasets/noxi_full/img/053_2016-04-12_Nottingham to /home/zl525/rds/hpc-work/datasets/noxi_full/img/053
Renaming: /home/zl525/rds/hpc-work/datasets/noxi_full/img/054_2016-04-12_Nottingham to /home/zl525/rds/hpc-work/datasets/noxi_full/img/054
Renaming: /home/zl525/rds/hpc-work/datasets/noxi_full/img/055_2016-04-14_Nottingham to /home/zl525/rds/hpc-work/datasets/noxi_full/img/055
Renaming: /home/zl525/rds/hpc-work/datasets/noxi_full/img/056_2016-04-14_Nottingham to /home/zl525/rds/hpc-work/datasets/noxi_full/img/056
Renaming: /home/zl525/rds/hpc-work/datasets/noxi_full/img/057_2016-04-14_Nottingham to /home/zl525/rds/hpc-work/datasets/noxi_full/img/057
Renaming: /home/zl525/rds/hpc-work/datasets/noxi_full/img/058_2016-04-14_Nottingham to /home/zl525/rds/hpc-work/datasets/noxi_full/img/058
Renaming: /home/zl525/rds/hpc-work/datasets/noxi_full/img/059_2016-04-14_Nottingham to /home/zl525/rds/hpc-work/datasets/noxi_full/img/059
Renaming: /home/zl525/rds/hpc-work/datasets/noxi_full/img/060_2016-04-14_Nottingham to /home/zl525/rds/hpc-work/datasets/noxi_full/img/060
Renaming: /home/zl525/rds/hpc-work/datasets/noxi_full/img/061_2016-04-14_Nottingham to /home/zl525/rds/hpc-work/datasets/noxi_full/img/061
Renaming: /home/zl525/rds/hpc-work/datasets/noxi_full/img/062_2016-04-14_Nottingham to /home/zl525/rds/hpc-work/datasets/noxi_full/img/062
Renaming: /home/zl525/rds/hpc-work/datasets/noxi_full/img/063_2016-04-14_Nottingham to /home/zl525/rds/hpc-work/datasets/noxi_full/img/063
Renaming: /home/zl525/rds/hpc-work/datasets/noxi_full/img/064_2016-04-14_Nottingham to /home/zl525/rds/hpc-work/datasets/noxi_full/img/064
Renaming: /home/zl525/rds/hpc-work/datasets/noxi_full/img/065_2016-04-14_Nottingham to /home/zl525/rds/hpc-work/datasets/noxi_full/img/065
Renaming: /home/zl525/rds/hpc-work/datasets/noxi_full/img/066_2016-05-23_Augsburg to /home/zl525/rds/hpc-work/datasets/noxi_full/img/066
Renaming: /home/zl525/rds/hpc-work/datasets/noxi_full/img/067_2016-05-23_Augsburg to /home/zl525/rds/hpc-work/datasets/noxi_full/img/067
Renaming: /home/zl525/rds/hpc-work/datasets/noxi_full/img/068_2016-05-23_Augsburg to /home/zl525/rds/hpc-work/datasets/noxi_full/img/068
Renaming: /home/zl525/rds/hpc-work/datasets/noxi_full/img/069_2016-05-23_Augsburg to /home/zl525/rds/hpc-work/datasets/noxi_full/img/069
Renaming: /home/zl525/rds/hpc-work/datasets/noxi_full/img/070_2016-05-23_Augsburg to /home/zl525/rds/hpc-work/datasets/noxi_full/img/070
Renaming: /home/zl525/rds/hpc-work/datasets/noxi_full/img/071_2016-05-23_Augsburg to /home/zl525/rds/hpc-work/datasets/noxi_full/img/071
Renaming: /home/zl525/rds/hpc-work/datasets/noxi_full/img/072_2016-05-23_Augsburg to /home/zl525/rds/hpc-work/datasets/noxi_full/img/072
Renaming: /home/zl525/rds/hpc-work/datasets/noxi_full/img/073_2016-05-23_Augsburg to /home/zl525/rds/hpc-work/datasets/noxi_full/img/073
Renaming: /home/zl525/rds/hpc-work/datasets/noxi_full/img/074_2016-05-23_Augsburg to /home/zl525/rds/hpc-work/datasets/noxi_full/img/074
Renaming: /home/zl525/rds/hpc-work/datasets/noxi_full/img/075_2016-05-23_Augsburg to /home/zl525/rds/hpc-work/datasets/noxi_full/img/075
Renaming: /home/zl525/rds/hpc-work/datasets/noxi_full/img/076_2016-05-24_Augsburg to /home/zl525/rds/hpc-work/datasets/noxi_full/img/076
Renaming: /home/zl525/rds/hpc-work/datasets/noxi_full/img/077_2016-05-24_Augsburg to /home/zl525/rds/hpc-work/datasets/noxi_full/img/077
Renaming: /home/zl525/rds/hpc-work/datasets/noxi_full/img/078_2016-05-24_Augsburg to /home/zl525/rds/hpc-work/datasets/noxi_full/img/078
Renaming: /home/zl525/rds/hpc-work/datasets/noxi_full/img/079_2016-05-24_Augsburg to /home/zl525/rds/hpc-work/datasets/noxi_full/img/079
Renaming: /home/zl525/rds/hpc-work/datasets/noxi_full/img/080_2016-05-24_Augsburg to /home/zl525/rds/hpc-work/datasets/noxi_full/img/080
Renaming: /home/zl525/rds/hpc-work/datasets/noxi_full/img/081_2016-05-24_Augsburg to /home/zl525/rds/hpc-work/datasets/noxi_full/img/081
Renaming: /home/zl525/rds/hpc-work/datasets/noxi_full/img/082_2016-05-25_Augsburg to /home/zl525/rds/hpc-work/datasets/noxi_full/img/082
Renaming: /home/zl525/rds/hpc-work/datasets/noxi_full/img/083_2016-05-25_Augsburg to /home/zl525/rds/hpc-work/datasets/noxi_full/img/083
Renaming: /home/zl525/rds/hpc-work/datasets/noxi_full/img/084_2016-05-31_Augsburg to /home/zl525/rds/hpc-work/datasets/noxi_full/img/084

"""