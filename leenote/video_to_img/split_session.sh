# /home/zl525/rds/hpc-work/datasets/udiva_full/train/recordings/talk_recordings_train_img 目录下有115个文件夹, 文件夹名称是6位数字
# 使用shell脚本将这些115个文件夹分成三个part并将三个part的文件夹move到part1/, part2/, part3/目录下, 
# 即: 第1和2个part都有38个文件夹，第3个part有39个文件夹, 即: 1-38个在part1/, 39-76个在part2/, 77-115个在part3/
# 当前目录下的所有文件夹一共有115个，为002003到191192文件夹. 1-38个:002003-034133, 39-76个:035040-102176, 77-115个:106108-191192

# 设置目录路径和文件夹名称前缀
directory="/home/zl525/rds/hpc-work/datasets/udiva_full/train/recordings/talk_recordings_train_img/"
# directory="/home/zl525/rds/hpc-work/datasets/udiva_temp_test"
prefix=""
cd $directory

# 创建part1、part2和part3目录（如果不存在）
mkdir -p part1
mkdir -p part2
mkdir -p part3

# 目前，当前目录下一共115+3个文件夹, 因此分割是: 1-38, 39-76, 77-115


# 正式移动前先测试一下
# echo "***************** test *****************"
# for folder in $(ls "$directory" | head -n 2); do
#     echo $folder
#     mv "$directory/$folder" part1/
# done

# 第一步: 移动part1的文件夹, 此时要注释掉后面的两个part，否则会重复移动
# echo "***************** part1: *****************"
# # echo $(ls "$directory" | head -n 38)
# for folder in $(ls "$directory" | head -n 38); do # head -n 38: 取前38个
#     echo $folder
#     mv "$directory/$folder" part1/
# done

# 第二步: 移动part2的文件夹, 此时要注释掉前面的part1以及后面的part3，否则会重复移动
# echo "***************** part2: *****************"
# # for folder in $(ls "$directory" | head -n 76 | tail -n 38); do # head -n 76: 取前76个; 然后在前76个中选择后38个 tail -n 38
# for folder in $(ls "$directory" | head -n 38); do # 因为上一步已经移动了前38个，所以这里只需要移动剩余所有文件的前38个
#     echo $folder
#     mv "$directory/$folder" part2/
# done


# 第三步: 移动part3的文件夹, 此时要注释掉前面的part1和part2，否则会重复移动
echo "***************** part3: *****************"
# 取第77-115个文件夹 mv 到part3/
# for folder in $(ls "$directory" | head -n 115 | tail -n 39); do # head -n 115: 取前115个; 然后在前115个中选择后39个 tail -n 39
for folder in $(ls "$directory" | head -n 39); do # 因为前两步已经移动了前76个，所以这里只需要移动剩余所有文件的前39个
    echo $folder
    mv "$directory/$folder" part3/
done
