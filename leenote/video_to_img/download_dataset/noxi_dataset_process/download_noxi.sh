# see https://tmp.link


# download dataset from https://tmp.link by wget
nohup wget -O  "19-30.tar.gz" https://tmp-hd7.vx-cdn.com/file-6418d980e9c43-64198959419c5/19-30.tar.gz >19-30.log 2>&1 &
nohup wget -O  "31-40.tar.gz" https://tmp-hd4.vx-cdn.com/file-6418f82c49d04-6419896abe220/31-40.tar.gz >31-40.log 2>&1 &
nohup wget -O  "41-50.tar.gz" https://tmp-hd7.vx-cdn.com/file-6419189931f55-6419897760fca/41-50.tar.gz >41-50.log 2>&1 &
nohup wget -O  "51-60.tar.gz" https://tmp-hd8.vx-cdn.com/file-64197ffe1ad0e-641989a77935c/51-60.tar.gz >51-60.log 2>&1 &
nohup wget -O  "61-70.tar.gz" https://tmp-hd8.vx-cdn.com/file-641972f8ce181-641989b846570/61-70.tar.gz >61-70.log 2>&1 &
nohup wget -O  "71-80.tar.gz" https://tmp-azeroth.vx-cdn.com/file-6419876920771-641989c559714/71-80.tar.gz >71-80.log 2>&1 &
nohup wget -O  "81-84.tar.gz" https://tmp-azeroth.vx-cdn.com/file-641988ab94050-641989d230843/81-84.tar.gz >81-84.log 2>&1 &
# ps -ef | grep wget | grep -v grep

# 5 8 11 14
wget -O  "5_8_11_14.tar.gz" https://tmp-azeroth.vx-cdn.com/file-6419932b67888-641993e628761/5_8_11_14.tar.gz


# tar dataset
tar -zxvf 19-30.tar.gz
nohup tar -zxvf 31-40.tar.gz >31-40.log 2>&1 &
nohup tar -zxvf 41-50.tar.gz >41-50.log 2>&1 &
nohup tar -zxvf 51-60.tar.gz >51-60.log 2>&1 &
nohup tar -zxvf 61-70.tar.gz >61-70.log 2>&1 &
nohup tar -zxvf 71-80.tar.gz >71-80.log 2>&1 &
nohup tar -zxvf 81-84.tar.gz >81-84.log 2>&1 &
nohup tar -zxvf 5_8_11_14.tar.gz >5_8_11_14.log 2>&1 &

# 将001.zip解压到 /home/zl525/rds/hpc-work/datasets/noxi_full/img/ 下, 生成 /home/zl525/rds/hpc-work/datasets/noxi_full/img/001/ 目录
# unzip 001.zip -d /home/zl525/rds/hpc-work/datasets/noxi_full/img/