# Udiva测试命令
    conda activate DeepPersonality
    cd /Users/lizejian/cambridge/mphil_project/learn/udiva/DeepPersonality
    find . -name ".DS_Store" -delete
    python ./script/run_exp.py --cfg_file ./config/demo/bimodal_resnet18_udiva.yaml

For quick start with tiny ChaLearn 2016 dataset, if you prepare the data by the instructions in above section, the following command will launch an experiment for bimodal-resnet18 model.
    命令： 
        cd /Users/lizejian/cambridge/mphil_project/learn/udiva/DeepPersonality & python ./script/run_exp.py --cfg_file ./config/demo/bimodal_resnet18.yaml
    
    结果： results/demo/unified_frame_images/03_bimodal_resnet/11-22_22-33
        (DeepPersonality)  ⚙ lizejian@lizejiandeMacBook-Pro-3  ~/cambridge/mphil_project/learn/udiva/DeepPersonality   main  python ./script/run_exp.py --cfg_file ./config/demo/bimodal_resnet18.yaml
        {
            "DATA":{
                "ROOT":"datasets",
                "SESSION":"talk",
                "TEST_AUD_DATA":"ChaLearn2016_tiny/voice_data/voice_librosa/test_data",
                "TEST_IMG_DATA":"ChaLearn2016_tiny/test_data",
                "TEST_IMG_FACE_DATA":"image_data/test_data_face",
                "TEST_LABEL_DATA":"ChaLearn2016_tiny/annotation/annotation_test.pkl",
                "TRAIN_AUD_DATA":"ChaLearn2016_tiny/voice_data/voice_librosa/train_data",
                "TRAIN_IMG_DATA":"ChaLearn2016_tiny/train_data",
                "TRAIN_IMG_FACE_DATA":"image_data/train_data_face",
                "TRAIN_LABEL_DATA":"ChaLearn2016_tiny/annotation/annotation_training.pkl",
                "TYPE":"frame",
                "VALID_AUD_DATA":"ChaLearn2016_tiny/voice_data/voice_librosa/valid_data",
                "VALID_IMG_DATA":"ChaLearn2016_tiny/valid_data",
                "VALID_IMG_FACE_DATA":"image_data/valid_data_face",
                "VALID_LABEL_DATA":"ChaLearn2016_tiny/annotation/annotation_validation.pkl",
                "VA_DATA":"va_data/cropped_aligned",
                "VA_ROOT":"datasets",
                "VA_TRAIN_LABEL":"va_data/va_label/VA_Set/Train_Set",
                "VA_VALID_LABEL":"va_data/va_label/VA_Set/Validation_Set"
            },
            "DATA_LOADER":{
                "DATASET":"",
                "DROP_LAST":true,
                "NAME":"bimodal_resnet_data_loader",
                "NUM_WORKERS":0,
                "SECOND_STAGE":{
                    "METHOD":"",
                    "TYPE":""
                },
                "SHUFFLE":true,
                "TRAIN_BATCH_SIZE":8,
                "TRANSFORM":"standard_frame_transform",
                "VALID_BATCH_SIZE":4
            },
            "LOSS":{
                "NAME":"mean_square_error"
            },
            "MODEL":{
                "NAME":"audiovisual_resnet",
                "NUM_CLASS":5,
                "PRETRAIN":false,
                "RETURN_FEATURE":false,
                "SPECTRUM_CHANNEL":50
            },
            "SOLVER":{
                "BETA_1":0.5,
                "BETA_2":0.999,
                "FACTOR":0.1,
                "LR_INIT":0.001,
                "MILESTONE":[
                    100,
                    200
                ],
                "MOMENTUM":0.9,
                "NAME":"sgd",
                "RESET_LR":false,
                "SCHEDULER":"multi_step_scale",
                "WEIGHT_DECAY":0.0005
            },
            "TEST":{
                "COMPUTE_CCC":true,
                "COMPUTE_PCC":true,
                "FULL_TEST":false,
                "SAVE_DATASET_OUTPUT":"",
                "TEST_ONLY":false,
                "WEIGHT":""
            },
            "TRAIN":{
                "LOG_INTERVAL":10,
                "MAX_EPOCH":30,
                "OUTPUT_DIR":"results/demo/unified_frame_images/03_bimodal_resnet",
                "PRE_TRAINED_MODEL":null,
                "RESUME":"",
                "START_EPOCH":0,
                "TRAINER":"BiModalTrainer",
                "VALID_INTERVAL":1
            }
        }
        Training: learning rate:0.001
        Valid: Epoch[001/030] Train Mean_Acc: 86.46% Valid Mean_Acc:89.20% OCEAN_ACC:[0.8868338  0.891785   0.88287103 0.90980685 0.8885487 ]

        Training: learning rate:0.001
        Valid: Epoch[002/030] Train Mean_Acc: 86.59% Valid Mean_Acc:89.04% OCEAN_ACC:[0.8893837  0.88626516 0.88231087 0.90603507 0.8882473 ]

        Training: learning rate:0.001
        Valid: Epoch[003/030] Train Mean_Acc: 87.33% Valid Mean_Acc:88.97% OCEAN_ACC:[0.8896352  0.88255596 0.88472337 0.90468514 0.887091  ]

        Training: learning rate:0.001
        Valid: Epoch[004/030] Train Mean_Acc: 87.72% Valid Mean_Acc:89.20% OCEAN_ACC:[0.8939072  0.88746786 0.88687265 0.90622425 0.885501  ]

        Training: learning rate:0.001
        Valid: Epoch[005/030] Train Mean_Acc: 87.61% Valid Mean_Acc:89.16% OCEAN_ACC:[0.89352685 0.88737774 0.8868763  0.9067521  0.8836449 ]

        Training: learning rate:0.001
        Valid: Epoch[006/030] Train Mean_Acc: 87.28% Valid Mean_Acc:89.12% OCEAN_ACC:[0.8959181  0.8846132  0.88571036 0.90734637 0.8823732 ]

        Training: learning rate:0.001
        Valid: Epoch[007/030] Train Mean_Acc: 87.75% Valid Mean_Acc:89.23% OCEAN_ACC:[0.8969954  0.88863057 0.88488865 0.90832293 0.88247174]

        Training: learning rate:0.001
        Valid: Epoch[008/030] Train Mean_Acc: 87.73% Valid Mean_Acc:89.21% OCEAN_ACC:[0.8947004  0.8897166  0.8868507  0.90559655 0.88354146]

        Training: learning rate:0.001
        Valid: Epoch[009/030] Train Mean_Acc: 88.07% Valid Mean_Acc:88.62% OCEAN_ACC:[0.88357604 0.88325197 0.8859577  0.8960439  0.88219947]

        Training: learning rate:0.001
        Valid: Epoch[010/030] Train Mean_Acc: 87.32% Valid Mean_Acc:88.98% OCEAN_ACC:[0.8925648  0.8886507  0.8857048  0.90001136 0.88215446]

        Training: learning rate:0.001
        Valid: Epoch[011/030] Train Mean_Acc: 87.72% Valid Mean_Acc:88.79% OCEAN_ACC:[0.8886455  0.89030266 0.88179266 0.8987888  0.8801845 ]

        Training: learning rate:0.001
        Valid: Epoch[012/030] Train Mean_Acc: 87.88% Valid Mean_Acc:88.72% OCEAN_ACC:[0.89158994 0.8873402  0.88169414 0.8952511  0.8802937 ]

        Training: learning rate:0.001
        Valid: Epoch[013/030] Train Mean_Acc: 87.86% Valid Mean_Acc:88.55% OCEAN_ACC:[0.88999856 0.88671845 0.8812276  0.89163303 0.8778639 ]

        Training: learning rate:0.001
        Valid: Epoch[014/030] Train Mean_Acc: 87.55% Valid Mean_Acc:88.58% OCEAN_ACC:[0.8897411 0.8859598 0.8816428 0.8930251 0.8786546]

        Training: learning rate:0.001
        Valid: Epoch[015/030] Train Mean_Acc: 87.61% Valid Mean_Acc:88.63% OCEAN_ACC:[0.8886882  0.8891287  0.882696   0.89196587 0.87901556]

        Training: learning rate:0.001
        Valid: Epoch[016/030] Train Mean_Acc: 87.89% Valid Mean_Acc:88.45% OCEAN_ACC:[0.88697517 0.88705665 0.8823989  0.890396   0.8758532 ]

        Training: learning rate:0.001
        Valid: Epoch[017/030] Train Mean_Acc: 87.99% Valid Mean_Acc:88.45% OCEAN_ACC:[0.8873121  0.8857298  0.8809088  0.89094067 0.87741244]

        Training: learning rate:0.001
        Valid: Epoch[018/030] Train Mean_Acc: 87.92% Valid Mean_Acc:88.41% OCEAN_ACC:[0.8861343  0.88842475 0.88031805 0.8898498  0.8756164 ]

        Training: learning rate:0.001
        Valid: Epoch[019/030] Train Mean_Acc: 87.93% Valid Mean_Acc:87.81% OCEAN_ACC:[0.8753214  0.88269293 0.87683713 0.88121474 0.87434846]

        Training: learning rate:0.001
        Valid: Epoch[020/030] Train Mean_Acc: 88.06% Valid Mean_Acc:88.26% OCEAN_ACC:[0.8877201  0.88457644 0.8780818  0.8864616  0.8762676 ]

        Training: learning rate:0.001
        Valid: Epoch[021/030] Train Mean_Acc: 87.78% Valid Mean_Acc:88.32% OCEAN_ACC:[0.8871659  0.8849638  0.88034534 0.8880669  0.87536687]

        Training: learning rate:0.001
        Valid: Epoch[022/030] Train Mean_Acc: 87.81% Valid Mean_Acc:88.42% OCEAN_ACC:[0.88782656 0.8861458  0.87905633 0.8910917  0.8767727 ]

        Training: learning rate:0.001
        Valid: Epoch[023/030] Train Mean_Acc: 87.80% Valid Mean_Acc:88.24% OCEAN_ACC:[0.88823444 0.88307655 0.8764304  0.8901783  0.87426454]

        Training: learning rate:0.001
        Valid: Epoch[024/030] Train Mean_Acc: 88.13% Valid Mean_Acc:88.16% OCEAN_ACC:[0.88657796 0.8828497  0.8788659  0.88600385 0.87378633]

        Training: learning rate:0.001
        Valid: Epoch[025/030] Train Mean_Acc: 87.65% Valid Mean_Acc:88.33% OCEAN_ACC:[0.8884166  0.8856441  0.8795376  0.88868856 0.8740834 ]

        Training: learning rate:0.001
        Valid: Epoch[026/030] Train Mean_Acc: 87.74% Valid Mean_Acc:87.71% OCEAN_ACC:[0.8731286  0.88361037 0.8743688  0.88273525 0.87168133]

        Training: learning rate:0.001
        Valid: Epoch[027/030] Train Mean_Acc: 87.92% Valid Mean_Acc:88.13% OCEAN_ACC:[0.88596934 0.88158834 0.87974226 0.8877676  0.8712848 ]

        Training: learning rate:0.001
        Valid: Epoch[028/030] Train Mean_Acc: 88.03% Valid Mean_Acc:88.27% OCEAN_ACC:[0.8886987 0.8836231 0.8795721 0.8884233 0.8733591]

        Training: learning rate:0.001
        Valid: Epoch[029/030] Train Mean_Acc: 88.23% Valid Mean_Acc:87.78% OCEAN_ACC:[0.8769518  0.88273287 0.8762001  0.8813925  0.8717836 ]

        Training: learning rate:0.001
        Valid: Epoch[030/030] Train Mean_Acc: 88.02% Valid Mean_Acc:88.17% OCEAN_ACC:[0.88736886 0.8821956  0.8786753  0.88730323 0.872865  ]

        11-22_22-40 done, best acc: 0.892261803150177 in :6
        Test only mode
        test with model results/demo/unified_frame_images/03_bimodal_resnet/11-22_22-33/checkpoint_6.pkl
        40%|██████████████████████████████████████████████████████████████████████▊                                                                                                          | 2/5 [00:00<00:00,  3.12it/s]
        Traceback (most recent call last):
        File "./script/run_exp.py", line 49, in <module>
            main()
        File "./script/run_exp.py", line 42, in main
            runner.run()
        File "/Users/lizejian/cambridge/mphil_project/learn/udiva/DeepPersonality/script/../dpcv/experiment/exp_runner.py", line 154, in run
            self.test()
        File "/Users/lizejian/cambridge/mphil_project/learn/udiva/DeepPersonality/script/../dpcv/experiment/exp_runner.py", line 128, in test
            ocean_acc_avg, ocean_acc, dataset_output, dataset_label, mse = self.trainer.test(
        File "/Users/lizejian/cambridge/mphil_project/learn/udiva/DeepPersonality/script/../dpcv/engine/bi_modal_trainer.py", line 115, in test
            for data in tqdm(data_loader):
        File "/Users/lizejian/opt/anaconda3/envs/DeepPersonality/lib/python3.8/site-packages/tqdm/std.py", line 1195, in __iter__
            for obj in iterable:
        File "/Users/lizejian/opt/anaconda3/envs/DeepPersonality/lib/python3.8/site-packages/torch/utils/data/dataloader.py", line 521, in __next__
            data = self._next_data()
        File "/Users/lizejian/opt/anaconda3/envs/DeepPersonality/lib/python3.8/site-packages/torch/utils/data/dataloader.py", line 561, in _next_data
            data = self._dataset_fetcher.fetch(index)  # may raise StopIteration
        File "/Users/lizejian/opt/anaconda3/envs/DeepPersonality/lib/python3.8/site-packages/torch/utils/data/_utils/fetch.py", line 44, in fetch
            data = [self.dataset[idx] for idx in possibly_batched_index]
        File "/Users/lizejian/opt/anaconda3/envs/DeepPersonality/lib/python3.8/site-packages/torch/utils/data/_utils/fetch.py", line 44, in <listcomp>
            data = [self.dataset[idx] for idx in possibly_batched_index]
        File "/Users/lizejian/cambridge/mphil_project/learn/udiva/DeepPersonality/script/../dpcv/data/datasets/audio_visual_data.py", line 25, in __getitem__
            label = self.get_ocean_label(idx)
        File "/Users/lizejian/cambridge/mphil_project/learn/udiva/DeepPersonality/script/../dpcv/data/datasets/bi_modal_data.py", line 66, in get_ocean_label
            self.annotation["openness"][video_name],
        KeyError: '.DS_Store.mp4'

    test阶段中断的解决办法： find . -name ".DS_Store" -delete  https://github.com/fastai/fastai/issues/488

img_dir_ls 打印结果：
    [DeepPersonality/dpcv/data/datasets/bi_modal_data.py] self.img_dir_ls:  
    [
        'datasets/ChaLearn2016_tiny/train_data/-AmMDnVl4s8.003', 
        'datasets/ChaLearn2016_tiny/train_data/2kqPuht5jTg.002', 
        'datasets/ChaLearn2016_tiny/train_data/4CSV8L7aVik.000', 
        'datasets/ChaLearn2016_tiny/train_data/50gokPvvMs8.000', 
        'datasets/ChaLearn2016_tiny/train_data/6KKNrufnL80.000', 
        'datasets/ChaLearn2016_tiny/train_data/83cmR2fkyy8.005', 
        'datasets/ChaLearn2016_tiny/train_data/98fnGDVky00.005', 
        'datasets/ChaLearn2016_tiny/train_data/9KAqOrdiZ4I.002', 
        'datasets/ChaLearn2016_tiny/train_data/9hqH1PJ6cG8.001', 
        'datasets/ChaLearn2016_tiny/train_data/A3StIKMjn4k.002', 
        'datasets/ChaLearn2016_tiny/train_data/BWAEpai6FK0.003', 
        'datasets/ChaLearn2016_tiny/train_data/C_NtwmmF2Ys.000', 
        'datasets/ChaLearn2016_tiny/train_data/DnTtbAR_Qyw.004', 
        'datasets/ChaLearn2016_tiny/train_data/F0_EI_X5JVk.003', 
        'datasets/ChaLearn2016_tiny/train_data/HegkSmkiBos.005', 
        'datasets/ChaLearn2016_tiny/train_data/JBdLI6AhJrw.000', 
        'datasets/ChaLearn2016_tiny/train_data/JIYZTruMpiI.003', 
        'datasets/ChaLearn2016_tiny/train_data/JiXJeI5_jGM.000', 
        'datasets/ChaLearn2016_tiny/train_data/KJ643kfjqLY.003', 
        'datasets/ChaLearn2016_tiny/train_data/L9sG80PI1Gw.003', 
        'datasets/ChaLearn2016_tiny/train_data/L_gmlaz-0s4.003', 
        'datasets/ChaLearn2016_tiny/train_data/MOXPVzRBDPo.002', 
        'datasets/ChaLearn2016_tiny/train_data/NDBCrVvp0Vg.003', 
        'datasets/ChaLearn2016_tiny/train_data/OWZ-qVZG14A.002', 
        'datasets/ChaLearn2016_tiny/train_data/Q2AI4XpApFs.002', 
        'datasets/ChaLearn2016_tiny/train_data/Qz_cjgCtDcM.003', 
        'datasets/ChaLearn2016_tiny/train_data/RlUuWWWFrhM.005', 
        'datasets/ChaLearn2016_tiny/train_data/T6CMGXdPUTA.001', 
        'datasets/ChaLearn2016_tiny/train_data/TPk6KiHuPag.004', 
        'datasets/ChaLearn2016_tiny/train_data/Tr3A7WODEuM.001', 
        'datasets/ChaLearn2016_tiny/train_data/Uu-NbXUPr-A.001', 
        'datasets/ChaLearn2016_tiny/train_data/W0FCCk0a0tg.001', 
        'datasets/ChaLearn2016_tiny/train_data/WT1YjeADatU.001', 
        'datasets/ChaLearn2016_tiny/train_data/Yj36y7ELRZE.004', 
        'datasets/ChaLearn2016_tiny/train_data/_uNup91ZYw0.002', 
        'datasets/ChaLearn2016_tiny/train_data/bt-ev53zZWE.004', 
        'datasets/ChaLearn2016_tiny/train_data/dd0z9mErfSo.003', 
        'datasets/ChaLearn2016_tiny/train_data/eD4b8sM-Tpw.000', 
        'datasets/ChaLearn2016_tiny/train_data/eI_7SimPnnQ.001', 
        'datasets/ChaLearn2016_tiny/train_data/geXpIfaFzF4.001', 
        'datasets/ChaLearn2016_tiny/train_data/in-HuMgiDCE.001', 
        'datasets/ChaLearn2016_tiny/train_data/jDdRrqRcSzM.002', 
        'datasets/ChaLearn2016_tiny/train_data/jTkEWnuDnbA.001', 
        'datasets/ChaLearn2016_tiny/train_data/jwcSbw4NDn0.005', 
        'datasets/ChaLearn2016_tiny/train_data/myhEW1aZRg4.000', 
        'datasets/ChaLearn2016_tiny/train_data/n8IiQJyqjiE.003', 
        'datasets/ChaLearn2016_tiny/train_data/nGGtTu6dSJE.000', 
        'datasets/ChaLearn2016_tiny/train_data/nOFHZ_s7Et4.005', 
        'datasets/ChaLearn2016_tiny/train_data/nZz1hK90gwA.004', 
        'datasets/ChaLearn2016_tiny/train_data/okSmKH2k5lE.002', 
        'datasets/ChaLearn2016_tiny/train_data/om-9kFEKJIs.004', 
        'datasets/ChaLearn2016_tiny/train_data/opEoJBrcmbI.002', 
        'datasets/ChaLearn2016_tiny/train_data/vMtF0akNUK4.000', 
        'datasets/ChaLearn2016_tiny/train_data/vr5FWHUkYRM.001', 
        'datasets/ChaLearn2016_tiny/train_data/vrMlwwTLWIE.005', 
        'datasets/ChaLearn2016_tiny/train_data/wTo1uZns2X8.000', 
        'datasets/ChaLearn2016_tiny/train_data/x0CZuHnJ0Hs.005', 
        'datasets/ChaLearn2016_tiny/train_data/yOzHZOg95Ug.003', 
        'datasets/ChaLearn2016_tiny/train_data/yOzHZOg95Ug.005', 
        'datasets/ChaLearn2016_tiny/train_data/yftfxiDNXko.002'
    ]