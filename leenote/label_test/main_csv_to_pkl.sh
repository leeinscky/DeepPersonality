# run csv_to_pkl.py


# train 
python3 csv_to_pkl.py \
--csv_path '/Users/lizejian/cambridge/mphil_project/learn/udiva/DeepPersonality/lzjnote/label_test/sessions_train.csv' \
--output_path '/Users/lizejian/cambridge/mphil_project/learn/udiva/DeepPersonality/datasets/udiva_tiny/train/label/sessions_train.pkl'
# csv 
# 打印结果： id_known_dict: {'128129': 1, '127128': 1, '127129': 1, '055128': 1, '059134': 0, '118125': 1, '055125': 0, '058110': 0, '058059': 1, '110136': 1, '009106': 0, '092107': 0, '139140': 1, '015126': 1, '015133': 0, '112132': 0, '113132': 0, '112113': 1, '041083': 1, '042084': 1, '041084': 1, '042092': 0, '079142': 1, '097098': 1, '098126': 0, '097142': 0, '076122': 0, '122144': 0, '013041': 0, '005013': 0, '003005': 1, '002003': 1, '005134': 0, '020150': 1, '017150': 1, '020149': 0, '017149': 0, '051076': 1, '044156': 0, '043152': 0, '152153': 1, '006007': 1, '006153': 0, '020025': 0, '018025': 0, '018020': 1, '027076': 0, '027113': 0, '004115': 0, '004096': 0, '092096': 1, '106108': 0, '092108': 1, '076143': 1, '020090': 0, '040090': 1, '035040': 0, '034133': 0, '034035': 1, '025157': 1, '025044': 0, '044157': 0, '148151': 1, '043079': 0, '106148': 0, '043143': 0, '118154': 0, '027154': 0, '027118': 0, '017027': 0, '010011': 1, '010034': 0, '011034': 0, '034121': 0, '035166': 0, '114166': 1, '035114': 0, '009167': 0, '167168': 1, '092168': 0, '082174': 1, '030082': 0, '030078': 0, '078156': 0, '136175': 0, '115175': 0, '171172': 1, '173176': 1, '144176': 0, '102176': 0, '102173': 0, '106173': 0, '173179': 1, '144169': 0, '156169': 1, '169171': 0, '144171': 0, '184185': 1, '127184': 0, '172185': 0, '100116': 1, '051100': 0, '030079': 0, '140170': 0, '188189': 0, '139189': 0, '101116': 0, '101139': 0, '079123': 0, '123188': 0, '023102': 0, '023191': 0, '191192': 1, '164165': 1, '151164': 1, '151165': 1}

# validation
python3 csv_to_pkl.py \
--csv_path /Users/lizejian/cambridge/mphil_project/learn/udiva/DeepPersonality/datasets/udiva_tiny/val/metadata/metadata_val/sessions_val.csv \
--output_path /Users/lizejian/cambridge/mphil_project/learn/udiva/DeepPersonality/datasets/udiva_tiny/val/label/sessions_val.pkl
# csv id_known_dict: {1081: True, 1080: True, 80081: True, 119145: False, 119124: True, 124145: False, 146147: True, 52057: True, 38039: True, 180181: True, 141182: True, 182183: False, 181182: True, 141183: False, 85186: False, 183186: True, 119190: False, 85190: False}
# 打印结果：id_known_dict: {'001081': (1, 0), '001080': (1, 0), '080081': (1, 0), '119145': (0, 1), '119124': (1, 0), '124145': (0, 1), '146147': (1, 0), '052057': (1, 0), '038039': (1, 0), '180181': (1, 0), '141182': (1, 0), '182183': (0, 1), '181182': (1, 0), '141183': (0, 1), '085186': (0, 1), '183186': (1, 0), '119190': (0, 1), '085190': (0, 1)}

# test
python3 csv_to_pkl.py \
--csv_path /Users/lizejian/cambridge/mphil_project/learn/udiva/DeepPersonality/datasets/udiva_tiny/test/metadata/metadata_test/sessions_test.csv \
--output_path /Users/lizejian/cambridge/mphil_project/learn/udiva/DeepPersonality/datasets/udiva_tiny/test/label/sessions_test.pkl
# csv id_known_dict: {111130: True, 86087: True, 88089: True, 86089: True, 87088: True, 87089: True, 105117: True, 8105: True, 56109: False, 137138: False, 66067: True}
# 打印结果：id_known_dict: {'111130': (1, 0), '086087': (1, 0), '088089': (1, 0), '086089': (1, 0), '087088': (1, 0), '087089': (1, 0), '105117': (1, 0), '008105': (1, 0), '056109': (0, 1), '137138': (0, 1), '066067': (1, 0)}