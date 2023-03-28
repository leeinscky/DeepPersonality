# [python可视化hdf5文件\_工科pai的博客-CSDN博客\_hdf5可视化](https://blog.csdn.net/weixin_45653050/article/details/111410478)
# 执行命令: vitables /Users/lizejian/cambridge/mphil_project/learn/udiva/DeepPersonality/datasets/udiva_tiny/train/annotations/animals_annotations_train/055128/FC1_A/annotations_raw.hdf5

# /Users/lizejian/cambridge/mphil_project/learn/udiva/DeepPersonality/datasets/udiva_tiny/train/annotations/animals_annotations_train/055128/FC1_A/annotations_raw.hdf5
import h5py

# 读取HDF5文件中的所有数据集
def traverse_datasets(hdf_file):
    def h5py_dataset_iterator(g, prefix=''):
        for key in g.keys():
            item = g[key]
            path = '{}/{}'.format(prefix, key)
            if isinstance(item, h5py.Dataset): # test for dataset
                yield (path, item)
            elif isinstance(item, h5py.Group): # test for group (go down)
                yield from h5py_dataset_iterator(item, path)

    with h5py.File(hdf_file, 'r') as f:
        for (path, dset) in h5py_dataset_iterator(f):
            print(path, dset)

    return None

# 传入路径即可
traverse_datasets('/Users/lizejian/cambridge/mphil_project/learn/udiva/DeepPersonality/datasets/udiva_tiny/train/annotations/animals_annotations_train/055128/FC1_A/annotations_raw.hdf5')


