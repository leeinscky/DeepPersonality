import torch
import torch.multiprocessing as mp
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def f(rank, x):
    x[rank] = rank
    print('Rank:', rank, 'x:', x)

if __name__ == '__main__':
    size = 4
    empty_tensor = torch.zeros((1)).to(device)
    print('empty_tensor.device:', empty_tensor.device)
    processes = []
    mp.set_start_method('spawn', force=True)
    # x = [empty_tensor] * size
    x = mp.Manager().list([empty_tensor] * size) # 使用共享内存 # source: mp.Manager() is inspired by chatGPT
    print('x:', x)
    # x = torch.tensor(x)
    # print('x:', x)
    # x[1] = 100
    # print('x:', x)
    
    # x = np.array(x)# list转numpy.array
    # x = torch.from_numpy(x) # array2tensor
    # print('x:', x, 'x.type:', x.type())
    
    for rank in range(size):
        p = mp.Process(target=f, args=(rank, x))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
    print('Final x:', x)


# def f(rank, tensor):
#     tensor += rank
#     print('Rank:', rank, 'Tensor:', tensor)

# if __name__ == '__main__':
#     size = 4
#     tensor = torch.zeros(size)
#     processes = []
#     for rank in range(size):
#         p = mp.Process(target=f, args=(rank, tensor))
#         p.start()
#         processes.append(p)
#     for p in processes:
#         p.join()
#     print('Final tensor:', tensor)