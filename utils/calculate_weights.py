import os
from tqdm import tqdm
import numpy as np
from mypath import Path

def calculate_weigths_labels(dataset, dataloader, num_classes):
    # Create an instance from the data loader
    z = np.zeros((num_classes,))
    # Initialize tqdm
    tqdm_batch = tqdm(dataloader)
    print('Calculating classes weights')
    for sample in tqdm_batch:
        y = sample['label']
        y = y.detach().cpu().numpy()
        # 返回一个新的Variable，从当前计算图中分离下来的，但是仍指向原变量的存放位置,不同之处只是requires_grad为false，
        # 得到的这个Variable永远不需要计算其梯度，不具有grad。即使之后重新将它的requires_grad置为true,它也不会具有梯度grad
        mask = (y >= 0) & (y < num_classes)    # mask = [True  True  True  True  True  True  True  True  True  True]
        labels = y[mask].astype(np.uint8)
        count_l = np.bincount(labels, minlength=num_classes)    # 统计数组中值的出现次数 (统计每一类别标签的个数)
        z += count_l
    tqdm_batch.close()
    total_frequency = np.sum(z)
    class_weights = []
    for frequency in z:
        class_weight = 1 / (np.log(1.02 + (frequency / total_frequency)))    # np.log(x):计算x的自然对数
        class_weights.append(class_weight)
    ret = np.array(class_weights)
    classes_weights_path = os.path.join(Path.db_root_dir(dataset), dataset+'_classes_weights.npy')
    np.save(classes_weights_path, ret)

    return ret