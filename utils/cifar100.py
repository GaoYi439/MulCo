import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.datasets as dsets
import numpy as np
import copy
# np.random.seed(2)


def load_cifar100(batch_size, num):
    test_transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize(mean=[0.5070751592371323, 0.48654887331495095, 0.4409178433670343],
                              std=[0.2673342858792401, 0.2564384629170883, 0.27615047132568404])])

    temp_train = dsets.CIFAR100(root='./data/cifar100', train=True, download=False, transform=transforms.ToTensor())
    data, labels = temp_train.data, torch.Tensor(temp_train.targets).long()
    # get original data and labels

    test_dataset = dsets.CIFAR100(root='./data/cifar100', train=False, transform=test_transform)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size * 4, shuffle=False,
                                              num_workers=4,
                                              sampler=torch.utils.data.distributed.DistributedSampler(test_dataset,
                                                                                                      shuffle=False))
    # set test dataloader

    com_label = generate_compl_labels(labels, num)
    # partialY = generate_uniform_cv_candidate_labels(labels, 0.1)
    # com_label = 1 - partialY
    # generate complementary labels

    # one-hot true labels matrix
    temp = torch.zeros(com_label.shape)
    temp[torch.arange(com_label.shape[0]), labels] = 1

    partial_matrix_dataset = CIFAR100_Augmentention(data, com_label.float(), labels.float())
    # generate partial label dataset

    train_sampler = torch.utils.data.distributed.DistributedSampler(partial_matrix_dataset)

    partial_matrix_train_loader = torch.utils.data.DataLoader(dataset=partial_matrix_dataset,
                                                              batch_size=batch_size,
                                                              shuffle=(train_sampler is None),
                                                              num_workers=4,
                                                              pin_memory=True,
                                                              sampler=train_sampler,
                                                              drop_last=True)
    return partial_matrix_train_loader, train_sampler, test_loader


class CIFAR100_Augmentention(Dataset):
    def __init__(self, images, com_labels, true_labels):
        '''
        :param images: data
        :param com_labels: generative complementary label
        :param true_labels: the true label of data, type: long
        '''

        self.images = images
        self.com_labels = com_labels
        # user-defined label (partial labels)
        self.true_labels = true_labels
        # self.weak_transform = transforms.Compose(
        #     [
        #         transforms.ToPILImage(),
        #         transforms.RandomResizedCrop(size=32, scale=(0.2, 1.)),  # 裁剪
        #         transforms.RandomHorizontalFlip(),  # 水平翻转
        #         transforms.RandomApply([
        #             transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        #         ], p=0.8),  # transforms.RandomApply：从给定的一系列transforms中选一个进行操作
        #         transforms.RandomGrayscale(p=0.2),  # 转灰度图
        #         transforms.ToTensor(),
        #         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])  # 对数据进行归一化
        # self.strong_transform = transforms.Compose(
        #     [
        #         transforms.ToPILImage(),
        #         # transforms.ToTensor() 将numpy的ndarray或PIL.Image读的图片转换成形状为(C,H, W)的Tensor格式，且/255归一化到[0,1.0]之间
        #         transforms.RandomResizedCrop(size=32, scale=(0.2, 1.)),
        #         transforms.RandomHorizontalFlip(),
        #         RandomAugment(3, 5),  # 数据增强
        #         transforms.ToTensor(),
        #         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        self.weak_transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, 4, padding_mode='reflect'),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5070751592371323, 0.48654887331495095, 0.4409178433670343],
                                     std=[0.2673342858792401, 0.2564384629170883, 0.27615047132568404]),])  # 对数据进行归一化
        self.strong_transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, 4, padding_mode='reflect'),
                # transforms.ToTensor(),
                # Cutout(n_holes=1, length=16),
                # transforms.ToPILImage(),
                # CIFAR10Policy(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5070751592371323, 0.48654887331495095, 0.4409178433670343],
                                     std=[0.2673342858792401, 0.2564384629170883, 0.27615047132568404]),])

    def __len__(self):
        return len(self.true_labels)

    def __getitem__(self, index):
        each_image_w = self.weak_transform(self.images[index])
        each_image_s = self.strong_transform(self.images[index])
        each_label = self.com_labels[index]
        each_true_label = self.true_labels[index]

        return each_image_w, each_image_s, each_label, each_true_label

def generate_compl_labels(train_labels, num_com):
    k = max(train_labels) + 1
    n = len(train_labels)
    index_ins = np.arange(n)  # torch type
    realY = np.zeros([n, k])
    realY[index_ins, train_labels] = 1
    partialY = np.ones([n, k])

    labels_hat = np.array(copy.deepcopy(train_labels))
    candidates = np.repeat(np.arange(k).reshape(1, k), len(labels_hat), 0) # candidate labels without true class
    mask = np.ones((len(labels_hat), k), dtype=bool)
    for i in range(num_com):
        mask[np.arange(n), labels_hat] = False
        candidates_ = candidates[mask].reshape(n, k-1-i)
        idx = np.random.randint(0, k-1-i, n)
        comp_labels = candidates_[np.arange(n), np.array(idx)]
        partialY[index_ins, torch.from_numpy(comp_labels)] = 0
        labels_hat = comp_labels
    return torch.from_numpy(1 - partialY)

def generate_uniform_cv_candidate_labels(train_labels, partial_rate=0.1):
    if torch.min(train_labels) > 1:
        raise RuntimeError('testError')
    elif torch.min(train_labels) == 1:
        train_labels = train_labels - 1

    K = int(torch.max(train_labels) - torch.min(train_labels) + 1)
    n = train_labels.shape[0]

    partialY = torch.zeros(n, K)
    partialY[torch.arange(n), train_labels] = 1.0
    p_1 = partial_rate
    transition_matrix = np.eye(K)  #返回对角线为1其余位置为0的矩阵
    transition_matrix[np.where(~np.eye(transition_matrix.shape[0],dtype=bool))]=p_1  #除对角线位置元素为1，其余位置元素为p_1
    print(transition_matrix)

    # 如果产生的随机数小于翻转概率，那么该位置的非真实标记将会标记为1
    random_n = np.random.uniform(0, 1, size=(n, K))

    for j in range(n):  # for each instance
        for jj in range(K): # for each class
            if jj == train_labels[j]: # except true class
                continue
            if random_n[j, jj] < transition_matrix[train_labels[j], jj]:
                partialY[j, jj] = 1.0

    print("Finish Generating Candidate Label Sets!\n")
    return partialY