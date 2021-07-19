
import numpy as np
import torch
import torch.utils.data


def get_dataset(dataset, data_path):
    from torchvision import datasets, transforms
    if dataset == 'MNIST':
        channel = 1
        im_size = (28, 28)
        num_classes = 10
        mean = [0.1307]
        std = [0.3081]
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
        dst_train = datasets.MNIST(data_path, train=True, download=True, transform=transform)  # no augmentation
        dst_test = datasets.MNIST(data_path, train=False, download=True, transform=transform)
        class_names = [str(c) for c in range(num_classes)]

    else:
        exit('unknown dataset: %s' % dataset)

    testloader = torch.utils.data.DataLoader(dst_test, batch_size=256, shuffle=False, num_workers=2)
    return channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader


channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader = get_dataset(
    'MNIST', '~/data_temp/')
images_all = []
labels_all = []
indices_class = [[] for c in range(num_classes)]

images_all = [torch.unsqueeze(dst_train[i][0], dim=0) for i in range(len(dst_train))]
labels_all = [dst_train[i][1] for i in range(len(dst_train))]
for i, lab in enumerate(labels_all):
    indices_class[lab].append(i)
images_all = torch.cat(images_all, dim=0).to('cuda')
labels_all = torch.tensor(labels_all, dtype=torch.long, device='cuda')


def get_images(c, n):  # get random n images from class c
    idx_shuffle = np.random.permutation(indices_class[c])[:n]
    return images_all[idx_shuffle].detach().clone()
