from typing import List

def extract_acc(path: str, to_print: bool=True) -> (List[float], List[float]):
    '''
    extract clean and attack accuracies from result files
    '''
    clean_acc, attack_acc = [], []
    lines = open(path, 'r').readlines()
    for l in lines:
        if 'Validate' in l:
            tmp = l.strip().split(',')
            for item in tmp:
                if 'Top1 Acc' in item:
                    acc = item.strip().split(':')[1].strip()
                    acc = float(acc)
                    
                    if 'Validate Clean' in l:
                        clean_acc.append(acc)
                    elif 'Validate Trigger Tgt' in l:
                        attack_acc.append(acc)
    if to_print:
        print('number of epochs: ', len(clean_acc))
        print(clean_acc)
        print(attack_acc)
    return clean_acc, attack_acc

# example
path = '/home/rbp5354/trojanzoo/result/cifar10/resnetcomp18/badnet/fine_tuning_full_size4.txt'
clean_acc, attack_acc = extract_acc(path)
