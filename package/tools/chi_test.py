import argparse
import numpy as np

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('file_a')
    parser.add_argument('file_b')
    args = parser.parse_args()

    a = np.load(args.file_a).item()
    b = np.load(args.file_b).item()
    succ_a = a['succ_idx']
    succ_b = b['succ_idx']
    test_a = a['test_idx']
    test_b = b['test_idx']

    assert len(test_a) == len(test_b)
    succ_both = list(set(succ_a).intersection(set(succ_b)))

    print('succ both: ', succ_both)
    print('succ a: ', succ_a)
    print('succ b: ', succ_b)
    print('test a: ', test_a)
    print('test b: ', test_b)

    print('succ both: ', len(succ_both))
    print('succ a: ', len(succ_a))
    print('succ b: ', len(succ_b))
    print('test: ', len(test_a))
