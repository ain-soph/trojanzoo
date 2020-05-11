import argparse
import numpy as np

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('file')
    args = parser.parse_args()

    a = np.load(args.file).item()
    succ = a['succ_idx']
    test = a['test_idx']

    print('succ: ', succ)
    print('test: ', test)

    print('succ: ', len(succ))
    print('test: ', len(test))

    print('rate: ', float(len(succ))/len(test))
