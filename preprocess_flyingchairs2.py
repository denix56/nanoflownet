import os

import numpy as np
from argparse import ArgumentParser

from tqdm import tqdm

def save_flow_as_np(name):
    if name[-3:] != 'flo':
        return

    f = open(name, 'rb')

    header = f.read(4)
    if header.decode("utf-8") != 'PIEH':
        raise Exception('Flow file header does not contain PIEH')

    width = np.fromfile(f, np.int32, 1).squeeze()
    height = np.fromfile(f, np.int32, 1).squeeze()

    flow = np.fromfile(f, np.float32, width * height * 2).reshape((height, width, 2))

    flow = flow.astype(np.float32)
    np.save(name[:-3] + 'npy', flow)
    os.remove(name)


if __name__ == '__main__':
    parser = ArgumentParser('Preprocess FlyingChairs2')
    parser.add_argument('--root', type=str, help='Root directory of FlyingChairs2')
    args = parser.parse_args()

    for split in ['train', 'val']:
        files = [f for f in os.listdir(os.path.join(args.root, split)) if f[-3:] == 'flo']
        for f in tqdm(files):
            save_flow_as_np(os.path.join(args.root, split, f))