import os
import sys
import h5py
import numpy as np

cat_name = sys.argv[1]
level = sys.argv[2]
print(cat_name, level)

in_dir = '../data/sem_seg_multi_labeling_h5/%s' % cat_name
out_dir = '../data/partnet_sem_seg/%s-%s' % (cat_name, level)
if os.path.exists(out_dir):
    print('ERROR: folder %s exist! Please check and delete it!' % out_dir)
    exit(1)
os.mkdir(out_dir)

stat_fn = '../data/partnet_stats/%s-level-%s.txt' % (cat_name, level)
with open(stat_fn, 'r') as fin:
    label_mask = [int(line.rstrip().split()[0]) for line in fin.readlines()]
label_mask = [0] + label_mask
label_mask = np.array(label_mask, dtype=np.int32)
print(label_mask)

def load_h5(fn):
    with h5py.File(fn, 'r') as fin:
        return (fin['pts'][:], fin['label'][:])

def save_h5(fn, data, data_num, label_seg):
    fout = h5py.File(fn, 'w')
    fout.create_dataset('data', data=data, compression='gzip', compression_opts=4, dtype='float64')
    fout.create_dataset('data_num', data=data_num, compression='gzip', compression_opts=4, dtype='int32')
    fout.create_dataset('label_seg', data=label_seg, compression='gzip', compression_opts=4, dtype='int32')
    fout.close()

train_filelist = []
val_filelist = []
test_filelist = []
for item in os.listdir(in_dir):
    if item.endswith('.h5'):
        if item.startswith('train-'):
            train_filelist.append('./'+item)
        elif item.startswith('val-'):
            val_filelist.append('./'+item)
        elif item.startswith('test-'):
            test_filelist.append('./'+item)
        else:
            print('Skip %s' % item)
            continue

        print(item)

        h5_fn = os.path.join(in_dir, item)
        pts, label = load_h5(h5_fn)
        label = label[:, :, label_mask]
        label[:, :, 0] = (np.sum(label[:, :, 1:], axis=-1) == 0)
        assert np.sum(np.sum(label, axis=-1) == 1) == pts.shape[0] * pts.shape[1]
        label = np.argmax(label, axis=-1)
        num_point = np.ones((pts.shape[0]), dtype=np.int32) * pts.shape[1]
        save_h5(os.path.join(out_dir, item), pts, num_point, label)

with open(os.path.join(out_dir, 'train_files.txt'), 'w') as fout:
    for item in train_filelist:
        fout.write('%s\n' % item)
with open(os.path.join(out_dir, 'val_files.txt'), 'w') as fout:
    for item in val_filelist:
        fout.write('%s\n' % item)
with open(os.path.join(out_dir, 'test_files.txt'), 'w') as fout:
    for item in test_filelist:
        fout.write('%s\n' % item)

