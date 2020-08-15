import os
import random
from shutil import copyfile

write_new_ori = False
data_path = "/work/SYNTHIA"
sub_dir = "RGB"

if write_new_ori:
    total_files = sorted([f for dp, dn, fn in os.walk(os.path.expanduser(os.path.join(data_path, sub_dir))) for f in fn])
    write_file = open(os.path.join(data_path, 'train_ori_new.txt'), 'w')
    for file_name in total_files:
        write_file.writelines(file_name + '\n')


random.seed(0)
list_path = os.path.join(data_path, 'train_ori_new.txt')
img_ids = [i_id for i_id in open(list_path)]

random.shuffle(img_ids)

val_len = int(len(img_ids) / 7)  # 7
val_list = img_ids[:val_len]
train_list = img_ids[val_len:]

write_val = open(os.path.join(data_path, 'val.txt'), 'w')
for val_file in val_list:
    write_val.writelines(val_file)
    val_file = val_file[:-1]
    file_path_val = os.path.join(data_path, sub_dir, val_file)
    if not os.path.exists(os.path.join(data_path, "val")):
        os.makedirs(os.path.join(data_path, "val"))
    copyfile(file_path_val, os.path.join(data_path, "val", val_file))

write_val.close()

write_train = open(os.path.join(data_path, 'train.txt'), 'w')
for train_file in train_list:
    write_train.writelines(train_file)
    train_file = train_file[:-1]
    file_path_train = os.path.join(data_path, sub_dir, train_file)
    if not os.path.exists(os.path.join(data_path, "train")):
        os.makedirs(os.path.join(data_path, "train"))
    copyfile(file_path_train, os.path.join(data_path, "train", train_file))
write_train.close()