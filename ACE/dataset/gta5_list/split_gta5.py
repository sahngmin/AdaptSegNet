import os
import random

write_new_ori = False
data_path = '/home/smyoo/CAG_UDA/dataset/GTA5'


if write_new_ori:
    total_files = sorted([f for dp, dn, fn in os.walk(os.path.expanduser(data_path)) for f in fn])
    write_file = open('train_ori_new.txt', 'w')
    for file_name in total_files:
        write_file.writelines(file_name)

random.seed(0)
list_path = 'train_ori.txt'
img_ids = [i_id for i_id in open(list_path)]

random.shuffle(img_ids)

val_len = int(len(img_ids) / 7)
val_list = img_ids[:val_len]
train_list = img_ids[val_len:]

write_val = open('val.txt', 'w')
for val_file in val_list:
    write_val.writelines(val_file)
write_val.close()

write_train = open('train.txt', 'w')
for train_file in train_list:
    write_train.writelines(train_file)
write_train.close()