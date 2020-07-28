import torch
import os
import os.path as osp
from options import TrainOptions, dataset_list


# ---------------------------------------------- save model -----------------------------------------------------
def save_model(i_iter, args, SegModel, model_D, optimizer, optimizer_D, optimizer_warp, snapshot_dir, dir_name):
    # Snapshots directory
    if args.multi_gpu:
        info = {'state_dict_seg': SegModel.module.state_dict()}
    else:
        info = {'state_dict_seg': SegModel.state_dict()}

    info['optimizer_seg'] = optimizer.state_dict()
    info['discriminator'] = model_D.state_dict()
    info['optimizer_disc'] = optimizer_D.state_dict()

    if (args.warper or args.spadeWarper):
        info['optimizer_warp'] = optimizer_warp.state_dict()
    else:
        info['optimizer_warp'] = None

    # SegNet_name = 'checkpoint'
    SegNet_name = args.segnet_name

    dir_path = osp.join(snapshot_dir, dir_name)

    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    cont_data_name = dataset_list[0]
    for i in range(1, args.num_dataset):
        cont_data_name += '_'
        cont_data_name += dataset_list[i]

    SegNet_name += cont_data_name

    model_save_name = osp.join(dir_path, SegNet_name)

    return model_save_name, info


def load_existing_state_dict(model, saved_state_dict):
    new_params = model.state_dict().copy()
    for i in saved_state_dict:
        if i in new_params.keys():
            new_params[i] = saved_state_dict[i]
    model.load_state_dict(new_params)

    return model
