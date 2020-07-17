import torch
import os
import os.path as osp
from options import TrainOptions, dataset_list


# ---------------------------------------------- save model -----------------------------------------------------
def save_model(i_iter, args, SegModel, model_D, optimizer, optimizer_D, optimizer_warp):
    # Snapshots directory
    if args.multi_gpu:
        info = {'state_dict_seg': SegModel.module.state_dict()}
    else:
        info = {'state_dict_seg': SegModel.state_dict()}

    info['optimizer_seg'] = optimizer.state_dict()
    info['optimizer_warp'] = optimizer_warp.state_dict()
    info['discriminator'] = model_D.state_dict()
    info['disc_optimizer'] = optimizer_D.state_dict()


    # dir_name = 'single_alignment_warper'
    # SegNet_name = 'checkpoint'

    dir_name = args.dir_name
    SegNet_name = args.segnet_name

    dir_path = osp.join(args.snapshot_dir, dir_name)

    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    cont_data_name = dataset_list[0]
    for i in range(1, args.num_dataset):
        cont_data_name += '_'
        cont_data_name += dataset_list[i]

    SegNet_name += cont_data_name

    model_save_name = osp.join(dir_path, SegNet_name)


    if i_iter >= args.num_steps_stop - 1:
        print('save model ...')
        model_save_name += '_' + str(args.num_steps_stop)

        torch.save(info, model_save_name + '.pth')

        return True

    if i_iter % args.save_pred_every == 0 and i_iter != 0:
        print('taking snapshot ...')
        model_save_name += '_' + str(i_iter)

        torch.save(info, model_save_name + '.pth')


