import torch
import os
import os.path as osp
from options import TrainOptions, dataset_dict


# ---------------------------------------------- save model -----------------------------------------------------
def save_model(i_iter, args, model, model_D):
    # Snapshots directory
    if args.source_only:
        if args.warper and args.memory:
            if not os.path.exists(osp.join(args.snapshot_dir, 'source_only_warp_DM')):
                os.makedirs(osp.join(args.snapshot_dir, 'source_only_warp_DM'))
        elif args.warper:
            if not os.path.exists(osp.join(args.snapshot_dir, 'source_only_warp')):
                os.makedirs(osp.join(args.snapshot_dir, 'source_only_warp'))
        elif args.memory:
            if not os.path.exists(osp.join(args.snapshot_dir, 'source_only_DM')):
                os.makedirs(osp.join(args.snapshot_dir, 'source_only_DM'))
        else:
            if not os.path.exists(osp.join(args.snapshot_dir, 'source_only')):
                os.makedirs(osp.join(args.snapshot_dir, 'source_only'))
    else:
        if args.warper and args.memory:
            if not os.path.exists(osp.join(args.snapshot_dir, 'single_alignment_warp_DM')):
                os.makedirs(osp.join(args.snapshot_dir, 'single_alignment_warp_DM'))
        elif args.warper:
            if not os.path.exists(osp.join(args.snapshot_dir, 'single_alignment_warp')):
                os.makedirs(osp.join(args.snapshot_dir, 'single_alignment_warp'))
        elif args.memory:
            if not os.path.exists(osp.join(args.snapshot_dir, 'single_alignment_DM')):
                os.makedirs(osp.join(args.snapshot_dir, 'single_alignment_DM'))
        else:
            if not os.path.exists(osp.join(args.snapshot_dir, 'single_alignment')):
                os.makedirs(osp.join(args.snapshot_dir, 'single_alignment'))

    if i_iter >= args.num_steps_stop - 1:
        print('save model ...')
        if args.source_only:
            if args.warper and args.memory:
                torch.save(model.state_dict(), osp.join(args.snapshot_dir, 'source_only_warp_DM',
                                                        'GTA5_' + str(args.num_steps_stop) + '.pth'))
            elif args.warper:
                torch.save(model.state_dict(), osp.join(args.snapshot_dir, 'source_only_warp',
                                                        'GTA5_' + str(args.num_steps_stop) + '.pth'))
            elif args.memory:
                torch.save(model.state_dict(), osp.join(args.snapshot_dir, 'source_only_DM',
                                                        'GTA5_' + str(args.num_steps_stop) + '.pth'))
            else:
                torch.save(model.state_dict(), osp.join(args.snapshot_dir, 'source_only',
                                                        'GTA5_' + str(args.num_steps_stop) + '.pth'))
        else:
            if args.num_dataset == 1:
                if args.warper and args.memory:
                    torch.save(model.state_dict(),
                               osp.join(args.snapshot_dir, 'single_alignment_warp_DM',
                                        'GTA5to' + str(args.target) + str(args.num_steps_stop) + '.pth'))
                    torch.save(model_D.state_dict(),
                               osp.join(args.snapshot_dir, 'single_alignment_warp_DM',
                                        'GTA5to' + str(args.target) + str(args.num_steps_stop) + '_D.pth'))
                elif args.warper:
                    torch.save(model.state_dict(),
                               osp.join(args.snapshot_dir, 'single_alignment_warp',
                                        'GTA5to' + str(args.target) + str(args.num_steps_stop) + '.pth'))
                    torch.save(model_D.state_dict(),
                               osp.join(args.snapshot_dir, 'single_alignment_warp',
                                        'GTA5to' + str(args.target) + str(args.num_steps_stop) + '_D.pth'))
                elif args.memory:
                    torch.save(model.state_dict(),
                               osp.join(args.snapshot_dir, 'single_alignment_DM',
                                        'GTA5to' + str(args.target) + str(args.num_steps_stop) + '.pth'))
                    torch.save(model_D.state_dict(),
                               osp.join(args.snapshot_dir, 'single_alignment_DM',
                                        'GTA5to' + str(args.target) + str(args.num_steps_stop) + '_D.pth'))
                else:
                    torch.save(model.state_dict(),
                               osp.join(args.snapshot_dir, 'single_alignment',
                                        'GTA5to' + str(args.target) + str(args.num_steps_stop) + '.pth'))
                    torch.save(model_D.state_dict(),
                               osp.join(args.snapshot_dir, 'single_alignment',
                                        'GTA5to' + str(args.target) + str(args.num_steps_stop) + '_D.pth'))
            else:
                targetlist = list(dataset_dict.keys())
                filename = 'GTA5to'
                for i in range(args.num_dataset - 1):
                    filename += targetlist[i]
                    filename += 'to'
                if args.warper and args.memory:
                    torch.save(model.state_dict(),
                               osp.join(args.snapshot_dir, 'single_alignment_warp_DM',
                                        filename + str(args.target) + str(args.num_steps_stop) + '.pth'))
                    torch.save(model_D.state_dict(),
                               osp.join(args.snapshot_dir, 'single_alignment_warp_DM',
                                        filename + str(args.target) + str(args.num_steps_stop) + '_D.pth'))
                elif args.warper:
                    torch.save(model.state_dict(),
                               osp.join(args.snapshot_dir, 'single_alignment_warp',
                                        filename + str(args.target) + str(args.num_steps_stop) + '.pth'))
                    torch.save(model_D.state_dict(),
                               osp.join(args.snapshot_dir, 'single_alignment_warp',
                                        filename + str(args.target) + str(args.num_steps_stop) + '_D.pth'))
                elif args.memory:
                    torch.save(model.state_dict(),
                               osp.join(args.snapshot_dir, 'single_alignment_DM',
                                        filename + str(args.target) + str(args.num_steps_stop) + '.pth'))
                    torch.save(model_D.state_dict(),
                               osp.join(args.snapshot_dir, 'single_alignment_DM',
                                        filename + str(args.target) + str(args.num_steps_stop) + '_D.pth'))
                else:
                    torch.save(model.state_dict(),
                               osp.join(args.snapshot_dir, 'single_alignment',
                                        filename + str(args.target) + str(args.num_steps_stop) + '.pth'))
                    torch.save(model_D.state_dict(),
                               osp.join(args.snapshot_dir, 'single_alignment',
                                        filename + str(args.target) + str(args.num_steps_stop) + '_D.pth'))

        return True

    if i_iter % args.save_pred_every == 0 and i_iter != 0:
        print('taking snapshot ...')
        if args.source_only:
            if args.warper and args.memory:
                torch.save(model.state_dict(), osp.join(args.snapshot_dir, 'source_only_warp_DM',
                                                        'GTA5_' + str(i_iter) + '.pth'))
            elif args.warper:
                torch.save(model.state_dict(), osp.join(args.snapshot_dir, 'source_only_warp',
                                                        'GTA5_' + str(i_iter) + '.pth'))
            elif args.memory:
                torch.save(model.state_dict(), osp.join(args.snapshot_dir, 'source_only_DM',
                                                        'GTA5_' + str(i_iter) + '.pth'))
            else:
                torch.save(model.state_dict(), osp.join(args.snapshot_dir, 'source_only',
                                                        'GTA5_' + str(i_iter) + '.pth'))
        else:
            if args.num_dataset == 1:
                if args.warper and args.memory:
                    torch.save(model.state_dict(),
                               osp.join(args.snapshot_dir, 'single_alignment_warp_DM',
                                        'GTA5to' + str(args.target) + str(i_iter) + '.pth'))
                    torch.save(model_D.state_dict(),
                               osp.join(args.snapshot_dir, 'single_alignment_warp_DM',
                                        'GTA5to' + str(args.target) + str(i_iter) + '_D.pth'))
                elif args.warper:
                    torch.save(model.state_dict(),
                               osp.join(args.snapshot_dir, 'single_alignment_warp',
                                        'GTA5to' + str(args.target) + str(i_iter) + '.pth'))
                    torch.save(model_D.state_dict(),
                               osp.join(args.snapshot_dir, 'single_alignment_warp',
                                        'GTA5to' + str(args.target) + str(i_iter) + '_D.pth'))
                elif args.memory:
                    torch.save(model.state_dict(),
                               osp.join(args.snapshot_dir, 'single_alignment_DM',
                                        'GTA5to' + str(args.target) + str(i_iter) + '.pth'))
                    torch.save(model_D.state_dict(),
                               osp.join(args.snapshot_dir, 'single_alignment_DM',
                                        'GTA5to' + str(args.target) + str(i_iter) + '_D.pth'))
                else:
                    torch.save(model.state_dict(),
                               osp.join(args.snapshot_dir, 'single_alignment',
                                        'GTA5to' + str(args.target) + str(i_iter) + '.pth'))
                    torch.save(model_D.state_dict(),
                               osp.join(args.snapshot_dir, 'single_alignment',
                                        'GTA5to' + str(args.target) + str(i_iter) + '_D.pth'))
            else:
                targetlist = list(dataset_dict.keys())
                filename = 'GTA5to'
                for i in range(args.num_dataset - 1):
                    filename += targetlist[i]
                    filename += 'to'
                if args.warper and args.memory:
                    torch.save(model.state_dict(),
                               osp.join(args.snapshot_dir, 'single_alignment_warp_DM',
                                        filename + str(args.target) + str(i_iter) + '.pth'))
                    torch.save(model_D.state_dict(),
                               osp.join(args.snapshot_dir, 'single_alignment_warp_DM',
                                        filename + str(args.target) + str(i_iter) + '_D.pth'))
                elif args.warper:
                    torch.save(model.state_dict(),
                               osp.join(args.snapshot_dir, 'single_alignment_warp',
                                        filename + str(args.target) + str(i_iter) + '.pth'))
                    torch.save(model_D.state_dict(),
                               osp.join(args.snapshot_dir, 'single_alignment_warp',
                                        filename + str(args.target) + str(i_iter) + '_D.pth'))
                elif args.memory:
                    torch.save(model.state_dict(),
                               osp.join(args.snapshot_dir, 'single_alignment_DM',
                                        filename + str(args.target) + str(i_iter) + '.pth'))
                    torch.save(model_D.state_dict(),
                               osp.join(args.snapshot_dir, 'single_alignment_DM',
                                        filename + str(args.target) + str(i_iter) + '_D.pth'))
                else:
                    torch.save(model.state_dict(),
                               osp.join(args.snapshot_dir, 'single_alignment',
                                        filename + str(args.target) + str(i_iter) + '.pth'))
                    torch.save(model_D.state_dict(),
                               osp.join(args.snapshot_dir, 'single_alignment',
                                        filename + str(args.target) + str(i_iter) + '_D.pth'))