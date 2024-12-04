import torch
import monai
import time
import wandb
from argparse import ArgumentParser

from methods.segm import create_segmenter
from data.utils import kvasir_data_to_dict
from utils import count_parameters

def main(args):
    for arg in vars(args):
        print("{} : {}".format(arg, getattr(args, arg)))

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    ### LOGGING ###
    if args.wandb_logging:
        wandb.login()
        wandb.init(project="CandidateNet", entity="eiphodos", config=args)

    ### DATA ###
    train_data = kvasir_data_to_dict(args.train_data_folder)
    val_data = kvasir_data_to_dict(args.val_data_folder)
    test_data = kvasir_data_to_dict(args.test_data_folder)

    transform_list = [monai.transforms.LoadImaged(keys=['image', 'label']),
              monai.transforms.EnsureChannelFirstD(keys=['image', "label"]),
              monai.transforms.NormalizeIntensityd(keys=['image'], divisor=[255, 255, 255], channel_wise=True),
              monai.transforms.ScaleIntensityRanged(keys=['label'], a_min=0, a_max=255, b_min=0, b_max=1,clip=True),
              monai.transforms.Resized(keys=["image"], spatial_size=(512, 512), mode='bilinear'),
              monai.transforms.Resized(keys=["label"], spatial_size=(512, 512), mode='nearest')]

    if args.augment:
              aug_list = [monai.transforms.RandRotate90d(keys=['image', 'label'], prob=0.1),
                          monai.transforms.RandGibbsNoised(keys=['image'], prob=0.1),
                          monai.transforms.RandAdjustContrastd(keys=['image'], prob=0.1),
                          monai.transforms.RandZoomd(keys=['image', 'label'], prob=0.1, min_zoom=0.8, max_zoom=1.2)
                          ]
              transform_list_train = transform_list + aug_list
    else:
        transform_list_train = transform_list
    transform_list_train.append(monai.transforms.ToTensord(keys=["image", "label"]))

    transform_list_val = transform_list
    transform_list_val.append(monai.transforms.ToTensord(keys=["image", "label"]))

    transforms_train = monai.transforms.Compose(transform_list_train)
    transforms_val = monai.transforms.Compose(transform_list_val)

    train_dataset = monai.data.Dataset(train_data, transform=transforms_train)
    val_dataset = monai.data.Dataset(val_data, transform=transforms_val)

    data_loader_val = monai.data.DataLoader(
        val_dataset,
        batch_size=1,
        num_workers=4,
        pin_memory=True,
        drop_last=False,
        persistent_workers=True
        )

    data_loader_train = monai.data.DataLoader(
        train_dataset,
        batch_size=8,
        num_workers=8,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True
        )
    
    ### MODEL ###
    model_cfg = {}
    model_cfg["image_size"] = (args.image_size, args.image_size)
    model_cfg["patch_size"] = args.patch_size
    model_cfg["d_model"] = args.d_encoder
    model_cfg["n_heads"] = args.n_heads_encoder
    model_cfg["n_layers"] = args.n_encoder_layers
    model_cfg["normalization"] = 'vit'
    model_cfg["distilled"] = False
    model_cfg["backbone"] = 'custom'
    model_cfg["dropout"] = 0.0
    model_cfg["drop_path_rate"] = 0.0
    model_cfg["n_cls"] = 2
    model_cfg["policy_method"] = args.policy_method # 'policy_net'  | 'no_sharing'
    #model_cfg["policy_schedule"] = (1024, 0)
    #model_cfg["policy_schedule"] = (512, 128)
    #model_cfg["policy_schedule"] = (128, 224)
    #model_cfg["policy_schedule"] = (64, 240)
    #model_cfg["policy_schedule"] = (512, 896)
    model_cfg["policy_schedule"] = args.policy_schedule
    model_cfg["policynet_ckpt"] = args.policynet_ckpt # 'policy_net_kvasir_150.pth'
    decoder_cfg = {}
    decoder_cfg["drop_path_rate"] = 0.0
    decoder_cfg["dropout"] = 0.0
    decoder_cfg["n_layers"] = args.n_decoder_layers
    decoder_cfg["name"] = 'mask_transformer'
    model_cfg["decoder"] = decoder_cfg 
    
    model = create_segmenter(model_cfg)
    model.to(device)
    print(model)
    print("Total parameters: {}".format(count_parameters(model)))
    print("Encoder parameters: {}".format(count_parameters(model.encoder)))
    print("Decoder parameters: {}".format(count_parameters(model.decoder)))
    if model_cfg["policy_method"] == 'policy_net':
        print("Policy net parameters: {}".format(count_parameters(model.policy_net)))

    ### OPTIMIZER/CRITERION ###
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    criterion = monai.losses.DiceCELoss(to_onehot_y=True, softmax=True, squared_pred=False)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    ### METRICS ###
    post_label = monai.transforms.AsDiscrete(to_onehot=2)
    post_pred = monai.transforms.AsDiscrete(argmax=True, to_onehot=2)
    dice_metric = monai.metrics.DiceMetric(include_background=True, reduction="mean", get_not_nans=True)
    iou_metric = monai.metrics.MeanIoU(include_background=True, reduction="mean", get_not_nans=True)
    dice_metric_val = monai.metrics.DiceMetric(include_background=True, reduction="mean", get_not_nans=True)
    iou_metric_val = monai.metrics.MeanIoU(include_background=True, reduction="mean", get_not_nans=True)

    ### TRAINING LOOP ###
    print("Starting training...")
    for e in range(args.epochs):
        epoch_start = time.time()
        for i, batch in enumerate(data_loader_train):
            batch_start = time.time()
            inputs, labels = (batch["image"], batch["label"])
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            outputs, _, _, _ = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            labels_convert = [post_label(labels[0])]
            output_convert = [post_pred(outputs[0])]
            dice_metric(y_pred=output_convert, y=labels_convert)
            iou_metric(y_pred=output_convert, y=labels_convert)
            batch_time = time.time() - batch_start
        dice_scores, dice_not_nans = dice_metric.aggregate()
        iou_scores, iou_not_nans = iou_metric.aggregate()
        epoch_time = time.time() - epoch_start
        print("Train Epoch: {}, Dice score: {:.4f}, IoU score: {:.4f}, loss: {:.4f}, epoch_time: {:.4f}".format(e, dice_scores.item(), iou_scores.item(), loss.item(), epoch_time))
        if wandb.run is not None:
            wandb.log({"train/epoch": e, 'train/dice': dice_scores.item(), 'train/iou': iou_scores.item(), 'train/loss': loss.item()}, step=e)
        scheduler.step()
        if ((e + 1) % args.val_every_n_epochs) == 0:
            with torch.no_grad():
                for j, batch in enumerate(data_loader_val):
                    inputs, labels = (batch["image"], batch["label"])
                    inputs = inputs.to(device, non_blocking=True)
                    labels = labels.to(device, non_blocking=True)
                    outputs, _, _, _ = model(inputs)
                    labels_convert = [post_label(labels[0])]
                    output_convert = [post_pred(outputs[0])]
                    dice_metric_val(y_pred=output_convert, y=labels_convert)
                    iou_metric_val(y_pred=output_convert, y=labels_convert)
                dice_scores_val, dice_not_nans_val = dice_metric_val.aggregate()
                iou_scores_val, iou_not_nans_val = iou_metric_val.aggregate()
                print("Validation - Dice score: {:.4f}, IoU score: {:.4f}".format(dice_scores_val.item(), iou_scores_val.item()))
                if wandb.run is not None:
                    wandb.log({"validate/epoch": e, 'validate/dice': dice_scores_val.item(), 'validate/iou': iou_scores_val.item()}, step=e)




if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--train_data_folder', type=str)
    parser.add_argument('--val_data_folder', type=str)
    parser.add_argument('--test_data_folder', type=str)
    parser.add_argument('--patch_size', type=int, default=16)
    parser.add_argument('--d_encoder', type=int, default=256)
    parser.add_argument('--n_heads_encoder', type=int, default=16)
    parser.add_argument('--n_encoder_layers', type=int, default=8)
    parser.add_argument('--n_decoder_layers', type=int, default=4)
    parser.add_argument('--policy_schedule', type=int, nargs='*', default=[512, 128])
    parser.add_argument('--policy_method', type=str, default='policy_net')
    parser.add_argument('--policynet_ckpt', type=str, default='policy_net_kvasir_150.pth')
    parser.add_argument('--image_size', type=int, default=512)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--wd', type=float, default=1e-5)
    parser.add_argument('--val_every_n_epochs', type=int, default=10)
    parser.add_argument('--wandb_logging', action='store_true')
    parser.add_argument('--augment', action='store_true')
    args = parser.parse_args()

    main(args)