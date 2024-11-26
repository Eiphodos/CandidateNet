import torch
import monai
import time
import wandb
from argparse import ArgumentParser

from oracle.models.transformer_segmenter import MultiResSegmenter
from oracle.image_utils import create_oracle_labels
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
              #monai.transforms.BorderPadd(keys=['image', 'label'], spatial_border=80),
              monai.transforms.Resized(keys=["image"], spatial_size=(512, 512), mode='bilinear'),
              monai.transforms.Resized(keys=["label"], spatial_size=(512, 512), mode='nearest'),
              monai.transforms.ToTensord(keys=["image", "label"])]
    transforms = monai.transforms.Compose(transform_list)

    train_dataset = monai.data.Dataset(train_data, transform=transforms)
    val_dataset = monai.data.Dataset(val_data, transform=transforms)

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
        batch_size=1,
        num_workers=8,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True
        )
    
    ### MODEL ###
    model = MultiResSegmenter(image_size=(args.image_size, args.image_size),
                          patch_size=args.patch_sizes[0],
                          channels=3,
                          n_layers_encoder=args.n_encoder_layers,
                          d_encoder=args.d_encoder,
                          n_heads_encoder=args.n_heads_encoder,
                          n_layers_decoder=args.n_decoder_layers,
                          d_decoder=args.d_decoder,
                          n_heads_decoder=args.n_heads_decoder,
                          n_scales=len(args.patch_sizes),
                          n_cls=2)

    model = model.to(device)
    print(model)
    print("Total parameters: {}".format(count_parameters(model)))
    print("Encoder parameters: {}".format(count_parameters(model.encoder)))
    print("Decoder parameters: {}".format(count_parameters(model.decoder)))

    ### OPTIMIZER/CRITERION ###
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
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
            oracle_labels_multires = []
            for ps in args.patch_sizes:
                ol = create_oracle_labels(labels, ps)
                oracle_labels_multires.append(ol.squeeze())
            outputs, _, _ = model(inputs, oracle_labels_multires)
            loss = criterion(outputs, labels)
            loss = loss / args.grad_accum_steps
            loss.backward()

            if (i + 1) % args.grad_accum_steps == 0:
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
                    oracle_labels_multires = []
                    for ps in args.patch_sizes:
                        ol = create_oracle_labels(labels, ps)
                        oracle_labels_multires.append(ol.squeeze())
                    outputs, _, _ = model(inputs, oracle_labels_multires)
                    labels_convert = [post_label(labels[0])]
                    output_convert = [post_pred(outputs[0])]
                    dice_metric_val(y_pred=output_convert, y=labels_convert)
                    iou_metric_val(y_pred=output_convert, y=labels_convert)
                dice_scores_val, dice_not_nans_val = dice_metric_val.aggregate()
                iou_scores_val, iou_not_nans_val = iou_metric_val.aggregate()
                print("Validation - Dice score: {:.4f}, IoU score: {:.4f}".format(dice_scores_val.item(), iou_scores_val.item()))
                if wandb.run is not None:
                    wandb.log({"validate/epoch": e, 'validate/dice': dice_scores.item(), 'validate/iou': iou_scores.item()}, step=e)




if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--train_data_folder', type=str)
    parser.add_argument('--val_data_folder', type=str)
    parser.add_argument('--test_data_folder', type=str)
    parser.add_argument('--patch_sizes', type=int, nargs='*', default=[64, 32, 16, 8, 4])
    parser.add_argument('--d_encoder', type=int, nargs='*', default=[256, 128, 128, 128, 128])
    parser.add_argument('--n_heads_encoder', type=int, nargs='*', default=[16, 8, 8, 8, 8])
    parser.add_argument('--n_encoder_layers', type=int, nargs='*', default=[1, 1, 1, 1, 4])
    parser.add_argument('--d_decoder', type=int, default=128)
    parser.add_argument('--n_decoder_layers', type=int, default=4)
    parser.add_argument('--n_heads_decoder', type=int, default=4)
    parser.add_argument('--grad_accum_steps', type=int, default=8)
    parser.add_argument('--image_size', type=int, default=512)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--val_every_n_epochs', type=int, default=10)
    parser.add_argument('--wandb_logging', action='store_true')
    args = parser.parse_args()

    main(args)