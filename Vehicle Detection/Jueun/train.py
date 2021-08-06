import numpy as np

from CarPlate_utils import get_CarPlate
import os
import torch
import torch.utils.data
import torchvision
import torchvision.models.detection
import torchvision.models.detection.mask_rcnn

from engine import train_one_epoch

import presets
import utils
import postProcess

save_model_pth_name = "model_CarPlate"
save_checkpoint_pth_name = "checkpoint_CarPlate"


def get_dataset(name, image_set, transform, data_path):
    paths = {
        "images": (data_path, get_CarPlate, 7)
    }
    _above_path, ds_fn, num_classes = paths[name]

    ds = ds_fn(_above_path, image_set=image_set, transforms=transform)

    return ds, num_classes


def get_transform(train, data_augmentation):
    return presets.DetectionPresetTrain(data_augmentation) if train else presets.DetectionPresetEval()


def get_args_parser(add_help=True):
    import argparse
    parser = argparse.ArgumentParser(description='PyTorch Detection Training', add_help=add_help)

    # 커맨드 실행 옵션을 문자열로 받을 것이다.
    # 옵션을 추가하는 부분
    # <옵션이름> <기본값 지정> <옵션 도움말>

    parser.add_argument('--data-path', default='Vehicle/Car', help='dataset')
    parser.add_argument('--dataset', default='images', help='dataset')
    parser.add_argument('--model', default='fasterrcnn_resnet50_fpn', help='model')
    parser.add_argument('--device', default='cuda', help='device')
    parser.add_argument('-b', '--batch-size', default=2, type=int,
                        help='images per gpu, the total batch size is $NGPU x batch_size')
    parser.add_argument('--epochs', default = 10, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M')
    parser.add_argument('--wd', '--weight-decay', default=0.0004, type=float, metavar='W', dest='weight_decay')
    parser.add_argument('--lr-scheduler', default='multisteplr')
    parser.add_argument('--lr-step-size', default=8, type=int)
    parser.add_argument('--lr-steps', default=[16, 22], nargs='+', type=int)
    parser.add_argument('--lr-gamma', default=0.1, type=float)
    parser.add_argument('--print-freq', default=20, type=int)
    parser.add_argument('--output-dir', default='model')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, help='start epoch')
    parser.add_argument('--aspect-ratio-group-factor', default=3, type=int)
    parser.add_argument('--rpn-score-thresh', default=None, type=float)
    parser.add_argument('--trainable-backbone-layers', default=3, type=int)
    parser.add_argument('--data-augmentation', default='hflip')
    parser.add_argument('--sync-bn', dest='sync-bn', action='store_true')
    parser.add_argument('--test-only', dest='test_only', action='store_true')
    parser.add_argument('--visualize-only', dest='visualize_only', action='store_true')
    parser.add_argument('--pretained', dest='pretrained', action='store_true')
    return parser


def main(args):
    if args.output_dir:
        utils.mkdir(args.output_dir)

    device = torch.device(args.device)

    # Data loading code

    dataset, num_classes = get_dataset(args.dataset, "train", get_transform(True, args.data_augmentation),
                                       args.data_path)

    dataset_test, _ = get_dataset(args.dataset, 'test', get_transform(False, args.data_augmentation), args.data_path)

    print("Creating data loaders")

    data_loader = torch.utils.data.DataLoader(dataset, num_workers=args.workers, collate_fn=utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(dataset_test, num_workers=args.workers, collate_fn=utils.collate_fn)

    kwargs = {
        'trainable_backbone_layers': args.trainable_backbone_layers
    }
    if "rcnn" in args.model:  # 모델을 불러오는 방법... 여기에서 in은 앞에잇는 문자열이 포함되는지 여부에 따라 true false를 반환단다.
        if args.rpn_score_thresh is not None:
            kwargs["rpn_score_thresh"] = args.rpn_score_thresh

    model = torchvision.models.detection.__dict__[args.model](num_classes=num_classes, pretrained=args.pretrained,
                                                              **kwargs)
    model.to(device)
    # if args.distributed and args.sync_bn:
    #     model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    model_without_ddp = model
    # if args.distributed:
    #     model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
    #     model_without_ddp = model.module
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    args.lr_scheduler = args.lr_scheduler.lower()

    if args.lr_scheduler == 'multisteplr':
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_steps, gamma=args.lr_gamma)
    elif args.lr_scheduler == 'cosineannealinglr':
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    else:
        raise RuntimeError("Invalid lr scheduler '{}'. Only MultiStepLR and CosineAnnealingLR "
                           "are supported.".format(args.lr_scheduler))

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1

    if args.test_only:

        os.makedirs('../Vehicle/Car/test/label', exist_ok=True)
        os.makedirs('../Vehicle/Car/test/label/detection', exist_ok=True)
        os.makedirs('../Vehicle/Car/test/label/groundtruth', exist_ok=True)
        root = 'Vehicle/Car/test/annotations'
        filename = os.listdir(root)

        model.eval()
        cpu_device = torch.device("cpu")
        with torch.no_grad():
            for (images, targets), fname in zip(data_loader_test, filename):
                images = list(img.to(device) for img in images)

                if torch.cuda.is_available():
                    torch.cuda.synchronize()

                outputs = model(images)
                outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
                targets = [{k: v.to(cpu_device) for k, v in t.items()} for t in targets]

                for output, target in zip(outputs, targets):
                    content = []

                    # groundtruth
                    for label, box in zip(target['labels'], target['boxes']):
                        label = label.item()
                        box = box.numpy()

                        # print(label, 1, box[0], box[1], box[2], box[3])
                        content.append("{} {} {} {} {} {}".format(label, 1, box[0], box[1], box[2], box[3]))
                        # print(content)
                    with open('Vehicle/Car/test/label/groundtruth/' + fname[:-4] + '.txt', 'w') as f:
                        for i in range(len(content)):
                            f.write(f'{content[i]}\n')

                    content = []
                    # detection
                    for label, conf, box in zip(output['labels'], output['scores'], output['boxes']):
                        label = label.item()
                        conf = round(conf.item(), 2)
                        box = box.numpy()

                        # print(label, conf, box[0], box[1], box[2], box[3])
                        content.append("{} {} {} {} {} {}".format(label, conf, box[0], box[1], box[2], box[3]))
                        # print(content)
                    with open('Vehicle/Car/test/label/detection/' + fname[:-4] + '.txt', 'w') as f:
                        for i in range(len(content)):
                            f.write(f'{content[i]}\n')

        return

    if args.visualize_only:
        model.eval()
        cpu_device = torch.device("cpu")
        with torch.no_grad():
            image_show_count = 20
            cnt = 0
            for images, targets in data_loader_test:
                images = list(img.to(device) for img in images)

                if torch.cuda.is_available():
                    torch.cuda.synchronize()

                outputs = model(images)
                outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
                targets = [{k: v.to(cpu_device) for k, v in t.items()} for t in targets]

                print('------------------ ---------------new images--------------------------------------')

                tensor_to_pil_image = torchvision.transforms.ToPILImage()
                classes = ("__background__", "Car", "Truck", "Bus", "Etc vehicle", "Bike", "License")
                from PIL import ImageDraw, ImageFont
                # import glob
                # for i in glob.glob(os.path.join("C:/Windows/Fonts/", "*")):
                #     print(i)
                font = ImageFont.truetype(R'C:/Windows/Fonts\YuGothL.ttc', size=12)

                ascent, descent = font.getmetrics()
                font_height = ascent + descent



                for image, output, target in zip(images, outputs, targets): # 이미지 하나씩
                    image = tensor_to_pil_image(image)
                    draw = ImageDraw.Draw(image)
                    bboxes = []

                    for label,score, box in zip(output['labels'], output['scores'], output['boxes']):
                        bbox = {"label" : label.item(), "score" : score.item(), "bbox" : box.tolist()}
                        bboxes.append(bbox)

                    # print("box_infos:",bboxes)
                    # print("type:",type(bboxes))

                    score_threshold = 0.85  # 점수 임계값
                    iou_threshold = 0.5     # iou 임계값
                    result_bboxes = postProcess.nms(bboxes, score_threshold, iou_threshold) # nms 호출

                    # box 그리기
                    for result_box in result_bboxes:
                        x1, y1, x2, y2 = result_box['bbox']
                        draw.rectangle(
                            ((x1, y1), (x2, y2)),
                            outline=(255, 0, 0),
                            width=10
                        )
                        label = result_box['label']
                        score = result_box['score']
                        text = f'{classes[label]} {100 * score:.2f}%'
                        (width, height), (offset_x, offset_y) = font.font.getsize(text)
                        draw.rectangle(
                            ((x1, y1), (x1 + width, y1 + height + offset_y)),
                            outline=(0, 0, 0),
                            fill=(0, 0, 0),
                            width=10
                        )
                        draw.text(
                            (x1, y1),
                            text,
                            (255, 0, 0),
                            font=font
                        )
                        print('RED', box, classes[label], f'{100 * score:.2f}%')

                    # target - 파란색
                    for box, label in zip(target['boxes'], target['labels']):
                        box = box.tolist()
                        label = label.item()

                        draw.rectangle(
                            ((box[0], box[1]), (box[2], box[3])),
                            outline=(0, 0, 255),
                            width=1
                        )

                        text = f'{classes[label]}'
                        (width, height), (offset_x, offset_y) = font.font.getsize(text)
                        draw.rectangle(
                            ((box[0], box[3] - font_height),
                             (box[0] + width, box[3] - font_height + height + offset_y)),
                            outline=(0, 0, 0),
                            fill=(0, 0, 0),
                            width=10
                        )
                        draw.text(
                            (box[0], box[3] - font_height),
                            text,
                            (0, 255, 255),
                            font=font
                        )
                        print('BLUE', box, classes[label])

                    cnt += 1
                    if cnt == image_show_count:
                        exit()
                    image.show()
        return

    print("start training")
    for epoch in range(args.start_epoch, args.epochs):
        train_one_epoch(model, optimizer, data_loader, device, epoch, args.print_freq)
        lr_scheduler.step()
        if args.output_dir:
            checkpoint = {
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'args': args,
                'epoch': epoch
            }
            utils.save_on_master(checkpoint, os.path.join(args.output_dir, 'model_{}.pth'.format(epoch)))
            utils.save_on_master(checkpoint, os.path.join(args.output_dir, 'checkpoint.pth'))


# 이 파이썬 프로그램을 실행 시킬때 밑에 있는 if문이 가장 먼저 실행된다
# 아래와 같이 조건문을 만들면 , def main(args)가 있는 상태에서 아래 조건문을 실행함
if __name__ == "__main__":
    print("Start")

    args = get_args_parser().parse_args()  # 여기에서 command 옵션을 받는 부분
    # 여기에서 위에 있는 get args parser함수를 읽어낸다.
    # 예를 들어 ..... --visualize only--- 이런것을 이야기 하는 것임
    main(args)

