import os
import sys
sys.path.append(os.getcwd())
import  warnings
warnings.filterwarnings("ignore") 

import json
import torch 
import argparse 
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from tqdm import tqdm 

from network import shufflenetv2,mobilenetv2

def get_model_info(model,args):
    from copy import deepcopy     
    from thop import profile
    from thop.utils import clever_format 

    x = torch.zeros((1, 3, args.input_size, args.input_size), device=next(model.parameters()).device) #[1, 3, 64, 64] next 返回迭代器的下一个项
   
    flops, params  = profile(deepcopy(model),inputs=(x,),verbose=False)

    macs, params =clever_format([flops,params],"%.3f") #
    
    print('Input Size:',x.shape)
    print("MACS：", macs, ' Params:',params)
    print('Done') 
# ./models/pretrained/shufflenetv2_x0.5.pth
# './models/pretrained/mobilenetv2_0.25.pth'
def main():

    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training') 

    parser.add_argument('--net_type', type=str, default='mobilenetv2', help='backbone type')

    parser.add_argument('--lr', type=float, default=0.05, help='learning rate')

    parser.add_argument('--lrf', type=float, default=0.1, help='lrf')

    parser.add_argument('--epochs', type=int, default=100, help='epoch')

    parser.add_argument('--batch_size', type=int, default=128, help='epoch')

    parser.add_argument('--num_class', type=int, default=4, help='num class')

    parser.add_argument('--num_worker', type=int, default=4, help='num worker')

    parser.add_argument('--width_mult', type=float, default=0.25, help='MobileNet model width multiplier.')

    parser.add_argument('--input_size', type=int, default=128, help='MobileNet model input resolution') 

    parser.add_argument('--pretrained', default='./models/pretrained/mobilenetv2_0.25.pth', type=str, help='path to pretrained weight (default: none)')
    
    parser.add_argument('--save_path', type=str, default='./models/checkpoints', help='MobileNet model input resolution') 

    args = parser.parse_args()   

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))
    
    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(int(args.input_size)),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize(int(args.input_size/0.875)),
                                   transforms.CenterCrop(int(args.input_size)),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}
    
    # data_root = os.path.abspath(os.path.join(os.getcwd(), "../.."))  # get data root path
    image_path = os.path.join("data","garbage")  # flower data set path
    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)
    train_dataset = datasets.ImageFolder(root=os.path.join(image_path, "train"),
                                         transform=data_transform["train"])
    train_num = len(train_dataset)

    # nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(args.num_worker))

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_size, shuffle=True,
                                               num_workers=args.num_worker)

    validate_dataset = datasets.ImageFolder(root=os.path.join(image_path, "val"),
                                            transform=data_transform["val"]) 
    val_num = len(validate_dataset)
    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  batch_size=args.batch_size, shuffle=False,
                                                  num_workers=args.num_worker)

    print("using {} images for training, {} images for validation.".format(train_num,
                                                                           val_num))

    # create model
    if args.net_type=="shufflenetv2":
        net = shufflenetv2(num_classes=args.num_class, width_mult=args.width_mult) 
     
    elif args.net_type=="mobilenetv2":
        net = mobilenetv2(num_classes=args.num_class, width_mult=args.width_mult) 

    if args.pretrained != "":
        weights_dict = torch.load(args.pretrained, map_location=device)
        load_weights_dict = {k: v for k, v in weights_dict.items()
                                if k in  net.state_dict() if net.state_dict()[k].numel() == v.numel()}
        print(net.load_state_dict(load_weights_dict, strict=False))

    net.to(device)  #  model = torch.nn.DataParallel(model).cuda()   
    get_model_info(net,args)
    # define loss function
    loss_function = nn.CrossEntropyLoss()

    # construct an optimizer 
    # params = [p for p in net.parameters() if p.requires_grad]
    # #optimizer = optim.Adam(params, lr=args.lr)

    optimizer = optim.SGD(net.parameters(),args.lr,
                            momentum=float(0.9),
                            weight_decay=float(4e-5))
    
    import math 
    lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf  # cosine
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    best_acc = 0.0

    save_path = os.path.join(args.save_path,args.net_type,'best_model.pth')

    train_steps = len(train_loader)
    for epoch in range(args.epochs):
        # train
        net.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader, file=sys.stdout)
        for step, data in enumerate(train_bar):

            images, labels = data
            optimizer.zero_grad()
            logits = net(images.to(device))
            loss = loss_function(logits, labels.to(device))
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()

            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                     args.epochs,
                                                                     loss)
        scheduler.step()

        # validate
        net.eval()
        acc = 0.0  # accumulate accurate number / epoch
        with torch.no_grad():
            val_bar = tqdm(validate_loader, file=sys.stdout)
            for val_data in val_bar:
                val_images, val_labels = val_data
                outputs = net(val_images.to(device))
                # loss = loss_function(outputs, test_labels)
                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()

                val_bar.desc = "valid epoch[{}/{}]".format(epoch + 1,
                                                           args.epochs)
        val_accurate = acc / val_num
        print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
              (epoch + 1, running_loss / train_steps, val_accurate))

        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(net.state_dict(), save_path)

    print('Finished Training')


if __name__ == '__main__':
    main()
