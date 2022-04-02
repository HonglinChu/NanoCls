import os
import argparse
import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

from network import mobilenetv2, shufflenetv2

class_indict={
    "0": "其他垃圾",
    "1": "厨余垃圾",
    "2": "可回收物",
    "3": "有害垃圾"
    }

def get_model_info(model,args):
    from copy import deepcopy     
    from thop import profile
    from thop.utils import clever_format 

    x = torch.zeros((1, 3, args.input_size, args.input_size), device=next(model.parameters()).device)
   
    flops, params  = profile(deepcopy(model),inputs=(x,),verbose=False)

    macs, params =clever_format([flops,params],"%.3f") #
    
    print('Input Size:',x.shape)
    print("MACS：", macs, ' Params:',params)
    print('Done')  

def main():
    
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training') 

    parser.add_argument('--net_type', type=str, default='mobilenetv2', help='backbone type')

    parser.add_argument('--num_class', type=int, default=4, help='num class')
    
    parser.add_argument('--image_path', type=str, default='./datasets/images/val/0', help='image_path')

    parser.add_argument('--width_mult', type=float, default=0.25, help='MobileNet model width multiplier.')

    parser.add_argument('--input_size', type=int, default=128, help='MobileNet model input resolution') 
    
    parser.add_argument('--weight', type=str, default='./models/checkpoints', help='model path') 

    args = parser.parse_args()  

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_transform = transforms.Compose(
        [transforms.Resize(int(args.input_size/0.875)),
         transforms.CenterCrop(int(args.input_size)),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    
    # create model
    if args.net_type=="mobilenetv2":
        model = mobilenetv2(num_classes=args.num_class, width_mult=args.width_mult).to(device) 
    elif args.net_type=="shufflenetv2":
        model = shufflenetv2(num_classes=args.num_class, width_mult=args.width_mult).to(device) 

    # load model weights
    model.load_state_dict(torch.load(os.path.join(args.weight,args.net_type,'best_model.pth'), map_location=device))
    model.eval()

    get_model_info(model,args)

    image_list=os.listdir(args.image_path)
    image_list.sort()
    for img_name in image_list:

        img = Image.open(os.path.join(args.image_path,img_name))

        #plt.imshow(img)
        # [N, C, H, W]
        img = data_transform(img)
        # expand batch dimension
        img = torch.unsqueeze(img, dim=0)

        with torch.no_grad():
            # predict class
            output = torch.squeeze(model(img.to(device))).cpu()
            predict = torch.softmax(output, dim=0)
            predict_cla = torch.argmax(predict).numpy()

        res = "{} class: {}   prob: {:.3}".format(img_name,class_indict[str(predict_cla)],
                                                    predict[predict_cla].numpy())
        print(res)

if __name__ == '__main__':
    main()
