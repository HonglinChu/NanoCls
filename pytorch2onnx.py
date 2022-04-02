import os
import sys
sys.path.append(os.getcwd())

import argparse
import torch 
from network import mobilenetv2 , shufflenetv2

def main():
    
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training') 
    
    parser.add_argument('--net_type', type=str, default='mobilenetv2', help='backbone type')

    parser.add_argument('--num_class', type=int, default=4, help='num class')
    
    parser.add_argument('--onnx_path', type=str, default='./models/onnx/nanocls_mobilenetv2_garbage.onnx', help='image_path') 

    parser.add_argument('--width_mult', type=float, default=0.25, help='MobileNet model width multiplier.')

    parser.add_argument('--input_size', type=int, default=128, help='MobileNet model input resolution') 
    
    parser.add_argument('--weight', type=str, default='./models/checkpoints/mobilenetv2/best_model.pth', help='model path') 

    args = parser.parse_args()  

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 

    # create model
    if args.net_type=="mobilenetv2":
        model = mobilenetv2(num_classes=args.num_class, width_mult=args.width_mult).to(device) 
    elif args.net_type=="shufflenetv2":
        model = shufflenetv2(num_classes=args.num_class, width_mult=args.width_mult).to(device) 

    # load model weights
    model.load_state_dict(torch.load(args.weight, map_location=device))

    model.eval()

    xz= torch.randn([1,3,args.input_size,args.input_size],device=device)
    
    torch.onnx.export(model, xz, args.onnx_path, input_names=['input'], output_names=['results'],verbose=True)  

    print('Convert pytorch model to onnx model success')

    # 模型简化,否则onnx转换成ncnn会报错 
    # """
    # 命令行： python3 -m  onnxsim   input_your_mode_name  output_onnx_model
    # github: github.com/daquexian/onnx-simplifier
    # """
    import onnx 
    from onnxsim  import simplify   # if no module named 'onnxsim' , you  should run pip install onnx-simplifier  in  terminal
    filename = './models/onnx/nanocls_mobilenetv2_garbage_sim.onnx'
    simplified_model,check =simplify( args.onnx_path,skip_fuse_bn=False) #跳过融合BN层，pytorch高版本融合bn层会出错,这里设置不起作用
    onnx.save_model(simplified_model,filename)

if __name__ == '__main__':
    main()
