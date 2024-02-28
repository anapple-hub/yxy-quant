import torch
import os, argparse
import quantize
from torch.cuda import amp
import torch.optim as optim
from pytorch_quantization import nn as quant_nn
import rules
from typing import Callable
from copy import deepcopy
def run_finetune(args, model, train_loader, val_loader, supervision_policy: Callable=None, fp16=True):
     
    summary = quantize.SummaryTool("finetune.json")
    # 训练的准备工作
    origin_model = deepcopy(model).eval()
    quantize.disable_quantization(origin_model).apply()
    
    
    model.train()
    model.requires_grad_(True)
    
    
    scaler = amp.GradScaler(enabled=fp16)  # fp16
    
    optimizer = optim.SGD(model.parameters(), lr=args.lr,)  # 优化器
    
    quant_lossfn = torch.nn.MSELoss()  # 损失函数
    
    device = next(model.parameters()).device
    
    lrschedule = {
        0: 1e-6,
        3: 1e-5,
        8: 1e-6
    }
    
    # hook 函数
    def make_layer_forward_hook(l):
        def forward_hook(m, input, output):
            l.append(output)
        return forward_hook
    
    
    # model & origin_model ==> supervision pairs
    supervision_module_pairs = []
    for (mname, ml), (oriname, ori) in zip(model.named_modules(), origin_model.named_modules()):
        if isinstance(ml, quant_nn.TensorQuantizer):
            continue
        
        if supervision_policy:
            if not supervision_policy(mname, ml):
                continue
        
        supervision_module_pairs.append([ml, ori])
            
            
    
    
    
    # 循环epoch
    best_ap = 0.
    for epoch in range(args.num_epoch):

        # 动态学习率
        if epoch in lrschedule:
            learning_rate = lrschedule[epoch]
            for g in optimizer.param_groups:
                g['lr'] = learning_rate
        
        model_outputs = []
        origin_outputs = []
        remove_handle = []
                
        for ml, ori in supervision_module_pairs:
            remove_handle.append(ml.register_forward_hook(make_layer_forward_hook(model_outputs)))
            remove_handle.append(ori.register_forward_hook(make_layer_forward_hook(origin_outputs)))        


        # 训练
        model.train()
        
        for idx_batch, datas in enumerate(train_loader):
            if idx_batch >= args.iters:
                break
            
            imgs = datas[0].to(device).float() / 255.0
            
            with amp.autocast(enabled=fp16):
                model(imgs)
                
                # origin model inference
                with torch.no_grad():
                    origin_model(imgs)
        
                # 计算量化损失
                quant_loss = 0
                for index, (mo, fo) in enumerate(zip(model_outputs, origin_outputs)):
                    quant_loss += quant_lossfn(mo, fo)
                
                model_outputs.clear()
                origin_outputs.clear()
        
            if fp16:
                scaler.scale(quant_loss).backward()
                scaler.step(optimizer)
                scaler.update()
        
            else:
                quant_loss.backward()
                optimizer.step()
            
            optimizer.zero_grad()
            
            print(f"QAT Finetuning {epoch + 1} / {args.num_epoch}, Loss: {quant_loss.detach().item():.5f}, LR: {learning_rate:g}")
        
        
        # 移除handle
        for rm in remove_handle:
            rm.remove()
        
        # 模型验证
        ap = quantize.evaluate_coco(model, val_loader)
        summary.append([f"QAT{epoch}", ap])
        
        if ap > best_ap:
            print(f"Save qat model to {args.qat} @ {ap:.5f}")
            best_ap = ap
            quantize.export_ptq(model, "qat_yolov7.onnx", device)
        
        
        
        

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--weights', type=str, default='yolov7.pt', help='initial weights path')
    parser.add_argument('--cocodir', type=str,  default="dataset/coco2017", help="coco directory")
    parser.add_argument('--batch_size', type=int,  default=8, help="batch size for data loader")
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--num_epoch', type=int, default=10, help=' max epoch for finetune')
    parser.add_argument("--iters", type=int, default=200, help="iters per epoch")
    parser.add_argument('--lr', type=float, default=1e-5, help=' learning rate for QAT finetune')

    
    parser.add_argument("--ignore_layers", type=str, default="model\.105\.m\.(.*)", help="regx")
    
    parser.add_argument("--save_ptq", type=bool, default=False, help="file")
    parser.add_argument("--ptq", type=str, default="ptq_yolov7.onnx", help="file")
    
    parser.add_argument("--save_qat", type=bool, default=False, help="file")
    parser.add_argument("--qat", type=str, default="qat_yolov7.onnx", help="file")
    
    parser.add_argument("--confidence", type=float, default=0.001, help="confidence threshold")
    parser.add_argument("--nmsthres", type=float, default=0.65, help="nms threshold")
    
    parser.add_argument("--eval_origin", action="store_true", help="do eval for origin model")
    parser.add_argument("--eval_ptq", action="store_true", help="do eval for ptq model")
    parser.add_argument("--eval_qat", action="store_true", help="do eval for qat model")
    
    parser.add_argument("--eval_summary", type=str, default="eval_summary.json", help="all evaluate data are saved in the summary save file")
    
    args = parser.parse_args()
    
    is_cuda = (args.device != 'cpu') and torch.cuda.is_available()
    device = torch.device("cuda:1" if is_cuda else "cpu")
    
    
    # prepare model
    print("Prepare Model ....")
    model = quantize.prepare_model(args.weights, device)
    quantize.replace_to_quantization_model(model, args.ignore_layers)
    
    
    # prepare dataset
    print("Prepare Dataset ....")
    val_dataloader = quantize.prepare_val_dataset(args.cocodir, batch_size=args.batch_size)
    train_dataloader = quantize.prepare_train_dataset(args.cocodir, batch_size=args.batch_size)
    
    # 在标定前将scale的工作给做掉
    rules.apply_custom_rules_to_quantizer(model, device)
    
    # calibration model
    print("Begining Calibration ....")
    quantize.calibrate_model(model, train_dataloader, device)

    
    summary = quantize.SummaryTool(args.eval_summary)
    if args.eval_origin:
        print("Evaluate Origin...")
        with quantize.disable_quantization(model):
            ap = quantize.evaluate_coco(model, val_dataloader, conf_thres=args.conf_thres, iou_thres=args.iou_thres)
            summary.append(["Origin", ap])
    if args.eval_ptq:
        print("Evaluate PTQ...")
        ap = quantize.evaluate_coco(model, val_dataloader, conf_thres=args.conf_thres, iou_thres=args.iou_thres)
        summary.append(["PTQ", ap])

    if args.save_ptq:
        print("Export PTQ...")
        quantize.export_ptq(model, args.ptq, device)
    
    
    # 判断传入的模块是否需要在QAT训练期间计算损失
    def supervision_policy():
        supervision_list = []
        for item in model.model:
            supervision_list.append(id(item))
        
        supervision_stride = 1
        keep_idx = list(range(0, len(model.model) - 1, supervision_stride))
        keep_idx.append(len(model.model) - 2)
        
        def impl(name, module):
            if id(module) not in supervision_list:
                return False
            
            idx = supervision_list.index(id(module))
            if idx in keep_idx:
                print(f"Supervision: {name} will compute loss with origin model during QAT training...")
            else:
                print(f"Supervision: {name} not compute loss during QAT training...")
                
                
            return idx in keep_idx  # True/False
        
        
        return impl
    
    
    
    print("Begining Finetune ....")
    
    run_finetune(args, model, train_dataloader, val_dataloader, supervision_policy=supervision_policy())
    
    print("QAT Finished ....")








