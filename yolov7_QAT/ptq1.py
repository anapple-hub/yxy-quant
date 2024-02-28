import torch
from pytorch_quantization import quant_modules
from yolov7.models.yolo import Model

def load_yolov7_model(weight, device='cpu'):
    ckpt = torch.load(weight, map_location=device)

    pass

def prepare_dataset():
    pass

def evaluate_coco():
    pass



if __name__ == "__main__":

    weight = "yolov7.pt"
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


    model = load_yolov7_model(weight, device)

    dataloader = prepare_dataset()

    ap = evaluate_coco()