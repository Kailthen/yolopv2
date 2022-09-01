import gradio as gr
import os

#os.system("wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt")
os.system("wget https://github.com/hustvl/YOLOP/raw/main/weights/End-to-end.pth")
#os.system("wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-e6e.pt")
#os.system("wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-e6.pt")
os.system("wget https://github.com/CAIC-AD/YOLOPv2/releases/download/V0.0.1/yolopv2.pt")

import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from lib.config import cfg

from lib.models import get_net
import torchvision.transforms as transforms
from lib.dataset.DemoDataset import LoadImages as LoadImages1
#from lib.core.general import non_max_suppression, scale_coords 
from lib.utils.plot import plot_one_box,show_seg_result as plot_one_box1,show_seg_result1
from tqdm import tqdm

from utils.functions import \
        time_synchronized,select_device, increment_path,\
        scale_coords,xyxy2xywh,non_max_suppression,split_for_trace_model,\
        driving_area_mask,lane_line_mask,plot_one_box,show_seg_result,\
        AverageMeter,\
        LoadImages
      

from PIL import Image
 

def detect(img,model):
    #with torch.no_grad()：
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=model+".pt", help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='Inference/', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--trace', action='store_true', help='trace model')
    opt = parser.parse_args()
    img.save("Inference/test.jpg")
    source, weights, view_img, save_txt, imgsz, trace = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, opt.trace
    save_img = True  # save inference images
    #webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        #('rtsp://', 'rtmp://', 'http://', 'https://'))
    #print(webcam)
    # Directories
    #save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    #(save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    #set_logging()
    device = select_device(opt.device)
    #print(device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    inf_time = AverageMeter()
    waste_time = AverageMeter()
    nms_time = AverageMeter()

    # Load model
    #model = attempt_load(weights, map_location=device)  # load FP32 model
    #stride = int(model.stride.max())  # model stride
    #imgsz = check_img_size(imgsz, s=stride)  # check img_size
    print(weights)
    if weights == 'yolopv2.pt':
        print(weights)
        stride =32
        model  = torch.jit.load(weights,map_location=device)
        model.eval()
    
        # Set Dataloader
        vid_path, vid_writer = None, None
        dataset = LoadImages(source, img_size=imgsz, stride=stride)
    
        # Run inference
        if device.type != 'cpu':
            model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
        t0 = time.time()
        for path, img, im0s, vid_cap in dataset:
            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            print(img.shape)
    
            if img.ndimension() == 3:
                img = img.unsqueeze(0)
    
            # Inference
            t1 = time_synchronized()
            [pred,anchor_grid],seg,ll= model(img)
            t2 = time_synchronized()
    
            # waste time: the incompatibility of  torch.jit.trace causes extra time consumption in demo version 
            # but this problem will not appear in offical version 
            tw1 = time_synchronized()
            pred = split_for_trace_model(pred,anchor_grid)
            tw2 = time_synchronized()
    
            # Apply NMS
            t3 = time_synchronized()
            pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
            t4 = time_synchronized()
    
            da_seg_mask = driving_area_mask(seg)
            ll_seg_mask = lane_line_mask(ll)
                
            print(da_seg_mask.shape)
            # Process detections
            for i, det in enumerate(pred):  # detections per image
              
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)
    
                p = Path(p)  # to Path
                #save_path = str(save_dir / p.name)  # img.jpg
                #txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
                s += '%gx%g ' % img.shape[2:]  # print string
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
    
                    # Print results
                    #for c in det[:, -1].unique():
                        #n = (det[:, -1] == c).sum()  # detections per class
                        #s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
    
                    # Write results
                    for *xyxy, conf, cls in reversed(det):
                        if save_txt:  # Write to file
                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                            line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
               
    
                        if save_img :  # Add bbox to image
                            plot_one_box(xyxy, im0, line_thickness=3)
    
                # Print time (inference)
                print(f'{s}Done. ({t2 - t1:.3f}s)')
                show_seg_result(im0, (da_seg_mask,ll_seg_mask), is_demo=True)
                
    if weights == 'yolop.pt':
        weights = 'End-to-end.pth'
        print(weights)
    
        normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            )
        
        transform=transforms.Compose([
                    transforms.ToTensor(),
                    normalize,
                ])
        model = get_net(cfg)
        checkpoint = torch.load(weights, map_location= device)
        #print(checkpoint)
        model.load_state_dict(checkpoint['state_dict'])
        model = model.to(device)

        dataset = LoadImages1(source, img_size=imgsz)
        bs = 1  # batch_size

        # Get names and colors
        names = model.module.names if hasattr(model, 'module') else model.names
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]
    
        # Run inference
        t0 = time.time()
    
        vid_path, vid_writer = None, None
        img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
        _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
        model.eval()
    
  
        for i, (path, img, img_det, vid_cap,shapes) in tqdm(enumerate(dataset),total = len(dataset)):
            img = transform(img).to(device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
            if img.ndimension() == 3:
                img = img.unsqueeze(0)
            # Inference
            t1 = time_synchronized()
            det_out, da_seg_out,ll_seg_out= model(img)
            t2 = time_synchronized()
            # if i == 0:
            #     print(det_out)
            inf_out, _ = det_out
            inf_time.update(t2-t1,img.size(0))
    
            # Apply NMS
            t3 = time_synchronized()
            det_pred = non_max_suppression(inf_out, conf_thres=opt.conf_thres, iou_thres=opt.iou_thres, classes=None, agnostic=False)
            t4 = time_synchronized()
    
            nms_time.update(t4-t3,img.size(0))
            det=det_pred[0]
    
            #save_path = str(save_dir +'/'+ 'img.jpg')
    
            _, _, height, width = img.shape
            h,w,_=img_det.shape
            pad_w, pad_h = shapes[1][1]
            pad_w = int(pad_w)
            pad_h = int(pad_h)
            ratio = shapes[1][0][1]
    
            da_predict = da_seg_out[:, :, pad_h:(height-pad_h),pad_w:(width-pad_w)]
            da_seg_mask = torch.nn.functional.interpolate(da_predict, scale_factor=int(1/ratio), mode='bilinear')
            _, da_seg_mask = torch.max(da_seg_mask, 1)
            da_seg_mask = da_seg_mask.int().squeeze().cpu().numpy()
            # da_seg_mask = morphological_process(da_seg_mask, kernel_size=7)
    
            
            ll_predict = ll_seg_out[:, :,pad_h:(height-pad_h),pad_w:(width-pad_w)]
            ll_seg_mask = torch.nn.functional.interpolate(ll_predict, scale_factor=int(1/ratio), mode='bilinear')
            _, ll_seg_mask = torch.max(ll_seg_mask, 1)
            ll_seg_mask = ll_seg_mask.int().squeeze().cpu().numpy()
            # Lane line post-processing
            #ll_seg_mask = morphological_process(ll_seg_mask, kernel_size=7, func_type=cv2.MORPH_OPEN)
            #ll_seg_mask = connect_lane(ll_seg_mask)
    
            img_det = show_seg_result1(img_det, (da_seg_mask, ll_seg_mask), _, _, is_demo=True)
    
            if len(det):
                det[:,:4] = scale_coords(img.shape[2:],det[:,:4],img_det.shape).round()
                for *xyxy,conf,cls in reversed(det):
                    label_det_pred = f'{names[int(cls)]} {conf:.2f}'
                    plot_one_box1(xyxy, img_det , label=label_det_pred, color=colors[int(cls)], line_thickness=2)
            
     
   
        im0 = img_det

        print('Done. (%.3fs)' % (time.time() - t0))
        print('inf : (%.4fs/frame)   nms : (%.4fs/frame)' % (inf_time.avg,nms_time.avg))
    
      
    #inf_time.update(t2-t1,img.size(0))
    #nms_time.update(t4-t3,img.size(0))
    #waste_time.update(tw2-tw1,img.size(0))
    #print('inf : (%.4fs/frame)   nms : (%.4fs/frame)' % (inf_time.avg,nms_time.avg))
    #print(f'Done. ({time.time() - t0:.3f}s)')
    #print(im0.shape)
   
    
    return Image.fromarray(im0[:,:,::-1])

   
gr.Interface(detect,[gr.Image(type="pil"),gr.Dropdown(choices=["yolopv2","yolop"])], gr.Image(type="pil"),title="Yolopv2",examples=[["example.jpeg", "yolopv2"]],description="demo for <a href='https://github.com/CAIC-AD/YOLOPv2' style='text-decoration: underline' target='_blank'>YOLOPv2</a> 🚀: Better, Faster, Stronger for Panoptic driving Perception").launch()
