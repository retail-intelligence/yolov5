# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Run YOLOv5 detection inference on images, videos, directories, globs, YouTube, webcam, streams, etc.

Usage - sources:
    $ python detect.py --weights yolov5s.pt --source 0                               # webcam
                                                     img.jpg                         # image
                                                     vid.mp4                         # video
                                                     screen                          # screenshot
                                                     path/                           # directory
                                                     list.txt                        # list of images
                                                     list.streams                    # list of streams
                                                     'path/*.jpg'                    # glob
                                                     'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                                     'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream

Usage - formats:
    $ python detect.py --weights yolov5s.pt                 # PyTorch
                                 yolov5s.torchscript        # TorchScript
                                 yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                 yolov5s_openvino_model     # OpenVINO
                                 yolov5s.engine             # TensorRT
                                 yolov5s.mlmodel            # CoreML (macOS-only)
                                 yolov5s_saved_model        # TensorFlow SavedModel
                                 yolov5s.pb                 # TensorFlow GraphDef
                                 yolov5s.tflite             # TensorFlow Lite
                                 yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
                                 yolov5s_paddle_model       # PaddlePaddle
"""

import argparse
import os
import platform
import sys
from datetime import datetime
from pathlib import Path
import cv2
import numpy as np

import torch

from utils.augmentations import letterbox

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams, LoadNumpy
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, smart_inference_mode

class YoloDetection():

    def __init__(self,
                 source=ROOT / 'data/images',  # file/dir/URL/glob/screen/0(webcam)
                 save_bbox_conf_cls=False, # save bboxes, confidences and classes to *.txt
                 view_img=False,  # show results
                 weights=ROOT / 'yolov5s.pt',  # model path or triton URL
                 device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu):
                 save_txt=False,  # save results to *.txt
                 project=ROOT / 'runs/detect',  # save results to project/name
                 name='exp',  # save results to project/name
                 exist_ok=False,  # existing project/name ok, do not increment
                 dnn=False,  # use OpenCV DNN for ONNX inference
                 data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
                 half=False,  # use FP16 half-precision inference
                 imgsz=(640, 640)):  # inference size (height, width)
        
        # Load model
        device = select_device(device)
        self.model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
        self.stride, self.names, self.pt = self.model.stride, self.model.names, self.model.pt
        self.imgsz = check_img_size(imgsz, s=self.stride)  # check image size

        self.view_img = view_img
        self.weights = weights
        self.save_txt = save_txt

        is_numpy = False

        # Directories
        self.save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
        (self.save_dir / 'labels' if self.save_txt else self.save_dir).mkdir(parents=True, exist_ok=True)  # make dir
        (self.save_dir / 'info' if save_bbox_conf_cls else self.save_dir).mkdir(parents=True, exist_ok=True) # make info dir

        # Dataloader
        is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
        is_url = str(source).lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
        self.webcam = str(source).isnumeric() or str(source).endswith('.streams') or (is_url and not is_file)
        
        bs = 1  # batch_size
        if self.webcam:    
            bs = len(self.dataset)
        
        self.model.warmup(imgsz=(1 if self.pt or self.model.triton else bs, 3, *self.imgsz))  # warmup
        self.seen, self.window, self.dt = 0, [], (Profile(), Profile(), Profile())
        

    def _load_video(self, path) -> list:
        vidcap = cv2.VideoCapture(path)
        success,image = vidcap.read()
        if not success:
            raise NameError("Could not open video.")
        img_list = []
        img_list.append(image)
        while success:
            success,image = vidcap.read()
            if success:
                img_list.append(image)
        return img_list


    def _format_data(self, np_img, stride=32, auto=True, transforms=None, vid_stride=1):
        
        img_size = np_img.shape[0]        
        stride = stride
        auto = auto
        transforms = transforms  # optional

        im0 = np_img
        if transforms:
            im = transforms(im0)  # transforms
        else:
            im = letterbox(im0, img_size, stride=stride, auto=auto)[0]  # padded resize
            im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
            im = np.ascontiguousarray(im)  # contiguous
        path = ''
        image_counter_string = ''

        return path, im, im0, None, image_counter_string
        
        pass

    @smart_inference_mode()
    def run(self,
            source=ROOT / 'data/images',  # file/dir/URL/glob/screen/0(webcam)
            save_bbox_conf_cls=False, # save bboxes, confidences and classes to *.txt
            conf_thres=0.25,  # confidence threshold
            iou_thres=0.45,  # NMS IOU threshold
            max_det=1000,  # maximum detections per image
            nosave=False,  # do not save images/videos
            save_conf=False,  # save confidences in --save-txt labels
            save_crop=False,  # save cropped prediction boxes
            classes=None,  # filter by class: --class 0, or --class 0 2 3
            agnostic_nms=False,  # class-agnostic NMS
            augment=False,  # augmented inference
            visualize=False,  # visualize features
            vid_stride=1,  # video frame-rate stride
            update=False,  # update all models
            line_thickness=3,  # bounding box thickness (pixels)
            hide_labels=False,  # hide labels
            hide_conf=False,  # hide confidences
    ):
        
        if type(source)==type(np.zeros(0)) or type(source)==type([]):
            np_source = source    
        else:
            source = str(source)
            if source.endswith('.mp4') or source.endswith('.api'):
                np_source = self._load_video(source)
        
        self.save_img = not nosave # save inference images
        
        # Dataloader
        # self.dataset = LoadNumpy(np_source, img_size=self.imgsz, stride=self.stride, auto=self.pt, vid_stride=vid_stride)

        # Run inference
        result_list = []
        
        path, im, im0s, vid_cap, s = self._format_data(np_source)
            
        with self.dt[0]:
            im = torch.from_numpy(im).to(self.model.device)
            im = im.half() if self.model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

        # Inference
        with self.dt[1]:
            visualize = increment_path(self.save_dir / Path(path).stem, mkdir=True) if visualize else False
            pred = self.model(im, augment=augment, visualize=visualize)

        # NMS
        with self.dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        date = datetime.now() # for saving time of detection
        time = f"{date.strftime('%Y%m%d%H%M%S%f')[:-3]}"
        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)
        
        # Process predictions
        for i, det in enumerate(pred):  # per image
            self.seen += 1
            p, im0, frame = path, im0s.copy(), 0

            p = Path(p)  # to Path
            save_path = str(self.save_dir / p.name)  # im.jpg
            txt_path = str(self.save_dir / 'labels' / p.stem) + f'_{frame}'  # im.txt
            info_path = str(self.save_dir / 'info' / p.stem) + f'_{str(frame).zfill(6)}'  # infos.txt
            s += '%gx%g ' % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, example=str(self.names))
            
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    s += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results

                # Saves time in .txt file containing bbox, confidences and classes
                if save_bbox_conf_cls:
                    with open(f'{info_path}','a') as f:
                        
                        f.write(f"{date.strftime('%Y%m%d%H%M%S%f')[:-3]}\n")
                        
                det_list = []
                for j, (*xyxy, conf, cls) in enumerate(reversed(det)):

                    detection = []
                    for value in range(len(det[j,:6])):
                        detection.append(det[j,:6][value].item())
                    det_list.append(detection)

                    if save_bbox_conf_cls:  # Write bbox, confidences and classes to file
                        with open(f'{info_path}','a') as f:
                            for value in range(len(det[j,:6])):
                                f.write(str(det[j,:6][value].item())+' ')
                            f.write("\n")

                    if self.save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(f'{txt_path}.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if self.save_img or save_crop or self.view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if hide_labels else (self.names[c] if hide_conf else f'{self.names[c]} {conf:.2f}')
                        annotator.box_label(xyxy, label, color=colors(c, True))
                    if save_crop:
                        save_one_box(xyxy, imc, file=self.save_dir / 'crops' / self.names[c] / f'{p.stem}.jpg', BGR=True)

            # Stream results
            im0 = annotator.result()
            if self.view_img:
                if platform.system() == 'Linux' and p not in self.window:
                    self.window.append(p)
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if self.save_img:
                if self.vid_path[i] != save_path:  # new video
                    self.vid_path[i] = save_path
                    if isinstance(self.vid_writer[i], cv2.VideoWriter):
                        self.vid_writer[i].release()  # release previous video writer
                    if vid_cap:  # video
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    else:  # stream
                        fps, w, h = 30, im0.shape[1], im0.shape[0]
                    save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                    self.vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                self.vid_writer[i].write(im0)
        if len(det):
            result_list.append(det_list)

        # Print time (inference-only)
        LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{self.dt[1].dt * 1E3:.1f}ms")

        ## END of old for loop
        
        # Print results
        t = tuple(x.t / self.seen * 1E3 for x in self.dt)  # speeds per image
        # LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *self.imgsz)}' % t)
        if self.save_txt or self.save_img:
            s = f"\n{len(list(self.save_dir.glob('labels/*.txt')))} labels saved to {self.save_dir / 'labels'}" if self.save_txt else ''
            LOGGER.info(f"Results saved to {colorstr('bold', self.save_dir)}{s}")
        if update:
            strip_optimizer(self.weights[0])  # update model (to fix SourceChangeWarning)

        return time, result_list


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save-bbox-conf-cls', action='store_true', help='save bbox, confidences and classes results to *.txt')
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s.pt', help='model path or triton URL')
    parser.add_argument('--source', type=str, default=ROOT / 'data/images', help='file/dir/URL/glob/screen/0(webcam)')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--vid-stride', type=int, default=1, help='video frame-rate stride')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt


def main(opt):
    check_requirements(exclude=('tensorboard', 'thop'))
    time, results = YoloDetection().run(
        source=opt.source,
        save_bbox_conf_cls=opt.save_bbox_conf_cls,
        data=opt.data,
        imgsz=opt.imgsz,
        conf_thres=opt.conf_thres,
        iou_thres=opt.iou_thres,
        nosave=opt.nosave,
        save_conf=opt.save_conf,
        save_crop=opt.save_crop,
        classes=opt.classes,
        agnostic_nms=opt.agnostic_nms,
    )


if __name__ == "__main__":
    opt = parse_opt()
    # img = cv2.imread('/home/eduardo/Downloads/people.jpeg')
    # opt.source = [img, img, img]
    print(f'   ===============>  {type(opt.source)}')
    main(opt)