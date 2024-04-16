# YOLOv5 🚀 by Ultralytics, AGPL-3.0 license
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
                                                     'https://youtu.be/LNwODJXcvt4'  # YouTube
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
import csv
import os
import platform
import sys
from pathlib import Path
from multiprocessing import Process, Manager
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import torch

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from ultralytics.utils.plotting import Annotator, colors, save_one_box

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.torch_utils import select_device, smart_inference_mode

@smart_inference_mode()

def update_matrix(xcoor_list, matrix_of_people, head_count):

    # Sort xcoor_list
    xcoor_list = np.sort(xcoor_list)
    # Deal with empty screen case
    if len(xcoor_list) == 0: 
        #print("Empty list supplied")
        update_iteration = 0
        for i in matrix_of_people[-2,:]:
            if (i != 0) or (matrix_of_people[0,update_iteration] != 0):
                matrix_of_people[-2,update_iteration] = matrix_of_people[-2,update_iteration] + 1
            update_iteration = update_iteration + 1
        index_count = 0
        for i in matrix_of_people[-2,:]:
            if i >= 3:
                if (np.count_nonzero(matrix_of_people[:,index_count])) > 5:
                    head_count = head_count + matrix_of_people[-1, index_count]
                matrix_of_people[:, index_count] = 0
                #print("Column",index_count,"cleared")
            index_count = index_count + 1
        return matrix_of_people, head_count
    
    # Move matrix down by 1 for new coordinates to update
    for i in range(len(matrix_of_people[:,0]) - 3):
        matrix_of_people[9-i,:] = matrix_of_people[8-i,:]
    matrix_of_people[0,:] = 0

    # Assume that the xcoor_list is longer than previously recorded in matrix, find the outlier (new person) coordinate
    outlier_matrix = np.zeros((len(xcoor_list), np.count_nonzero(matrix_of_people[1,:])), dtype=float)
    #print("OUTLIER MATRIX BEFORE")
    #print(outlier_matrix)
    if len(xcoor_list) > np.count_nonzero(matrix_of_people[1,:]):
        countI = 0
        for i in xcoor_list:
            countJ = 0
            for j in np.nonzero(matrix_of_people[1,:])[0]:
                outlier_matrix[countI, countJ] = abs(i - j)
                countJ = countJ + 1
            countI = countI + 1
        #print("OUTLIER MATRIX AFTER")
        #print(outlier_matrix)
  

    
        # Elimination of outliers
        for i in range(np.count_nonzero(matrix_of_people[1,:])):
            min_index = np.argmin(outlier_matrix)
            min_row, min_col = np.unravel_index(min_index, outlier_matrix.shape)
            outlier_matrix[min_row,:] = 3
        #print("OUTLIER MATRIX EDITED")
        #print(outlier_matrix)
        removing_numbers = []
        for i in range(len(xcoor_list) - np.count_nonzero(matrix_of_people[1,:])):
            if np.count_nonzero(outlier_matrix):
                min_index = np.argmin(outlier_matrix)
                min_row, min_col = np.unravel_index(min_index, outlier_matrix.shape)
                outlier_matrix[min_row,:] = 3
                removing_numbers.append(xcoor_list[min_row])
                for j in range(len(matrix_of_people[0,:])):
                    if (matrix_of_people[1,j] == 0) and (matrix_of_people[0,j] == 0):
                        matrix_of_people[0,j] = xcoor_list[min_row]
                        if matrix_of_people[-2,j] != 0:
                            matrix_of_people[-2,j] = 0
                        #print("UPDATED NO:", xcoor_list[min_row])
                        break
        
        # remove the extra numbers from xcoor_list before proceeding for the matching
        xcoor_list = [x for x in xcoor_list if x not in removing_numbers]
        #print("LENGTH OF XCOORLIST:", len(xcoor_list))
            



    # Assign closest coordinate to its corresponding index
    taken_slot = np.zeros((len(matrix_of_people[0,:])), dtype=float)
    for i in range(len(xcoor_list)):
        differences = np.abs(xcoor_list[i] - matrix_of_people[1, :])
        for j in range(len(matrix_of_people[0,:])):
            if ((matrix_of_people[1,j] == 0) and (len(xcoor_list) <= np.count_nonzero(matrix_of_people[1,:])) or (taken_slot[j] == 1)):
                differences[j] = 3

        min_difference_index = np.argmin(differences)
        taken_slot[min_difference_index] = 1
        matrix_of_people[0, min_difference_index] = xcoor_list[i]
        if matrix_of_people[-2, min_difference_index] != 0:
            matrix_of_people[-2,min_difference_index] = 0
        #print("INPUTTING: ", xcoor_list[i])
        #print(matrix_of_people)
    # Check for missing frames / coordinates, update error count and paste value to remove motion change
    for i in range(len(matrix_of_people[0,:])):
        if (matrix_of_people[0,i] == 0) and (matrix_of_people[1,i] != 0):
            matrix_of_people[-2,i] = matrix_of_people[-2,i] + 1
            matrix_of_people[0,i] = matrix_of_people[1,i]

    # Calculate average change in motion
    
    for i in range(len(matrix_of_people[0,:])):
        sumJ = 0
        for j in range(len(matrix_of_people[:,0]) - 3):
            sumJ = sumJ + (matrix_of_people[9-j,i] - matrix_of_people[8-j,i])
        if (sumJ / (len(matrix_of_people[:,0]) - 3)) < 0:
            matrix_of_people[-1,i] = -1
        elif (sumJ / (len(matrix_of_people[:,0]) - 3)) > 0:
            matrix_of_people[-1,i] = 1
        else:
            matrix_of_people[-1,i] = 0

    # Check for over certain amount of repetition before deleting the entire column
    index_count = 0
    for i in matrix_of_people[-2,:]:
        if i >= 3:
            if (np.count_nonzero(matrix_of_people[:,index_count])) > 5:
                head_count = head_count + matrix_of_people[-1, index_count]
            matrix_of_people[:, index_count] = 0
            #print("Column",index_count,"cleared")
        index_count = index_count + 1
    
    



    return matrix_of_people, head_count

def checkpoint_detection(top_row, bottom_row):
    global previous_set_top
    global previous_set_bot
    
    # Identify valid elements in the previous_set_top
    valid_indices = np.nonzero(previous_set_top)[0]

    # If there are valid elements, proceed
    if valid_indices.size > 0:
        # Find the closest matching value between present and past arrays
        closest_value_indices = np.argmin(np.abs(previous_set_top[valid_indices, None] - top_row), axis=0)

        # Update previous_set_bot with the corresponding motions
        previous_set_bot[valid_indices[closest_value_indices]] = bottom_row

        # Identify left-out indices
        left_out_indices = set(valid_indices) - set(valid_indices[closest_value_indices])
        
        # Use all indices if there are no valid indices
        left_out_indices = left_out_indices if left_out_indices else set(range(len(previous_set_bot)))

        # Use left-out indices to get the corresponding motions
        left_out_motions = previous_set_bot[list(left_out_indices)]

        # Update previous_set_bot for left-out coordinates (assumed direction is 0)
        for left_out_index in left_out_indices:
            previous_set_bot[left_out_index] = 0
    else:
        # All elements in previous_set_top are zero, update previous_set_bot with the entire bottom_row
        previous_set_bot = bottom_row
        left_out_motions = []

    # Update global variables
    previous_set_top = top_row

    # Return the array of motions for left-out coordinates
    return left_out_motions

def run(
        update_matrix,
        matrix_of_people,
        head_count,
        weights=ROOT / 'yolov5s.pt',  # model path or triton URL
        source=ROOT / 'data/images',  # file/dir/URL/glob/screen/0(webcam)
        data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=True,  # show results
        save_txt=True,  # save results to *.txt
        save_csv=False,  # save results in CSV format
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        vid_stride=1,  # video frame-rate stride
):
    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.streams') or (is_url and not is_file)
    screenshot = source.lower().startswith('screen')
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    bs = 1  # batch_size
    if webcam:
        view_img = check_imshow(warn=True)
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        bs = len(dataset)
    elif screenshot:
        dataset = LoadScreenshots(source, img_=imgsz, stride=stride, auto=pt)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    # Run inference
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
    for path, im, im0s, vid_cap, s in dataset:
        with dt[0]:
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

        # Inference
        with dt[1]:
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            pred = model(im, augment=augment, visualize=visualize)

        # NMS
        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # Define the path for the CSV file
        csv_path = save_dir / 'predictions.csv'

        # Create or append to the CSV file
        def write_to_csv(image_name, prediction, confidence):
            data = {'Image Name': image_name, 'Prediction': prediction, 'Confidence': confidence}
            with open(csv_path, mode='a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=data.keys())
                if not csv_path.is_file():
                    writer.writeheader()
                writer.writerow(data)

        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            #s += '%gx%g ' % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                xcoor_list = []
                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)  # integer class
                    label = names[c] if hide_conf else f'{names[c]}'
                    confidence = float(conf)
                    confidence_str = f'{confidence:.2f}'

                    

                    if save_csv:
                        write_to_csv(p.name, label, confidence_str)

                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(f'{txt_path}.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')
                        
                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        annotator.box_label(xyxy, label, color=colors(c, True))
                        # Extracting data =================================================================================================================
                        difference_len = 10
                        motion_boundary = 0.02
                        test_points = 2
                        object_name = names[c]
                        current_motion = 0 # 0=stationary / 1=left / 2=right
                        interval_boundary = 0.1
                        if object_name == 'person':
                            global prevX
                            global diff
                            global latest_tested_coor

                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                            [xxxxx,yyyyy,wwwww,hhhhh] = xywh
                            xcoor_list.append(xxxxx)
                            xy = [xxxxx, yyyyy]
                            #print(xy)
                            if len(diff) < difference_len:
                                diff.append(xxxxx - prevX)
                            else:
                                diff.pop(0)
                                diff.append(xxxxx - prevX)
                            avg = sum(diff) / len(diff)
                            if avg < (motion_boundary/2) and avg > (motion_boundary/-2):
                                #print("Stationary")
                                current_motion = 0
                            elif avg < 0:
                                #print("Left")
                                current_motion = 1
                            else:
                                #print("Right")
                                current_motion = 2
                            prevX = xxxxx

                            # Identify entrance and exit
                            interval_point = 1 / (test_points + 1)
                            for tp_range in range(test_points):
                                if xxxxx < ((interval_point*(tp_range+1))+(interval_boundary/2)) and xxxxx > ((interval_point*(tp_range+1))-(interval_boundary/2)) and latest_tested_coor != (interval_point*(tp_range+1)):
                                    latest_tested_coor = interval_point * (tp_range+1)
                                    #print("Tested_Point:",interval_point*(tp_range+1))
                                    '''
                                    if current_motion == 0:
                                        print("Stationary")
                                    elif current_motion == 1:
                                        print("Left")
                                    else:
                                        print("Right")
                                    '''
                        # END EDIT ===============================================================================================
                    if save_crop:
                        save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)
            else:
                xcoor_list = []
            
            # Stream results
            im0 = annotator.result()
            if view_img:
                if platform.system() == 'Linux' and p not in windows:
                    windows.append(p)
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[i].write(im0)

        # Print time (inference-only)
        #LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")
    
        # ADDED CODE RUN HERE ================================================
        print("xcoor_list:",xcoor_list)
        print("LEN:", len(xcoor_list))
        matrix_of_people, head_count = update_matrix(xcoor_list, matrix_of_people, head_count)
        #print(xcoor_list)
        print(matrix_of_people)
        print("Head Count:", head_count)

        # Filter code to only show top and bottom row (current coordinate & direction)
        #top_row = matrix_of_people[0, :]
        #bottom_row = matrix_of_people[-1, :]
        #left_out_motions = checkpoint_detection(top_row, bottom_row)
        #print("left out motion: ",left_out_motions)
        # END EDIT ==========================================================

    # Print results
    t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights[0])  # update model (to fix SourceChangeWarning)


def parse_opt():
    parser = argparse.ArgumentParser()
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
    parser.add_argument('--save-csv', action='store_true', help='save results in CSV format')
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
    global prevX
    global diff
    global latest_tested_coor
    global matrix_of_people
    global consecutive_empty_iterations
    global previous_set_top
    global previous_set_bot
    xcoor_list = []
    head_count = 0
    prevX = 0
    diff= []
    latest_tested_coor = 69420
    matrix_of_people = matrix_of_people = np.zeros((12, 11), dtype=float)
    consecutive_empty_iterations = {}
    previous_set_top = np.array([], dtype=float)
    previous_set_bot = []
    check_requirements(ROOT / 'requirements.txt', exclude=('tensorboard', 'thop'))

    run(update_matrix=update_matrix, matrix_of_people=matrix_of_people, head_count=head_count, **vars(opt))

if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
