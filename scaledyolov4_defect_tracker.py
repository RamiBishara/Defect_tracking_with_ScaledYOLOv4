import argparse
import time
from pathlib import Path
import ntpath
import clip

ntpath.basename("a/b/c")
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import numpy as np
import json
import math

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import xyxy2xywh, xywh2xyxy, \
    strip_optimizer, set_logging, increment_path, scale_coords
from utils.plots import plot_one_box
from utils.torch_utils import select_device, time_synchronized
# from utils.roboflow import predict_image

# deep sort imports
from deep_sort import preprocessing, nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_clip_detections as gdet

from utils.yolov5 import Yolov5Engine

# from utils.yolov4 import Yolov4Engine


classes = []

names = []

MOVEMENT_THRESHOLD = 0.0002

moved_here = []
counters_detected = []
counters_moved = []


def update_tracks(tracker, detections, frame_count, save_txt, txt_path, save_img, view_img, im0, gn):
    if len(tracker.tracks):
        print("[Tracks]", len(tracker.tracks))

    # print(frame_count)

    # for d in detections:
    #   print(d.confidence)
    print("LEN DET", len(detections))
    offset = 0
    BBOXTOPRINT = [0, 0, 0, 0]
    conf = 0
    class_name = ''
    idt = 0

    for track in tracker.tracks:
        print(track.is_confirmed())
        if not track.is_confirmed() or track.time_since_update > 1:
            # offset = offset + 1
            print("Not confirmed")
            continue
        # else:
        # print(track.track_id)
        # print(track.track_id - offset)
        # if (len(detections) == len(tracker.tracks)):
        # conf = detections[track.track_id-offset].confidence
        # else:
        try:
            conf = detections[offset].confidence
        except:
            print("detection out of index")

        print("conf - ", conf)
        offset = offset + 1
        # print("test inside individual conf", conf)

        xyxy = track.to_tlbr()
        class_num = track.class_num
        bbox = xyxy
        BBOXTOPRINT = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh

        print("bbox - ", BBOXTOPRINT)

        class_name = names[int(class_num)] if opt.detection_engine == "yolov5" else class_num
        if opt.info:
            print("Tracker ID: {}, Class: {}, Conf: {}, BBox Coords (xmin, ymin, xmax, ymax): {}".format(
                str(track.track_id), class_name, conf, (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))))
        # here constr datadrame
        idt = str(track.track_id)
        if save_txt:  # Write to file
            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh

            with open(txt_path + '.txt', 'a') as f:
                f.write('frame: {}; track: {}; class: {}; bbox: {};\n'.format(frame_count, track.track_id, class_num,
                                                                              *xywh))

        if save_img or view_img:  # Add bbox to image
            label = f'{class_name} #{track.track_id}'
            plot_one_box(xyxy, im0, label=label,
                         color=get_color_for(label), line_thickness=opt.thickness)

    return BBOXTOPRINT, idt, class_name, conf


def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)


def get_color_for(class_num):
    colors = [
        "#4892EA",
        "#00EEC3",
        "#FE4EF0",
        "#F4004E",
        "#FA7200",
        "#EEEE17",
        "#90FF00",
        "#78C1D2",
        "#8C29FF"
    ]

    num = hash(class_num)  # may actually be a number or a string
    hex = colors[num % len(colors)]

    # adapted from https://stackoverflow.com/questions/29643352/converting-hex-to-rgb-value-in-python
    rgb = tuple(int(hex.lstrip('#')[i:i + 2], 16) for i in (0, 2, 4))

    return rgb


def position_moved(coordinate1: [], coordinate2: [], threshold):
    x1 = (coordinate1[0] + coordinate1[1]) / 2
    y1 = (coordinate1[2] + coordinate1[3]) / 2

    x2 = (coordinate2[0] + coordinate2[1]) / 2
    y2 = (coordinate2[2] + coordinate2[3]) / 2

    distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    print("distance " + str(distance))
    if distance < threshold:
        return False

    return True


def detect(save_img=False):  # take care of the default value for save_img

    t0 = time_synchronized()

    nms_max_overlap = opt.nms_max_overlap
    max_cosine_distance = opt.max_cosine_distance
    nn_budget = opt.nn_budget
    video_filename = opt.source
    width_vid = opt.img_size
    height_vid = opt.img_size

    print("#########")
    # initialize deep sort
    model_filename = "ViT-B/16"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    half = device != "cpu"
    model, transform = clip.load(model_filename, device=device, jit=False)
    model.eval()
    encoder = gdet.create_box_encoder(model, transform, batch_size=1, device=device)
    # calculate cosine distance metric
    metric = nn_matching.NearestNeighborDistanceMetric(
        "cosine", max_cosine_distance, nn_budget)

    # load yolov5 model here
    if opt.detection_engine == "yolov5":
        yolov5_engine = Yolov5Engine(opt.weights, device, opt.classes, opt.confidence, opt.overlap, opt.agnostic_nms,
                                     opt.augment, half)
        global names
        names = yolov5_engine.get_names()
    elif opt.detection_engine == "yolov4":
        yolov4_engine = Yolov4Engine(opt.weights, opt.cfg, device, opt.names, opt.classes, opt.confidence, opt.overlap,
                                     opt.agnostic_nms, opt.augment, half)

    # initialize tracker
    tracker = Tracker(metric)

    source, weights, view_img, save_txt, imgsz = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
    webcam = source == 'pylon' or source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://'))
    print("webcam: " + str(webcam))
    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name,
                                   exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True,
                                                          exist_ok=True)  # make dir
    json_name = path_leaf(Path(source))
    # coordinates = {"x1": 1,
    #                  "x2": 2,
    #                 "y1": 0,
    #                  "y2": 8}
    # detect_dict = {"category": "beetle",
    #                 "coordinates": coordinates,
    #                 "id": 0,
    #                 "rate": 0.98 }
    # frame_i = {"frame_index": 2,
    #             "objects": [detect_dict, detect_dict]}
    # frame_annotations = {"frame_annotations": frame_i}

    # json_data = {"video_name": source,
    #                 "width": imgsz,
    #                 "height": imgsz,
    #                 "frame_annotations": frame_annotations}
    # print(json_data)
    # with open("{}.json".format(source), "w") as f:
    #   json.dump(json_data, f, indent = 2)

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = True
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=(640, 640))
    else:
        save_img = True  # comment to false if you do not want to save the detection/tracking video and moreover, check the image size below
        dataset = LoadImages(source, img_size=(640, 640))

    frame_annotations = {}

    frame_count = 0
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    if opt.detection_engine == "yolov5":
        _ = yolov5_engine.infer(img.half() if half else img) if device.type != 'cpu' else None  # run once

    # position movement counter
    counter_move = 0
    prediction_counter = 0
    existing_frame_counter = 0
    prev_coordinate = []
    movements = []

    for path, img, im0s, vid_cap in dataset:

        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Roboflow Inference
        t1 = time_synchronized()
        p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)
        # print("Size :"+str(im0))
        # choose between prediction engines (yolov5 and roboflow)
        if opt.detection_engine == "roboflow":
            pred, classes = predict_image(im0, opt.api_key, opt.url, opt.confidence, opt.overlap, frame_count)
            pred = [torch.tensor(pred)]
        elif opt.detection_engine == "yolov5":
            print("yolov5 inference")
            pred = yolov5_engine.infer(img)
        # else:
        #  print("yolov4 inference {}".format(im0.shape))
        #  pred = yolov4_engine.infer(im0)
        #  pred, classes = yolov4_engine.postprocess(pred, im0.shape)
        #  pred = [torch.tensor(pred)]

        t2 = time_synchronized()

        detection_list = []  # see where this needs to be initialized may not be best place

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            existing_frame_counter += 1
            # moved up to roboflow inference
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(
                ), dataset.count
            # else:
            # p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            # p = Path(p)  # to Path
            save_path = str(save_dir / Path(p).name)  # img.jpg
            txt_path = str(save_dir / 'labels' / Path(p).stem) + \
                       ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt

            # normalization gain whwh
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]
            if len(det):
                prediction_counter += 1

                print("\n[Detections]")
                if opt.detection_engine == "roboflow":
                    # Print resultsq
                    clss = np.array(classes)
                    for c in np.unique(clss):
                        n = (clss == c).sum()  # detections per class
                        s += f'{n} {c}, '  # add to string

                    trans_bboxes = det[:, :4].clone()
                    bboxes = trans_bboxes[:, :4].cpu()
                    confs = det[:, 4]

                elif opt.detection_engine == "yolov4":

                    # Print results
                    # Rescale boxes from img_size to im0 size
                    # det[:, :4] = scale_coords([1,1], det[:, :4], im0.shape).round()
                    clss = np.array(classes)
                    for c in np.unique(clss):
                        n = (clss == c).sum()  # detections per class
                        s += f'{n} {c}, '  # add to string

                    # Transform bboxes from tlbr to tlwh
                    trans_bboxes = det[:, :4].clone()
                    bboxes = trans_bboxes[:, :4].cpu()
                    confs = det[:, 4]

                    """for idx, box in enumerate(bboxes):
                        plot_one_box(xywh2xyxy(torch.tensor(box).view(1, 4))[0], im0, label=classes[idx],
                                     color=get_color_for(classes[idx]), line_thickness=opt.thickness)"""

                    print(s)
                else:

                    # Print results
                    # Rescale boxes from img_size to im0 size

                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += f'{n} {names[int(c)]}s, '  # add to string

                    # Transform bboxes from tlbr to tlwh
                    trans_bboxes = det[:, :4].clone()
                    trans_bboxes[:, 2:] -= trans_bboxes[:, :2]
                    bboxes = trans_bboxes[:, :4].cpu()
                    confs = det[:, 4]
                    class_nums = det[:, -1]
                    classes = class_nums

                    print(s)

                # encode yolo detections and feed to tracker
                features = encoder(im0, bboxes)
                detections = [Detection(bbox, conf, class_num, feature) for bbox, conf, class_num, feature in zip(
                    bboxes, confs, classes, features)]

                # run non-maxima supression
                boxs = np.array([d.tlwh for d in detections])
                scores = np.array([d.confidence for d in detections])
                class_nums = np.array([d.class_num.cpu() for d in detections])
                indices = preprocessing.non_max_suppression(
                    boxs, class_nums, nms_max_overlap, scores)
                detections = [detections[i] for i in indices]

                # Call the tracker
                tracker.predict()
                tracker.update(detections)

                # update tracks
                coordinates, t_id, class_name, conf = update_tracks(tracker, detections, frame_count, save_txt,
                                                                    txt_path, save_img, view_img, im0, gn)

                print("started distance calculation")
                print("coordinate length: " + str(len(coordinates)))
                if len(prev_coordinate) == 0:
                    prev_coordinate = coordinates

                else:
                    moved = position_moved(coordinates, prev_coordinate, MOVEMENT_THRESHOLD)

                    prev_coordinate = coordinates

                    print("prev_coordinate: " + str(prev_coordinate))
                    print("coordinates: " + str(coordinates))

                    # 1. did it move : bool
                    # 2. what frame out of detected frames it moved : int
                    # 3. what frame out of existing frames it moved : int
                    movements.append([moved, prediction_counter, existing_frame_counter])
                    if moved:
                        print("has moved")
                        counter_move += 1
                    else:
                        print("didn't move")
                print("ended distance calculation")

                coord_dict = {"x1": coordinates[0],
                              "x2": coordinates[1],
                              "y1": coordinates[2],
                              "y2": coordinates[3]}

                detect_dict = {"category": class_name,
                               "coordinates": coord_dict,
                               "id": t_id,
                               "rate": conf}

                detection_list.append(detect_dict)

            # Print time (inference + NMS)
            print(f'Done. ({t2 - t1:.3f}s)')
            fps = 1 / (t2 - t1)
            print("frame index - ", frame_count)

            frame_i = {"frame_index": frame_count,
                       "objects": detection_list}

            frame_annotations[frame_count] = frame_i

            # Stream results
            print("VIEWIMG: " + str(view_img))
            if view_img:
                im0 = cv2.resize(im0, (1280, 1280))
                cv2.imshow(str(p), im0)
                # print("***", im0.shape)
                if cv2.waitKey(1) == ord('q'):  # q to quit
                    raise StopIteration

            # Save results (image with detections)
            if save_img:
                print("SAVEIMG")
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video'
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer

                        fourcc = 'mp4v'  # output video codec
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        vid_writer = cv2.VideoWriter(
                            save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
                    vid_writer.write(im0)

            frame_count = frame_count + 1

        # frames detected
        moved_here.append(movements)
        counters_detected.append(prediction_counter)
        counters_moved.append(counter_move)
    print("percentage moved %" + str(round(100 * (counter_move / prediction_counter), 2)))
    print("percentage of detected frames %" + str(round(100 * (prediction_counter / existing_frame_counter), 2)))
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        print(f"Results saved to {save_dir}{s}")

    json_data = {"video_name": source,
                 "width": imgsz,
                 "height": imgsz,
                 "fps": fps,
                 "frame_annotations": frame_annotations}

    print(json_data)
    with open("{}.json".format(source), "w") as f:
        json.dump(json_data, f, indent=2)
    print(f'Done. ({time.time() - t0:.3f}s)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str,
                        default='scaledyolov4.pt', help='model.pt path(s)')
    parser.add_argument('--cfg', type=str,
                        default='yolov4-p6.yaml', help='scaledyolov4 model cfg file path')
    parser.add_argument('--names', type=str,
                        default='coco.names', help='scaledyolov4 names file, file path')
    parser.add_argument('--source', type=str,
                        default='data/images', help='source')
    parser.add_argument('--img-size', type=int, default=640,
                        help='inference size ')
    parser.add_argument('--confidence', type=float,
                        default=0.25, help='object confidence threshold')
    parser.add_argument('--overlap', type=float,
                        default=0.50, help='IOU threshold for NMS')
    parser.add_argument('--thickness', type=int,
                        default=3, help='Thickness of the bounding box strokes')
    parser.add_argument('--device', default='',
                        help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true',
                        help='display results')
    parser.add_argument('--save-txt', action='store_true',
                        help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true',
                        help='save confidences in --save-txt labels')
    parser.add_argument('--classes', nargs='+', type=int,
                        help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true',
                        help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true',
                        help='augmented inference')
    parser.add_argument('--update', action='store_true',
                        help='update all models')
    parser.add_argument('--project', default='runs/detect',
                        help='save results to project/name')
    parser.add_argument('--name', default='exp',
                        help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true',
                        help='existing project/name ok, do not increment')
    parser.add_argument('--nms_max_overlap', type=float, default=1.0,
                        help='Non-maxima suppression threshold: Maximum detection overlap.')
    parser.add_argument('--max_cosine_distance', type=float, default=0.4,
                        help='Gating threshold for cosine distance metric (object appearance).')
    parser.add_argument('--nn_budget', type=int, default=None,
                        help='Maximum size of the appearance descriptors allery. If None, no budget is enforced.')
    parser.add_argument('--info', action='store_true',
                        help='Print debugging info.')
    parser.add_argument("--detection-engine", default="roboflow",
                        help="Which engine you want to use for object detection (scaledyoov4, yolov5, yolov4, roboflow).")
    opt = parser.parse_args()
    print(opt)

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']:
                detect(opt.source)
                strip_optimizer(opt.weights)
        else:
            detect()
