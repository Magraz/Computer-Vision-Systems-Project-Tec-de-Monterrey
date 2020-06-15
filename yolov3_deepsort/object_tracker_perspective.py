import time, random
import numpy as np
from absl import app, flags, logging
from absl.flags import FLAGS
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from scipy.spatial.distance import pdist, squareform

from yolov3_tf2.models import (
    YoloV3, YoloV3Tiny
)
from yolov3_tf2.dataset import transform_images
from yolov3_tf2.utils import draw_outputs, convert_boxes

from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
from PIL import Image

flags.DEFINE_string('classes', './data/labels/coco.names', 'path to classes file')
flags.DEFINE_string('weights', './weights/yolov3.tf',
                    'path to weights file')
flags.DEFINE_boolean('tiny', False, 'yolov3 or yolov3-tiny')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_string('video', './data/video/test.mp4',
                    'path to video file or number for webcam)')
flags.DEFINE_string('output', None, 'path to output video')
flags.DEFINE_string('output_format', 'XVID', 'codec used in VideoWriter when saving video to file')
flags.DEFINE_integer('num_classes', 80, 'number of classes in the model')


# globals
mouse_pts = None
canvas_img = None
temp_img = None

def get_mouse_points(event, x, y, flags, param):
    global  mouse_pts, canvas_img, temp_img
    temp_img = canvas_img.copy()

    if event == cv2.EVENT_LBUTTONDOWN:
        mouseX, mouseY = x, y
        if len(mouse_pts) < 4:
            cv2.circle(canvas_img, (x, y), 2, (255, 90, 50), 4, cv2.LINE_AA)
        elif len(mouse_pts) < 6:
            cv2.circle(canvas_img, (x, y), 2, (25, 255, 240), 4, cv2.LINE_AA)
        else:
            cv2.circle(canvas_img, (x, y), 2, (50, 255, 75), 4, cv2.LINE_AA)
        mouse_pts.append((x, y))

    if len(mouse_pts) % 2 != 0:
        if len(mouse_pts) < 4:
            cv2.line(temp_img, mouse_pts[-1], (x, y), (255, 90, 50), 2, cv2.LINE_AA)
        elif len(mouse_pts) < 6:
            cv2.line(temp_img, mouse_pts[-1], (x, y), (25, 255, 240), 2, cv2.LINE_AA)
        else:
            cv2.rectangle(temp_img, mouse_pts[-1], (x, y), (50, 255, 75), 2, cv2.LINE_AA)

    elif len(mouse_pts) != 0 and event == cv2.EVENT_LBUTTONDOWN:
        if len(mouse_pts) <= 4:
            cv2.line(canvas_img, mouse_pts[-2], (x, y), (255, 90, 50), 2, cv2.LINE_AA)
        else:
            cv2.line(canvas_img, mouse_pts[-2], (x, y), (25, 255, 240), 2, cv2.LINE_AA)


def euclidean(point1, point2):
    point1 = np.asarray(point1)
    point2 = np.asarray(point2)

    sq = (point1 - point2) ** 2
    euc = np.sum(sq, axis=0) ** 0.5
    return int(euc)


def main(_argv):
    global  mouse_pts, canvas_img, temp_img

    # Definition of the parameters
    max_cosine_distance = 0.5
    nn_budget = None
    nms_max_overlap = 1.0
    pure_yolo = False
    
    #initialize deep sort
    output_name = FLAGS.output
    model_filename = 'model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)

    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

    if FLAGS.tiny:
        yolo = YoloV3Tiny(classes=FLAGS.num_classes)
    else:
        yolo = YoloV3(classes=FLAGS.num_classes)

    yolo.load_weights(FLAGS.weights)
    logging.info('weights loaded')

    class_names = [c.strip() for c in open(FLAGS.classes).readlines()]
    logging.info('classes loaded')

    try:
        vid = cv2.VideoCapture(int(FLAGS.video))
    except:
        vid = cv2.VideoCapture(FLAGS.video)

    out = None

    if output_name:
        # by default VideoCapture returns float instead of int
        width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(vid.get(cv2.CAP_PROP_FPS))
        codec = cv2.VideoWriter_fourcc(*FLAGS.output_format)
        out = cv2.VideoWriter(output_name, codec, fps, (width + width // 3 + 3, height))
        list_file = open('detection.txt', 'w')
        frame_index = -1 

    fps = 0.0
    count = 0
    setup = False

    while True:
        _, img = vid.read()
        image = img.copy()
        h_image, w_image = image.shape[:2]

        if img is None:
            logging.warning("Empty Frame")
            time.sleep(0.1)
            count+=1
            if count < 3:
                continue
            else: 
                break

        if not setup:
            cv2.namedWindow("input_image")
            cv2.setMouseCallback("input_image", get_mouse_points)

            # setup for UI
            canvas_img = image.copy()
            temp_img = canvas_img.copy()
            h_temp, w_temp = temp_img.shape[:-1]
            mouse_pts = []

            # UI loop -------------------------------------------------------------------
            while True:
                cv2.imshow("input_image", temp_img)
                key = cv2.waitKey(1)
                if len(mouse_pts) < 4:
                    cv2.putText(temp_img, 'Selecciona dos lienas paralelas en un plano', (w_temp // 20, h_temp // 15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 1, cv2.LINE_AA)
                elif len(mouse_pts) < 6:
                    cv2.putText(temp_img, 'Selecciona la altura de una persona', (w_temp // 20, h_temp // 15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 1, cv2.LINE_AA)
                else:
                    cv2.putText(temp_img, 'Selecciona el area que quieres monitorear', (w_temp // 20, h_temp // 15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 1, cv2.LINE_AA)
                    
                if len(mouse_pts) == 8 or key == ord('q'):
                    cv2.destroyWindow("input_image")
                    break
            # ----------------------------------------------------------------------------

            cv2.namedWindow("Worker Monitoring", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Worker Monitoring", 1000, 700)

            # points of reference and ROI chosen by UI
            ref_pts = np.array(mouse_pts[:4])
            ref_len_pts = np.array(mouse_pts[4:6])
            roi = mouse_pts[6:]

            # length between reference points
            h_dst = max(euclidean(ref_pts[0], ref_pts[1]), euclidean(ref_pts[2], ref_pts[3]))
            w_dst = max(euclidean(ref_pts[0], ref_pts[2]), euclidean(ref_pts[1], ref_pts[3]))

            ref_len = euclidean(ref_len_pts[0], ref_len_pts[1])

            # calculating parallel vectors of lines
            comp_pts = np.array(mouse_pts)
            comp_pts = comp_pts * (-1)

            v_1 = np.zeros(2, dtype='float')
            v_1 = comp_pts[1] - comp_pts[0]
            mag = np.sqrt(np.sum(np.square(v_1, dtype='float')))
            v_1 = v_1 * (1 / mag)
            v_1 = np.array([v_1[1], -v_1[0]], dtype='float')
            c_1 = np.zeros(2, dtype='float')
            c_1[0] = comp_pts[1][0] + v_1[0] * w_dst
            c_1[1] = comp_pts[1][1] + v_1[1] * w_dst
            c_1[0] = c_1[0] * (-1)
            c_1[1] = c_1[1] * (-1)

            v_2 = np.zeros(2, dtype='float')
            v_2 = comp_pts[0] - comp_pts[1]
            mag = np.sqrt(np.sum(np.square(v_2, dtype='float')))
            v_2 = v_2 * (1 / mag)
            v_2 = np.array([-v_2[1], v_2[0]], dtype='float')
            c_2 = np.zeros(2, dtype='float')
            c_2[0] = comp_pts[0][0] + v_2[0] * w_dst
            c_2[1] = comp_pts[0][1] + v_2[1] * w_dst
            c_2[0] = c_2[0] * (-1)
            c_2[1] = c_2[1] * (-1)

            # getting the transformation matrix between the original reference
            # and the perpendicular "corrected" points
            dst = [ref_pts[0], ref_pts[1], c_2, c_1]
            dst = np.array(dst)
            dst = np.array(dst, dtype='float32')
            ref_pts = np.array(ref_pts, dtype='float32')
            M = cv2.getPerspectiveTransform(ref_pts, dst)

            corner_pts = np.array([roi[0], [roi[1][0], roi[0][1]], roi[1], [roi[0][0], roi[1][1]]] , dtype='float32').reshape(-1, 1, 2)
            transformed_corner_pts = cv2.perspectiveTransform(corner_pts, M)

            # calculating transformation matrix with translation correction
            [xmin, ymin] = np.int32(transformed_corner_pts.min(axis=0).ravel() - 0.5)
            [xmax, ymax] = np.int32(transformed_corner_pts.max(axis=0).ravel() + 0.5)
            t = [-(xmin), -(ymin)]
            Ht = np.array([[1,0,t[0]],[0,1,t[1]],[0,0,1]])
            new_M = Ht.dot(M)

            setup = True

        img_in = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
        img_in = tf.expand_dims(img_in, 0)
        img_in = transform_images(img_in, FLAGS.size)

        t1 = time.time()
        boxes, scores, classes, nums = yolo.predict(img_in)
        classes = classes[0]
        names = []
        for i in range(len(classes)):
            names.append(class_names[int(classes[i])])
        names = np.array(names)
        converted_boxes = convert_boxes(img, boxes[0])
        features = encoder(img, converted_boxes)    
        detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in zip(converted_boxes, scores[0], names, features)]
        
        #initialize color map
        cmap = plt.get_cmap('tab20b')
        colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

        # run non-maxima suppresion
        boxs = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        classes = np.array([d.class_name for d in detections])
        indices = preprocessing.non_max_suppression(boxs, classes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]        

        # Call the tracker
        tracker.predict()
        tracker.update(detections)

        obj_pts = []
        obj_classes = []

        if not pure_yolo:
            for track in tracker.tracks:
                if not track.is_confirmed() or track.time_since_update > 1:
                    continue
                bbox = track.to_tlbr()
                class_name = track.get_class()
                color = colors[int(track.track_id) % len(colors)]
                color = [i * 255 for i in color]
                # cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
                # cv2.rectangle(img, (int(bbox[0]), int(bbox[1]-30)), (int(bbox[0])+(len(class_name)+len(str(track.track_id)))*17, int(bbox[1])), color, -1)
                # cv2.putText(img, class_name + "-" + str(track.track_id),(int(bbox[0]), int(bbox[1]-10)),0, 0.75, (255,255,255),2)

                cntr_x = (bbox[0] + bbox[2]) / 2
                cntr_y = (bbox[1] + bbox[3]) / 2
                obj_pts.append([cntr_x, cntr_y])
                obj_classes.append(str(class_name))
        else:
            ## UNCOMMENT BELOW IF YOU WANT CONSTANTLY CHANGING YOLO DETECTIONS TO BE SHOWN ON SCREEN
            for det in detections:
                bbox = det.to_tlbr()
                class_name = det.get_class()
                # cv2.rectangle(img,(int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(255,0,0), 2)

                cntr_x = (bbox[0] + bbox[2]) / 2
                cntr_y = (bbox[1] + bbox[3]) / 2
                obj_pts.append([cntr_x, cntr_y])
                obj_classes.append(str(class_name))
        
        if len(obj_pts) == 0:
            temp_canvas = np.zeros((h_image, w_image + w_image // 3 + 3, 3), dtype='uint8')
            temp_canvas[:, w_image // 3 + 3:, :] = image

            cv2.imshow('Worker Monitoring', temp_canvas)

            if cv2.waitKey(1) == ord('q'):
                break

            continue

        obj_pts = np.array(obj_pts, dtype ='float32').reshape(-1, 1, 2)


        # transforming the object coordinates and the reference length points
        transformed_obj_pts = cv2.perspectiveTransform(obj_pts, new_M).astype('int').reshape(-1, 2)

        # filtering the points that are not in the ROI
        valid_pts = []
        valid_classes = []
        for i, pt in enumerate(transformed_obj_pts):
            if np.all(pt >= 0):
                valid_pts.append(pt)
                valid_classes.append(obj_classes[i])

        valid_pts = np.array(valid_pts)

        AVG_PERSON_HEIGHT = 1.7  # meters
        DANGER_THRESHOLD = 2.5  # meters
        px_per_meter = ref_len / AVG_PERSON_HEIGHT

        # determining pairs in danger
        dist_condensed = pdist(valid_pts)
        dist_matrix = squareform(dist_condensed)
        dist_matrix = dist_matrix / px_per_meter
        indexes_1, indexes_2 = np.where(dist_matrix < DANGER_THRESHOLD)

        indices_in_danger = []
        classes_in_danger = []

        pairs_history = [[] for _ in range(len(valid_pts))]
        for p1, p2 in zip(indexes_1, indexes_2):
            if p1 != p2:
                if p1 not in pairs_history[p2] and p2 not in pairs_history[p1]:
                    if obj_classes[p1] != obj_classes[p2]:
                        indices_in_danger.append([p1, p2])
                        classes_in_danger.append([obj_classes[p1], obj_classes[p2]])
                    pairs_history[p1].append(p2)
                    pairs_history[p2].append(p1)

        # visual space ---------------------------------------------------------------------------------------
        visual_corner_pts = np.array([[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]]).astype('float32').reshape(-1, 1, 2)
        visual_corner_pts = cv2.perspectiveTransform(visual_corner_pts, Ht).astype('int').reshape(-1, 2)

        h_visual = visual_corner_pts[3][1]
        w_visual = visual_corner_pts[1][0]
        H_CANVAS = h_image
        W_CANVAS = w_image // 3

        h_canvas_factor = H_CANVAS / h_visual
        w_canvas_factor = W_CANVAS / w_visual
        visual_factor = np.array(w_canvas_factor, h_canvas_factor)

        visual_canvas = np.zeros((H_CANVAS, W_CANVAS, 3))

        # displaying information
        for pt in valid_pts:
            pt[0] = (pt[0] * w_canvas_factor).astype('int')
            pt[1] = (pt[1] * h_canvas_factor).astype('int')
            cv2.circle(visual_canvas, tuple(pt), 3, (255, 255, 255), -1)

        obj_pts = obj_pts.reshape(-1, 2)
        for pt in obj_pts:
            cv2.circle(image, tuple(pt), 10, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.circle(image, tuple(pt), 2, (0, 0, 0), -1, cv2.LINE_AA)

        for index in indices_in_danger:
            pt1 = valid_pts[index[0]]
            pt2 = valid_pts[index[1]]
            cv2.line(visual_canvas, tuple(pt1), tuple(pt2), (0, 0, 255), 2, cv2.LINE_AA)

            pt1 = obj_pts[index[0]]
            pt2 = obj_pts[index[1]]
            cv2.line(image, tuple(pt1), tuple(pt2), (0, 0, 255), 2, cv2.LINE_AA)
        
        if len(indices_in_danger) > 0:
            cv2.rectangle(image, (w_temp // 25, h_temp // 100), (w_temp // 3, h_temp // 10), (0, 0, 0), -1)
            cv2.putText(image, 'WARNING', (w_temp // 25, h_temp // 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.namedWindow("Worker Monitoring", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Worker Monitoring", 1000, 700)

        final_visualization = np.zeros((h_image, w_image + W_CANVAS + 3, 3), dtype='uint8')
        final_visualization[:, :W_CANVAS, :] = visual_canvas
        final_visualization[:, W_CANVAS:W_CANVAS + 3, :] = 255
        final_visualization[:, W_CANVAS + 3:, :] = image

        cv2.imshow("Worker Monitoring", final_visualization)

        # print fps on screen 
        # fps  = ( fps + (1./(time.time()-t1)) ) / 2
        # cv2.putText(img, "FPS: {:.2f}".format(fps), (0, 30),
        #                   cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)
        # cv2.imshow('output', img)

        if output_name:
            out.write(final_visualization)
            frame_index = frame_index + 1
            list_file.write(str(frame_index)+' ')
            if len(converted_boxes) != 0:
                for i in range(0,len(converted_boxes)):
                    list_file.write(str(converted_boxes[i][0]) + ' '+str(converted_boxes[i][1]) + ' '+str(converted_boxes[i][2]) + ' '+str(converted_boxes[i][3]) + ' ')
            list_file.write('\n')

        # press q to quit
        if cv2.waitKey(1) == ord('q'):
            cv2.destroyAllWindows()
            break

    vid.release()
    if output_name:
        out.release()
        list_file.close()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
