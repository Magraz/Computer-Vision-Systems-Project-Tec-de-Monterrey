import cv2
import numpy as np
import time, random
from PIL import Image
import tensorflow as tf
from absl.flags import FLAGS
import matplotlib.pyplot as plt
from absl import app, flags, logging
from scipy.spatial.distance import pdist, squareform

from yolov3_tf2.dataset import transform_images
from yolov3_tf2.models import YoloV3, YoloV3Tiny
from yolov3_tf2.utils import draw_outputs, convert_boxes

from deep_sort import nn_matching
from deep_sort import preprocessing
from deep_sort.tracker import Tracker
from deep_sort.detection import Detection
from tools import generate_detections as gdet
import perspective_utils as pu


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


def main(_argv):
    # PARAMS
    max_cosine_distance = 0.5
    nn_budget = None
    nms_max_overlap = 1.0
    pure_yolo = False

    AVG_PERSON_HEIGHT = 1.7  # meters
    DANGER_THRESHOLD = 3.0  # meters
    
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
            mouse_pts = pu.get_reference_pts_by_ui(image, pu.ui_callback)

            cv2.namedWindow("Worker Monitoring", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Worker Monitoring", 1000, 700)

            # points of reference and ROI chosen by UI
            ref_pts = np.array(mouse_pts[:4])
            ref_len_pts = np.array(mouse_pts[4:6])
            roi = mouse_pts[6:]

            # length between reference points
            w_dst = max(pu.euclidean(ref_pts[0], ref_pts[2]), pu.euclidean(ref_pts[1], ref_pts[3]))
            ref_len = pu.euclidean(ref_len_pts[0], ref_len_pts[1])

            # calculating parallel vectors of lines
            c_1 = pu.get_perpendicular_vector(mouse_pts[0], mouse_pts[1],
                                              direction='ccw', magnitude=w_dst)
            c_2 = pu.get_perpendicular_vector(mouse_pts[1], mouse_pts[0],
                                              direction='cw', magnitude=w_dst)

            # getting the transformation matrix between the original reference
            # and the perpendicular "corrected" points
            dst = [ref_pts[0], ref_pts[1], c_2, c_1]
            new_M, Ht, borders = pu.get_homography_matrix(ref_pts, dst, roi)

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

                cntr_x = (bbox[0] + bbox[2]) / 2
                cntr_y = (bbox[1] + bbox[3]) / 2
                obj_pts.append([cntr_x, cntr_y])
                obj_classes.append(str(class_name))
        else:
            for det in detections:
                bbox = det.to_tlbr()
                class_name = det.get_class()

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
        valid_pts, valid_classes = pu.remove_objects_off_limits(transformed_obj_pts, obj_classes)

        px_per_meter = ref_len / AVG_PERSON_HEIGHT

        indices_in_danger, classes_in_danger = pu.detect_in_danger(valid_pts, valid_classes,
                                                                   px_per_meter, DANGER_THRESHOLD)

        final_visualization = pu.visualize(image, borders, valid_pts, obj_pts, indices_in_danger, Ht)

        cv2.namedWindow("Worker Monitoring", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Worker Monitoring", 1000, 700)
        cv2.imshow("Worker Monitoring", final_visualization)

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
