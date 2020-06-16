import cv2
import numpy as np
from scipy.spatial.distance import pdist, squareform


mouse_pts = None
canvas_img = None
temp_img = None


class ProgramExit(Exception):
    pass


def ui_callback(event, x, y, flags, param):
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


def get_reference_pts_by_ui(img, callback):
    global  mouse_pts, canvas_img, temp_img

    cv2.namedWindow("Input Image")
    cv2.setMouseCallback("Input Image", callback)

    canvas_img = img.copy()
    temp_img = canvas_img.copy()
    h_temp, w_temp = temp_img.shape[:-1]
    mouse_pts = []

    while True:
        cv2.imshow("Input Image", temp_img)
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

        if len(mouse_pts) == 8:
            cv2.destroyWindow("Input Image")
            break
        if key == ord('q'):
            cv2.destroyWindow("Input Image")
            raise ProgramExit('Program was stopped by user with the "q" key')

    return mouse_pts


def euclidean(point1, point2):
    point1 = np.asarray(point1)
    point2 = np.asarray(point2)

    sq = (point1 - point2) ** 2
    euc = np.sum(sq, axis=0) ** 0.5
    return int(euc)


def get_perpendicular_vector(from_point, to_point, direction='cw', magnitude=100):

    from_point = np.array(from_point)
    to_point = np.array(to_point)

    v = np.zeros(2)
    v = to_point - from_point
    mag = np.sqrt(np.sum(np.square(v)))
    v = v * (1 / mag)

    if direction == 'ccw':
        v = np.array([v[1], -v[0]], dtype='float')
    else:
        v = np.array([-v[1], v[0]], dtype='float')

    r = np.zeros(2, dtype='float')
    r[0] = to_point[0] + v[0] * magnitude
    r[1] = to_point[1] + v[1] * magnitude

    return r


def get_homography_matrix(src_points, dst_points, roi_points):
    roi = roi_points
    # calculating original homography matrix
    dst = np.array(dst_points, dtype='float32')
    src = np.array(src_points, dtype='float32')
    M = cv2.getPerspectiveTransform(src, dst)

    corner_pts = np.array([roi[0], [roi[1][0], roi[0][1]], roi[1], [roi[0][0], roi[1][1]]],
                          dtype='float32').reshape(-1, 1, 2)
    transformed_corner_pts = cv2.perspectiveTransform(corner_pts, M)

    # calculating homography matrix with translation correction
    [xmin, ymin] = np.int32(transformed_corner_pts.min(axis=0).ravel() - 0.5)
    [xmax, ymax] = np.int32(transformed_corner_pts.max(axis=0).ravel() + 0.5)
    t = [-(xmin), -(ymin)]

    Ht = np.array([[1,0,t[0]],[0,1,t[1]],[0,0,1]])
    new_M = Ht.dot(M)
    
    borders = [[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]]
    
    return new_M, Ht, borders


def remove_objects_off_limits(obj_points, obj_classes):
    valid_pts = []
    valid_classes = []
    for i, pt in enumerate(obj_points):
        if np.all(pt >= 0):
            valid_pts.append(pt)
            valid_classes.append(obj_classes[i])

    valid_pts = np.array(valid_pts)

    return valid_pts, valid_classes


def detect_in_danger(obj_points, obj_classes, px_per_meter, threshold):
    dist_condensed = pdist(obj_points)
    dist_matrix = squareform(dist_condensed)
    dist_matrix = dist_matrix / px_per_meter
    indexes_1, indexes_2 = np.where(dist_matrix < threshold)

    indices_in_danger = []
    classes_in_danger = []

    pairs_history = [[] for _ in range(len(obj_points))]
    for p1, p2 in zip(indexes_1, indexes_2):
        if p1 != p2:
            if p1 not in pairs_history[p2] and p2 not in pairs_history[p1]:
                if obj_classes[p1] != obj_classes[p2]:
                    indices_in_danger.append([p1, p2])
                    classes_in_danger.append([obj_classes[p1], obj_classes[p2]])
                pairs_history[p1].append(p2)
                pairs_history[p2].append(p1)

    return indices_in_danger, classes_in_danger


def visualize(image, borders, valid_obj_pts, og_obj_pts, indices_in_danger, Ht):
    h_image, w_image = image.shape[:-1]
    visual_corner_pts = np.array(borders).astype('float32').reshape(-1, 1, 2)
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
    for pt in valid_obj_pts:
        pt[0] = (pt[0] * w_canvas_factor).astype('int')
        pt[1] = (pt[1] * h_canvas_factor).astype('int')
        cv2.circle(visual_canvas, tuple(pt), 3, (255, 255, 255), -1)

    og_obj_pts = og_obj_pts.reshape(-1, 2)
    for pt in og_obj_pts:
        cv2.circle(image, tuple(pt), 10, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.circle(image, tuple(pt), 2, (0, 0, 0), -1, cv2.LINE_AA)

    for index in indices_in_danger:
        pt1 = valid_obj_pts[index[0]]
        pt2 = valid_obj_pts[index[1]]
        cv2.line(visual_canvas, tuple(pt1), tuple(pt2), (0, 0, 255), 2, cv2.LINE_AA)

        pt1 = og_obj_pts[index[0]]
        pt2 = og_obj_pts[index[1]]
        cv2.line(image, tuple(pt1), tuple(pt2), (0, 0, 255), 2, cv2.LINE_AA)

    if len(indices_in_danger) > 0:
        cv2.rectangle(image, (w_image // 25, h_image // 100), (w_image // 3, h_image // 10), (0, 0, 0), -1)
        cv2.putText(image, 'WARNING', (w_image // 25, h_image // 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2, cv2.LINE_AA)

    final_visualization = np.zeros((h_image, w_image + W_CANVAS + 3, 3), dtype='uint8')
    final_visualization[:, :W_CANVAS, :] = visual_canvas
    final_visualization[:, W_CANVAS:W_CANVAS + 3, :] = 255
    final_visualization[:, W_CANVAS + 3:, :] = image
    
    return final_visualization
