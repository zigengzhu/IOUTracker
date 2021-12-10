import time
import cv2 as cv
import yolo_detector

new_boundaries = []


def append_click_pos(event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONDBLCLK:
        new_boundaries.append((x, y))


def draw_bbox(im, bboxes, blx, ury):
    color_dict = {3: (255, 0, 0), 2: (0, 0, 255), 7: (0, 255, 0), 5: (0, 255, 255)}
    label_dict = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
    if len(new_boundaries) > 0:
        for box in bboxes:
            pt1 = (box[0] + blx, box[1] + ury)
            pt2 = (box[2] + blx, box[3] + ury)
            text_pos = (box[0] + blx, box[1] + ury - 2)
            label = box[5]
            id = box[4]
            color = color_dict[box[5]]
            im = cv.rectangle(im, pt1, pt2, color, 2)
            cv.putText(im, 'ID-' + str(id) + ': ' + label_dict[label], text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                       color, 1)
    else:
        for box in bboxes:
            im = cv.rectangle(im, (box[0], box[1]), (box[2], box[3]), color_dict[box[5]], 2)
    return im


class DetectionSystem:

    def __init__(self, input_path, model='yolov5m', size=640, half=True, mode='deep_sort', frame_count=1000, show=True,
                 show_framerate=True, crop=True, scale=1.0):
        self.detector = yolo_detector.YoloDetector(model, size=size, half=half, mode=mode)
        self.frame_count = frame_count
        self.capture = cv.VideoCapture(input_path)
        self.show = show
        self.show_framerate = show_framerate
        self.crop = crop
        self.scale = scale
        self.frame_rate = []
        self.frame_processing_time = []

    def run(self):
        _, init_frame = self.capture.read()
        W = int(init_frame.shape[1] * self.scale)
        H = int(init_frame.shape[0] * self.scale)
        frame_count = 0

        if self.show:
            cv.namedWindow('window')

        if self.crop:
            new_boundaries.clear()
            cv.setMouseCallback('window', append_click_pos)
            while True:
                if len(new_boundaries) == 1:
                    break
                cv.imshow('window', cv.resize(init_frame, (W, H)))
                k = cv.waitKey(20) & 0xFF
                if k == 27:
                    break

            while True:
                if len(new_boundaries) == 2:
                    break
                cv.imshow('window', cv.resize(init_frame, (W, H)))
                k = cv.waitKey(20) & 0xFF
                if k == 27:
                    break

            if len(new_boundaries) != 2:
                self.capture.release()
                cv.destroyAllWindows()

            if new_boundaries[0][0] < new_boundaries[1][0] and new_boundaries[0][1] > new_boundaries[1][1]:
                new_boundaries.append(new_boundaries[0])
                new_boundaries.pop(0)

            print("Upper Right Coordinate: ", new_boundaries[0])
            print("Bottom Left Coordinate: ", new_boundaries[1])
        else:
            new_boundaries.append((0, 0))
            new_boundaries.append((0, 0))

        prev_frame_time = 0

        while True:
            if frame_count == self.frame_count:
                break

            new_frame_time = time.time()
            fps = 1 / (new_frame_time - prev_frame_time)
            self.frame_rate.append(fps)
            prev_frame_time = new_frame_time

            _, frame = self.capture.read()
            frame = cv.resize(frame, (W, H))

            if self.crop:
                urx, ury = new_boundaries[0]
                blx, bly = new_boundaries[1]

                cropped = frame[ury:bly, blx:urx, :]

                detect_start_time = time.time()
                bboxes = self.detector.detect(cropped)
                self.frame_processing_time.append(time.time() - detect_start_time)

                frame = cv.rectangle(frame, new_boundaries[0], new_boundaries[1], (255, 255, 255), 1)
                if len(bboxes) > 0:
                    frame = draw_bbox(frame, bboxes, blx, ury)
            else:
                detect_start_time = time.time()
                bboxes = self.detector.detect(frame)
                self.frame_processing_time.append(time.time() - detect_start_time)
                if len(bboxes) > 0:
                    frame = draw_bbox(frame, bboxes, 0, 0)

            if self.show_framerate:
                cv.putText(frame, str(int(fps)), (7, 70), cv2.FONT_HERSHEY_SIMPLEX, 3, (100, 255, 0), 3, cv2.LINE_AA)

            if self.show:
                cv.imshow('window', frame)

            if cv.waitKey(1) & 0xFF == 27:
                break

            frame_count += 1

        self.capture.release()
        cv.destroyAllWindows()
