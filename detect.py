import os
import time
import cv2 as cv

color_dict = {0: (255, 255, 255), 1: (255, 0, 0), 2: (0, 255, 0), 3: (0, 0, 255), 5: (255, 255, 0), 7: (0, 255, 255)}
label_dict = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}


def draw_bbox(im, bboxes, output, fcount):
    drawn_ids = []
    for box in bboxes:
        pt1 = (int(box[0]), int(box[1]))
        pt2 = (int(box[2]), int(box[3]))
        text_pos = (int(box[0]), int(box[1]) - 2)
        label = box[5]
        id = box[4]
        color = color_dict[box[5]]
        im = cv.rectangle(im, pt1, pt2, color, 2)
        if id not in drawn_ids:
            if output is not None:
                output.write(str(fcount)+', '+str(float(id))+', '+str(float(box[0]))+', '+str(float(box[1]))+', '+str(float(
                             box[2]-box[0]))+', '+str(float(box[3]-box[1])) + ', -1, -1, -1, -1\n')
            drawn_ids.append(id)
        cv.putText(im, str(id) + ': ' + label_dict[label], text_pos, cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return im


class DetectionSystem:

    def __init__(self, detector, input_path, output, is_video=False, show=True, show_framerate=True):
        self.detector = detector
        self.input_path = input_path
        self.is_video = is_video
        self.show = show
        self.show_framerate = show_framerate

        self.frame_rate = []
        self.frame_processing_time = []
        self.output = output

    def run(self):
        frame_count = 1
        if self.show:
            cv.namedWindow('Detector')
        if self.is_video:
            capture = cv.VideoCapture(self.input_path)
            prev_frame_time = 0
            while True:
                new_frame_time = time.time()
                fps = 1 / (new_frame_time - prev_frame_time)
                self.frame_rate.append(fps)
                prev_frame_time = new_frame_time
                _, frame = capture.read()
                if frame is None:
                    break
                detect_start_time = time.time()
                bboxes = self.detector.detect(frame)
                self.frame_processing_time.append(time.time() - detect_start_time)
                if len(bboxes) > 0:
                    frame = draw_bbox(frame, bboxes, self.output, frame_count)
                if self.show_framerate:
                    cv.putText(frame, str(int(fps)), (7, 70), cv.FONT_HERSHEY_SIMPLEX, 3, (100, 255, 0), 3, cv.LINE_AA)
                if self.show:
                    cv.imshow('Detector', frame)
                if cv.waitKey(1) & 0xFF == 27:
                    break
                frame_count += 1
            capture.release()
            cv.destroyAllWindows()
        else:
            prev_frame_time = 0
            for framedir in os.listdir(self.input_path):
                frame = cv.imread(self.input_path + framedir)
                new_frame_time = time.time()
                fps = 1 / (new_frame_time - prev_frame_time)
                self.frame_rate.append(fps)
                prev_frame_time = new_frame_time
                detect_start_time = time.time()
                bboxes = self.detector.detect(frame)
                self.frame_processing_time.append(time.time() - detect_start_time)
                if len(bboxes) > 0:
                    frame = draw_bbox(frame, bboxes, self.output, frame_count)
                if self.show_framerate:
                    cv.putText(frame, str(int(fps)), (7, 70), cv.FONT_HERSHEY_SIMPLEX, 3, (100, 255, 0), 3, cv.LINE_AA)
                if self.show:
                    cv.imshow('Detector', frame)
                if cv.waitKey(1) & 0xFF == 27:
                    break
                frame_count += 1
            cv.destroyAllWindows()
        if self.output is not None:
            self.output.close()
