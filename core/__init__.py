from detection import Detector
import cv2


class compose():

    def  __init__(self, Detector=Detector()):
        self.Detector = Detector

    def composing(self):

        self.Detector.detector.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        cv2.startWindowThread()
        cap = self.Detector.streamer.stream
        # out = self.Detector.streamer.writer

        while True:
            # Capture frame-by-frame
            frame = self.Detector.streamer.get_frame()

            # drow bounding boxes
            self.Detector.bboxes(frame) #, (640, 480))

            # Write the output video
            # out.write(frame.astype('uint8'))
            # Display the resulting frame
            self.Detector.streamer.display(frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        # When everything done, release the capture
        cap.release()
        # and release the output
        # out.release()
        # finally, close the window
        cv2.destroyAllWindows()
        cv2.waitKey(1)


if __name__ == "__main__":
    comp = compose()
    comp.composing()
