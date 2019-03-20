
import cv2 as cv

#caffe_root = '/home/ubuntu/caffe/'
#image = cv2.imread('/home/ubuntu/caffe/examples/images/cat.jpg')
#labels_file = caffe_root + 'data/ilsvrc12/synset_words.txt'

model = 'mobilnet_iter_73000.caffemodel'
prototxt = 'patched_deploi.prototxt'


net = cv.dnn.readNet('.xml','.bin')

net.setPreferableTarget(cv.dnn.DNN_TARGET_MYRIAD)

frame = cv.imread('photo.jpg')

blob = cv.dnn.blobFromImage(frame, size=(672, 384), ddepth=cv.CV_8U)
net.setInput(blob)
out = net.forward()

# Draw detected faces on the frame
for detection in out.reshape(-1, 7):
    confidence = float(detection[2])
    xmin = int(detection[3] * frame.shape[1])
    ymin = int(detection[4] * frame.shape[0])
    xmax = int(detection[5] * frame.shape[1])
    ymax = int(detection[6] * frame.shape[0])

    if confidence > 0.5:
        cv.rectangle(frame, (xmin, ymin), (xmax, ymax), color=(0, 255, 0))

# Save the frame to an image file
cv.imwrite('out.png', frame)
