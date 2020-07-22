import cv2
import numpy as np
import apriltag

# results:
# [Detection(tag_family=b'tag36h11', 
# tag_id=10, hamming=0, 
# goodness=0.0, decision_margin=41.16551208496094, homography=array([[ 7.74365698e-01, -3.40121929e-01,  1.24594942e+00],
#        [ 1.62301978e-01,  2.76849434e-01,  3.08658771e+00],
#        [-1.77523417e-04, -6.36762632e-04,  9.06039461e-03]]), 
#  center=array([137.51602145, 340.66813242]), 
#  corners=array([[ 82.20069885, 268.10348511],
#        [247.9546051 , 312.20111084],
#        [203.75588989, 427.56399536],
#        [ 15.28420162, 372.17501831]]))]
# options = apriltag.Detectoroptions(families='tag36h11',
#                                  border=1,
#                                  nthreads=4,
#                                  quad_decimate=1.0,
#                                  quad_blur=0.0,
#                                  refine_edges=True,
#                                  refine_decode=False,
#                                  refine_pose=False,
#                                  debug=False,
#                                  quad_contours=True)
# detector = apriltag.Detector(options)
cap = cv2.VideoCapture(0)
detector = apriltag.Detector()
while True:
    ret, img = cap.read()
    gray_scale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    results = detector.detect(gray_scale)
    for cur_result in results:
        tag_id = cur_result.tag_id
        cx = cur_result.center.astype(np.int32)
        corners = cur_result.corners.astype(np.int32)
        for pt in corners:
            cv2.circle(img, tuple(pt), 3, (255, 0, 0))
        cv2.circle(img, tuple(cx), 3, (0, 0, 255))
        cv2.putText(img, str(tag_id), tuple(cx), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 0))

    cv2.imshow('img', img)
    cv2.waitKey(2)