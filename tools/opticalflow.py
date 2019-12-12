import cv2
import numpy as np

frame1 = cv2.imread('action_movie7_15.png')
frame2 = cv2.imread('action_movie7_16.png')

prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
next = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)

width = frame1.shape[1]
height = frame1.shape[0]

flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
channel = flow.reshape(height,width, 2)

max_val = channel.max()
col_motion = channel[:,:,0]
row_motion = channel[:,:,1]
# 여기서 정의된 col_motion과 row_motion 채널을 사용하기
# 모션을 쌓아서 돌릴 때 R, G, B, ROW_MOTION, COL_MOTION 이렇게 하면
# 아래는 볼 필요 없는 코드 (시각화)

cv2.imshow('col motion',col_motion / max_val)
cv2.imshow('row motion',row_motion / max_val)
mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
hsv = np.zeros_like(frame1)
hsv[...,0] = ang*180/np.pi/2
hsv[...,1] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
hsv[...,2] = 255
rgb = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
cv2.imshow('visualized',rgb)
cv2.waitKey()
cv2.destroyAllWindows()
