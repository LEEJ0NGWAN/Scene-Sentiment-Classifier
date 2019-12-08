from random import randint as rand
import cv2 as cv
import math
import re
import os
import numpy as np

# resize 크기
width   = 480
height  = 270

# 영화 장르
vid_type        = ['action', 'horror', 'romance', 'ani2D', 'ani3D', 'sf']

# 각 구간의 시작, 끝 프레임과 길이를 저장할 리스트
range_list      = []

# 각 구간에서 뽑을 프레임의 개수 리스트
randSize_list   = []

# 각 구간에서 랜덤으로 뽑은 프레임을 나열한 리스트
randFrame_list  = []

# 입력 받는 시간대 hh:mm:ss를 hh, mm, ss로 끊는 놈
timeSplitter    = re.compile('[0-6][0-9]')

# 저장할 파일의 시작 넘버링
file_num        = 1

# 몇 프레임을 연속으로 저장할 건지
shot_length     = 2

# 뽑을 프레임 수
extraction_size = 600
random_limit    = math.ceil(extraction_size/shot_length)

# 동영상 불러오기
while True:
    print("장르 : ", vid_type)
    typeName    = input("장르 선택 (ex: horror) >>> ")

    if typeName not in vid_type:
        print("입력 오류")
        continue

    fileName    = input("동영상 이름 (ex: test.avi) >>>")

    try:
        vid  = cv.VideoCapture(fileName)
        vid.set(cv.CAP_PROP_FPS, 24)
        if not vid.isOpened(): raise NameError('동영상이 아니거나 파일이 없음')

    except cv.error     as e : print(e)
    except Exception    as e : print(e)

    else: break

# 입력 동영상 총 프레임 수
total_length    = int(vid.get(cv.CAP_PROP_FRAME_COUNT))
fps             = vid.get(cv.CAP_PROP_FPS)

# 프레임 뽑을 구간의 개수 정하기
while True:
    try:
        count = int(input("뽑을 구간의 개수 (ex: 2) >>>"))
    
    except ValueError   : print("공백ㄴㄴ 소수ㄴㄴ 양의 정수만 가능")
    if count < 1        : print("최소 1개여야함")

    else: break

while True:
    accumulated_length = 0
    range_list.clear()
    # 각 구간의 시작, 끝 시간 입력
    for index in range(count):
        while True:
            
            print("\n%d번째 구간"%index)

            inputTime   = input("시작 시간 입력 (ex: 00:12:53) >>>")
            start_time  = timeSplitter.findall(inputTime)

            if not len(start_time) == 3:
                print("제대로 입력하셈 hh:mm:ss 맞춰야함")
                continue
            
            start_seconds   = int(start_time[0]) * 3600 + int(start_time[1]) * 60 + int(start_time[2])
            start_frame     = int(fps * start_seconds)

            if total_length < start_frame:
                print("잘못된 시간인거 같은디")
                continue

            inputTime   = input("끝 시간 입력 (ex: 00:12:53) >>>")
            end_time    = timeSplitter.findall(inputTime)

            if not len(end_time) == 3:
                print("제대로 입력하셈 hh:mm:ss 맞춰야함")
                continue

            end_seconds = int(end_time[0]) * 3600 + int(end_time[1]) * 60 + int(end_time[2])
            end_frame   = int(fps * end_seconds)

            if total_length < end_frame:
                print("잘못된 시간인거 같은디")
                continue

            if start_frame >= end_frame:
                print("왜 시작 시간이 더 뒤에 있음;;;")
                continue

            part_length = end_frame - start_frame
            accumulated_length += part_length

            # 구간의 시작,끝 프레임과 길이를 저장
            tmp_list = []
            tmp_list.append(start_frame)
            tmp_list.append(end_frame)
            tmp_list.append(part_length)
            range_list.append(tmp_list)

            break

    if accumulated_length < extraction_size:    print("너무 짧음")
    else:   break

# 랜덤으로 프레임 뽑기
for iteration in range(random_limit):

    while True:
        random_index = rand(0,count-1)
        random_frame = rand(range_list[random_index][0], range_list[random_index][1])

        if random_frame % shot_length != 0  : continue
        if random_frame in randFrame_list   : continue

        else:
            randFrame_list.append(random_frame)
            break

# 랜덤 프레임 정렬 여부
while True:
    print("\n랜덤으로 뽑은 프레임 정렬할까요? y 또는 n 입력")
    select = input("선택 (ex: y)>>>")
    if select == "y":
        randFrame_list.sort()
        break
    if select == "n":
        break

# 영화 장르 이름으로 폴더 생성
try:
    if not(os.path.isdir(typeName)):
        os.makedirs(os.path.join(typeName))
except OSError as e:
    if e.errno != OSError.errno.EEXIST:
        raise
os.chdir(typeName)

# 영화 넘버링 확인
for index in range(101):
    if index == 0: continue
    if os.path.exists(typeName + '_movie%d_1.png'%index): continue
    else:
        movie_num = index
        break

# 각 구간에서 뽑은 랜덤 프레임으로 이동하면서 연속된 샷의 프레임들을 뽑아서 저장
prev_frame = np.zeros((800,1920,3), np.float32)
for frame_num in randFrame_list:
    for shot in range(shot_length):
        vid.set(cv.CAP_PROP_POS_FRAMES, frame_num + shot)
        flag, frame     = vid.read()

        while True:
            diff = (np.sum((frame - prev_frame)*(frame - prev_frame)) / (width*height*3))
            if(diff > 100):
                print(diff)
                break
            else:
                flag, frame     = vid.read()
        
        resized_frame   = cv.resize(frame, (width,height))

        cv.imshow(fileName, resized_frame)
        cv.waitKey(10)
        cv.imwrite(typeName + '_movie%d_%d.png'%(movie_num, file_num), resized_frame)

        np.copyto(prev_frame, frame)
        file_num += 1

vid.release()
cv.destroyAllWindows()
