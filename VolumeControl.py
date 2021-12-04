import numpy as np
import cv2
import mediapipe as mp
import time
import HandTrackingModule as htm
import math
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

WidthCamera, HeightCamera = 640, 480

cap = cv2.VideoCapture(1)
cap.set(3, WidthCamera)
cap.set(4, HeightCamera)
previous_time = 0

detector = htm.HandDetector()

devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
# volume.GetMute()
volume.GetMasterVolumeLevel()
volume_range = volume.GetVolumeRange()
min_volume = volume_range[0]
max_volume = volume_range[1]
vol = 0
vol_bar = 400
vol_percentage = 0

while True:
    success, img = cap.read()
    img = detector.find_hands(img)
    land_mark_list = detector.find_position(img, draw=False)
    if len(land_mark_list) != 0:
        # print(land_mark_list[4], land_mark_list[8])
        x1, y1 = land_mark_list[4][1], land_mark_list[4][2]
        x2, y2 = land_mark_list[8][1], land_mark_list[8][2]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        cv2.circle(img, (x1, y1), 10, (255, 0, 255), cv2.FILLED)
        cv2.circle(img, (x2, y2), 10, (255, 0, 255), cv2.FILLED)
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
        cv2.circle(img, (cx, cy), 10, (255, 0, 255), cv2.FILLED)

        length = math.hypot(x2 - x1, y2 - y1)
        # print(length)

        # Hand range -> 30 - 240
        # Volume Range -> -96 - 0

        vol = np.interp(length, [30, 240], [min_volume, max_volume])
        vol_bar = np.interp(length, [30, 240], [400, 150])
        vol_percentage = np.interp(length, [30, 240], [0, 100])
        print(int(length), vol)
        volume.SetMasterVolumeLevel(vol, None)

        if length < 50:
            cv2.circle(img, (cx, cy), 12, (255, 255, 255), cv2.FILLED)

    cv2.rectangle(img, (50, 150), (85, 400), (255, 0, 255), 3)
    cv2.rectangle(img, (50, int(vol_bar)), (85, 400), (255, 0, 255), cv2.FILLED)

    current_time = time.time()
    fps = 1 / (current_time - previous_time)
    previous_time = current_time

    cv2.putText(img, f'FPS: {int(fps)}', (40, 100), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 255), 2)
    cv2.putText(img, f' {int(vol_percentage)}%', (40, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 255), 2)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
