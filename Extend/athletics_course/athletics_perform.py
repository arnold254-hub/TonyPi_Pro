#!/usr/bin/python3
# coding=utf8
# 4.拓展课程学习\11.拓展课程之田径运动课程\第4课 田径运动(4.Advanced Lessons\11.Athletics Sport Lesson\Lesson4 Athletics Performance)
import os
import sys
import cv2
import time
import math
import threading
import numpy as np
import hiwonder.ros_robot_controller_sdk as rrc
from hiwonder.Controller import Controller
import hiwonder.Misc as Misc
import hiwonder.PID as PID
import hiwonder.ActionGroupControl as AGC
import hiwonder.yaml_handle as yaml_handle

if sys.version_info.major == 2:
    print('Please run this program with python3!')
    sys.exit(0)

# 田径表演(athletics performance)

go_forward = 'go_forward'
go_forward_one_step = 'go_forward_one_step'
turn_right = 'turn_right_small_step_a'
turn_left  = 'turn_left_small_step_a'        
left_move = 'left_move_20'
right_move = 'right_move_20'
go_turn_right = 'turn_right'
go_turn_left = 'turn_left'

lab_data = None
servo_data = None

def load_config():
    global lab_data, servo_data
    
    lab_data = yaml_handle.get_yaml_data(yaml_handle.lab_file_path)
    servo_data = yaml_handle.get_yaml_data(yaml_handle.servo_file_path)

load_config()

board = rrc.Board()
ctl = Controller(board)

# 初始位置(initial position)
def initMove():
    ctl.set_pwm_servo_pulse(1,servo_data['servo1'],500)
    ctl.set_pwm_servo_pulse(2,servo_data['servo2'],500)   

object_left_x, object_right_x, object_center_y, object_angle = -2, -2, -2, 0
switch = False
# 变量重置(variable reset)
def reset():
    global object_left_x, object_right_x
    global object_center_y, object_angle, switch
    
    switch = False
    object_left_x, object_right_x, object_center_y, object_angle = -2, -2, -2, 0
    
def init():
    load_config()
    initMove()
    reset()


# 找出面积最大的轮廓(find out the contour with the maximal area)
# 参数为要比较的轮廓的列表(the parameter is a list of contours to compare)
def getAreaMaxContour(contours, area_min=10):
    contour_area_temp = 0
    contour_area_max = 0
    area_max_contour = None

    for c in contours:  # 历遍所有轮廓(iterate through all contours)
        contour_area_temp = math.fabs(cv2.contourArea(c))  # 计算轮廓面积(calculate contour area)
        if contour_area_temp > contour_area_max:
            contour_area_max = contour_area_temp
            if contour_area_temp >= area_min:  # 只有在面积大于设定值时，最大面积的轮廓才是有效的，以过滤干扰(only when the area is greater than the set value, the contour with the largest area is considered valid, to filter out interference)
                area_max_contour = c

    return area_max_contour, contour_area_max  # 返回最大的轮廓(return the contour with the maximal area)


line_centerx = 8888     # 8888 = no line seen yet
size = (640, 480)
roi = [ 
        (300, 340,  0, 640, 0.1), 
        (360, 400,  0, 640, 0.3), 
        (420, 480,  0, 640, 0.6)
      ]

roi_h1 = roi[0][0]
roi_h2 = roi[1][0] - roi[0][0]
roi_h3 = roi[2][0] - roi[1][0]
roi_h_list = [roi_h1, roi_h2, roi_h3]

#巡线视觉处理函数(line following vision processing function)
# Numbers that decide what counts as the black line. All pixel values refer to a 640 x 480 picture.
LINE = {
    'contrast': 60,      # a piece of line must be at least this much darker (L of LAB, 0-255) than the mat in the same band
    'max_L': 120,        # ... and darker than this whatever the mat; shadows on the mat are usually lighter
    'side_margin': 60,   # columns ignored at the left and right edge (the robot's own feet, the edge of the mat)
    'min_area': 120,     # smallest dark blob taken for a piece of line
    'max_width': 400,    # widest dark blob taken for a piece of line (a line at a shallow angle looks wide in a band)
    'line_width': 90,    # the line itself is at most this wide in the picture; a blob much wider than that is only a line if it runs diagonally
    'jump': 200,         # a piece of line further than this from where the line was is ignored
    'lost_frames': 10,   # after this many pictures without the line, the line may be anywhere again
}
last_line_x = None      # where the line was in the bottom band last time it was seen
last_seen_x = None      # the last line position found at all (for turning back to it when it is lost)
lost_count = 0
line_info = ''          # what the finder saw, for --view

def line_patrol(img, img_draw, target_color = 'black'):
    global last_line_x, last_seen_x, lost_count, line_info

    img_h, img_w = img.shape[:2]
    frame_resize = cv2.resize(img_draw, size, interpolation=cv2.INTER_NEAREST)
    frame_gb = cv2.GaussianBlur(frame_resize, (3, 3), 3)
    # the colour range of lab_config.yaml, except that the brightness limit is LINE['max_L']
    lo = (0, lab_data[target_color]['min'][1], lab_data[target_color]['min'][2])
    hi = (LINE['max_L'], lab_data[target_color]['max'][1], lab_data[target_color]['max'][2])
    margin = LINE['side_margin']

    centroid_x_sum = 0
    weight_sum = 0
    center_y = 0
    infos = []
    # start with the band nearest the feet: the line is expected where it was last time (or in the middle),
    # and in each band above it is expected near where it was found in the band below
    expect = last_line_x if last_line_x is not None else x_center
    for n in range(len(roi) - 1, -1, -1):
        r = roi[n]
        roi_h = roi_h_list[n]
        blobs = frame_gb[r[0]:r[1], r[2]:r[3]]
        frame_lab = cv2.cvtColor(blobs, cv2.COLOR_BGR2LAB)  # 将图像转换到LAB空间(convert the image to LAB space)
        L = frame_lab[:, :, 0]
        mat = float(np.percentile(L[:, margin:size[0] - margin], 90))    # how bright the mat is in this band
        frame_mask = cv2.inRange(frame_lab, lo, hi)                     # the black range of lab_config.yaml ...
        frame_mask[L >= mat - LINE['contrast']] = 0                      # ... and clearly darker than the mat
        frame_mask[:, :margin] = 0
        frame_mask[:, size[0] - margin:] = 0
        opened = cv2.morphologyEx(frame_mask, cv2.MORPH_OPEN, np.ones((6, 6), np.uint8))  # 开运算(opening operation)
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, np.ones((6, 6), np.uint8))  # 闭运算(closing operation)
        cnts = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)[-2]  # 找出所有轮廓(find out all contours)

        best, best_gap = None, None
        for c in cnts:
            area = math.fabs(cv2.contourArea(c))
            x, y, w, h = cv2.boundingRect(c)
            if area < LINE['min_area'] or w > LINE['max_width']:
                continue
            # a line runs across the band: it is narrow, or it is wider because it runs diagonally and then
            # fills only part of its box. A wide blob that fills its box is a dark patch, not a line.
            if w > 1.3 * LINE['line_width'] and area > 0.75 * w * h:
                continue
            gap = abs(x + w / 2 - expect)
            if gap > LINE['jump']:
                continue
            # the line has mat on both sides of it; the table or floor at the edge of the mat has not
            if x - 14 < 0 or x + w + 14 > size[0]:
                continue
            beside = min(np.median(L[y:y + h, x - 14:x - 4]), np.median(L[y:y + h, x + w + 4:x + w + 14]))
            if beside < mat - LINE['contrast'] / 2:
                continue
            if best is None or gap < best_gap:
                best, best_gap = c, gap

        line_L = None
        if best is not None:
            chosen = np.zeros_like(closed)
            cv2.drawContours(chosen, [best], -1, 255, -1)
            line_L = int(np.median(L[chosen > 0]))
        infos.append('%d/%s' % (mat, line_L if line_L is not None else '-'))
        if best is None:
            continue
        rect = cv2.minAreaRect(best)  # 最小外接矩形(the minimum bounding rectangle)
        box = np.int32(cv2.boxPoints(rect))  # 最小外接矩形的四个顶点(the four vertices of the minimum bounding rectangle)
        for i in range(4):
            box[i, 1] = box[i, 1] + r[0]
            box[i, 1] = int(Misc.map(box[i, 1], 0, size[1], 0, img_h))
            box[i, 0] = int(Misc.map(box[i, 0], 0, size[0], 0, img_w))
        cv2.drawContours(img_draw, [box], -1, (0, 0, 255, 255), 2)  # 画出四个点组成的矩形(draw the rectangle)

        # 获取矩形的对角点(retrieve the diagonal points of the rectangle)
        pt1_x, pt1_y = box[0, 0], box[0, 1]
        pt3_x, pt3_y = box[2, 0], box[2, 1]
        center_x, center_y = (pt1_x + pt3_x) / 2, (pt1_y + pt3_y) / 2  # 中心点(center point)
        cv2.circle(img_draw, (int(center_x), int(center_y)), 5, (0, 0, 255), -1)  # 画出中心点(draw center point)
        expect = int(Misc.map(center_x, 0, img_w, 0, size[0]))
        if n == len(roi) - 1:
            last_line_x = expect
        # 按权重不同对上中下三个中心点进行求和(sum the weighted centroids of the three regions)
        centroid_x_sum += center_x * r[4]
        weight_sum += r[4]

    line_info = 'mat/line brightness, bottom band first: ' + '   '.join(infos)
    if weight_sum != 0:
        lost_count = 0
        # 求最终得到的中心点(calculate the final obtained center point)
        line_centerx = int(centroid_x_sum / weight_sum)
        last_seen_x = line_centerx
        cv2.circle(img_draw, (line_centerx, int(center_y)), 10, (0, 255, 255), -1)  # 画出中心(draw center)
    else:
        lost_count += 1
        if lost_count >= LINE['lost_frames']:
            last_line_x = None
        line_centerx = 8888

    return line_centerx


# 色块定位视觉处理函数(the visual processing function for color block localization)
OBSTACLE = {
    'min_area': 400,     # smallest blob (pixels at 640 x 480) taken for a hurdle or a red stripe
    'min_width': 120,    # a hurdle or stripe lies across the path, so it is at least this wide in the picture
}
obstacles_on = True      # --no-obstacles turns the hurdle and stair detection off

def color_identify(img, img_draw, target_color = 'blue'):
    
    img_w = img.shape[:2][1]
    img_h = img.shape[:2][0]
    img_resize = cv2.resize(img, (size[0], size[1]), interpolation = cv2.INTER_CUBIC)
    GaussianBlur_img = cv2.GaussianBlur(img_resize, (3, 3), 0)#高斯模糊(Gaussian blur)
    frame_lab = cv2.cvtColor(GaussianBlur_img, cv2.COLOR_BGR2LAB) #将图像转换到LAB空间(convert the image to LAB space)
    frame_mask = cv2.inRange(frame_lab,
                                 (lab_data[target_color]['min'][0],
                                  lab_data[target_color]['min'][1],
                                  lab_data[target_color]['min'][2]),
                                 (lab_data[target_color]['max'][0],
                                  lab_data[target_color]['max'][1],
                                  lab_data[target_color]['max'][2]))  #对原图像和掩模进行位运算(perform bitwise operation to original image and mask)
    opened = cv2.morphologyEx(frame_mask, cv2.MORPH_OPEN, np.ones((3,3),np.uint8))#开运算(opening operation)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, np.ones((3,3),np.uint8))#闭运算(closing operation)
    contours = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[-2] #找出所有外轮廓(find out all the bounding contours)
    areaMax_contour = getAreaMaxContour(contours, area_min=OBSTACLE['min_area'])[0] #找到最大的轮廓(find out the contour with the maximal area)

    left_x, right_x, center_y, angle = -1, -1, -1, 0
    if areaMax_contour is not None and cv2.boundingRect(areaMax_contour)[2] < OBSTACLE['min_width']:
        areaMax_contour = None      # a speck of that colour, not a bar across the path
    if areaMax_contour is not None:
        down_x = (areaMax_contour[areaMax_contour[:,:,1].argmax()][0])[0]
        down_y = (areaMax_contour[areaMax_contour[:,:,1].argmax()][0])[1]

        left_x = (areaMax_contour[areaMax_contour[:,:,0].argmin()][0])[0]
        left_y = (areaMax_contour[areaMax_contour[:,:,0].argmin()][0])[1]

        right_x = (areaMax_contour[areaMax_contour[:,:,0].argmax()][0])[0]
        right_y = (areaMax_contour[areaMax_contour[:,:,0].argmax()][0])[1]
        
        if pow(down_x - left_x, 2) + pow(down_y - left_y, 2) > pow(down_x - right_x, 2) + pow(down_y - right_y, 2):
            left_x = int(Misc.map(left_x, 0, size[0], 0, img_w))
            left_y = int(Misc.map(left_y, 0, size[1], 0, img_h))  
            right_x = int(Misc.map(down_x, 0, size[0], 0, img_w))
            right_y = int(Misc.map(down_y, 0, size[1], 0, img_h))
        else:
            left_x = int(Misc.map(down_x, 0, size[0], 0, img_w))
            left_y = int(Misc.map(down_y, 0, size[1], 0, img_h))
            right_x = int(Misc.map(right_x, 0, size[0], 0, img_w))
            right_y = int(Misc.map(right_y, 0, size[1], 0, img_h))

        center_y = int(Misc.map((areaMax_contour[areaMax_contour[:,:,1].argmax()][0])[1], 0, size[1], 0, img_h))
        angle = int(math.degrees(math.atan2(right_y - left_y, right_x - left_x)))
        
        cv2.line(img_draw, (left_x, left_y), (right_x, right_y), (255, 0, 0), 2)     
            
    return left_x, right_x, center_y, angle      


skip = 1
skip_st = True
items = None
line_st = True
fence_st = True
strp_up = True
x_center = 360
STEER = {
    'straight': 35,      # line this close (pixels) to x_center: walk straight on
    'small_turn': 120,   # further off: turn with small steps; further than this: full turn
    'search_turns': 6,   # line lost: at most this many small-step turns back towards where it was last seen, then wait
}
search_count = 0
#机器人跟踪线程(robot tracking thread)
def move():
    global object_center_y,line_centerx,items
    global line_st,strp_up,fence_st,skip_st
    global search_count
    
    while True:
        if switch:
            if object_center_y >= 300:  #检测到台阶或者跨栏,进行位置微调(detect steps or hurdles and perform position refinement)
                
                if 20 <= object_angle < 90:
                    AGC.runActionGroup(go_turn_right)
                    time.sleep(0.2)           
                elif -20 >= object_angle > -90:
                    AGC.runActionGroup(go_turn_left)
                    time.sleep(0.2)
                
                elif line_centerx - x_center > 15:
                    AGC.runAction(right_move)
                elif line_centerx - x_center < -15:
                    AGC.runAction(left_move)
                
                elif 3 < object_angle < 20:
                    AGC.runActionGroup(turn_right)
                    time.sleep(0.2)           
                elif -5 > object_angle > -20:
                    AGC.runActionGroup(turn_left)
                    time.sleep(0.2)
                    
                elif 300 <= object_center_y < 430:    #在中心(in the center)
                    AGC.runActionGroup(go_forward_one_step)
                    time.sleep(0.2)
                    
                elif object_center_y >= 430: #位置靠近，可以跨栏或者上下台阶(the positions are close, allowing for stepping over or stepping onto/down from the platform)
                    time.sleep(0.5)
                    if object_center_y >= 430:
                        board.set_buzzer(1900, 0.1, 0.9, 1)
                        AGC.runActionGroup(go_forward_one_step) #前进一步(take one step forward)
                        time.sleep(0.5)
                        AGC.runActionGroup(go_forward_one_step) #前进一步(take one step forward)
                        time.sleep(0.5)
                        
                        if items == 'hurdles':# 跨栏(hurdles)
                            
                            AGC.runActionGroup('hurdles')
                            skip_st = True
                            strp_up = True
                            items = None
                        elif items == 'stairway':
                            if strp_up: # 上台阶(go up stairs)
                                AGC.runActionGroup('climb_stairs')
                                strp_up = False
                            else: # 下台阶(go down stairs)
                                for i in range(2):
                                    AGC.runActionGroup(go_forward_one_step)
                                time.sleep(0.2)
                                AGC.runActionGroup('down_floor')
                                strp_up = True
                            items = None
                            skip_st = True
                        time.sleep(0.5)
                        object_center_y = -1
                    
                else:
                    time.sleep(0.01)
                    
            elif line_st and line_centerx != 8888: #巡线(line following)
                search_count = 0
                error = line_centerx - x_center
                if abs(error) <= STEER['straight']:
                    AGC.runAction(go_forward)
                    time.sleep(0.2)
                elif abs(error) <= STEER['small_turn']:
                    AGC.runActionGroup(turn_right if error > 0 else turn_left)
                    time.sleep(0.2)
                else:
                    AGC.runAction(go_turn_right if error > 0 else go_turn_left)
                    time.sleep(0.2)
            elif line_st and last_seen_x is not None and search_count < STEER['search_turns']:
                # the line is out of sight: turn back towards the side it was last seen on
                search_count += 1
                AGC.runActionGroup(turn_right if last_seen_x > x_center else turn_left)
                time.sleep(0.3)
            else:
                time.sleep(0.01)
        else:
            time.sleep(0.01)
                
            
#作为子线程开启(start as a sub-thread)
th = threading.Thread(target=move)
th.daemon = True
th.start()


VIEW_TINT = {'black': (0, 255, 255), 'red': (0, 0, 255), 'blue': (255, 0, 0)}   # yellow, red, blue

# --view: tint the pixels that fall inside the black, red and blue ranges of lab_config.yaml
# and draw the picture rows and column the program steers by
def colour_view(img, img_draw):
    img_h, img_w = img.shape[:2]
    frame_lab = cv2.cvtColor(cv2.GaussianBlur(img, (3, 3), 0), cv2.COLOR_BGR2LAB)
    tint = img_draw.copy()
    for name, colour in VIEW_TINT.items():
        mask = cv2.inRange(frame_lab, tuple(lab_data[name]['min']), tuple(lab_data[name]['max']))
        tint[mask > 0] = colour
    cv2.addWeighted(img_draw, 0.5, tint, 0.5, 0, dst=img_draw)

    for r in roi:   # the three bands the line is looked for in
        y1 = int(Misc.map(r[0], 0, size[1], 0, img_h))
        y2 = int(Misc.map(r[1], 0, size[1], 0, img_h))
        cv2.rectangle(img_draw, (0, y1), (img_w - 1, y2), (0, 255, 0), 1)
    cx = int(Misc.map(x_center, 0, size[0], 0, img_w))   # the column the line is kept at
    cv2.line(img_draw, (cx, 0), (cx, img_h), (0, 255, 0), 1)
    for row, label in ((260, 'obstacle seen'), (300, 'line up to it'), (430, 'step over / on')):
        y = int(Misc.map(row, 0, size[1], 0, img_h))
        cv2.line(img_draw, (0, y), (img_w - 1, y), (255, 255, 255), 1)
        cv2.putText(img_draw, label, (4, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    text = 'yellow = dark   red/blue = stripe/hurdle   red boxes = line taken   (m hides, Esc/q quits)'
    cv2.putText(img_draw, line_info, (4, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3)
    cv2.putText(img_draw, line_info, (4, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    for x in (LINE['side_margin'], size[0] - LINE['side_margin']):   # the ignored side columns
        x = int(Misc.map(x, 0, size[0], 0, img_w))
        cv2.line(img_draw, (x, 0), (x, img_h), (0, 255, 0), 1)
    cv2.putText(img_draw, text, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3)
    cv2.putText(img_draw, text, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)


def run(img):
    global skip_st, line_st, strp_st, fence_st
    global object_left_x, object_right_x, skip, items
    global object_center_y, object_angle, line_centerx
    
    img_copy = img.copy()
    img_h, img_w = img.shape[:2]
    
    # 巡线(line following)
    line_centerx = line_patrol(img, img_copy, target_color = 'black') 
    
    # 跨栏(hurdles)
    if not obstacles_on:
        pass
    elif skip == 1:
        object_left_x, object_right_x, object_center_y, object_angle = color_identify(img, img_copy, target_color = 'blue')
        print('hurdles',object_left_x, object_right_x, object_center_y, object_angle)# 打印位置角度参数(print position angle parameter)
        if object_center_y >= 260:#准备跨栏，关闭错帧检测(prepare for hurdles, disable error frame detection)
            skip_st = False
            items = 'hurdles'
        elif object_center_y == -1:
            skip_st = True

    # 台阶(stair)
    elif skip == 2:
        object_left_x, object_right_x, object_center_y, object_angle = color_identify(img, img_copy, target_color = 'red')
        print('stairway',object_left_x, object_right_x, object_center_y, object_angle)# 打印位置角度参数(print position angle parameter)
        if object_center_y >= 260:#准备上台阶，关闭错帧检测(prepare to ascend stairs, disable error frame detection)
            skip_st = False
            items = 'stairway'
        elif object_center_y == -1:
            skip_st = True
               
    if skip_st: # 设置跨栏和台阶错帧检测(enable error frame detection for hurdles and stairs)
        skip += 1 
        if skip > 2:
            skip = 1
        
    return img_copy

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description='TonyPi athletics: follow the black line, step over the blue hurdle, '
                    'step onto and off the red-edged platform.')
    parser.add_argument('--view', action='store_true',
                        help='only look: show what the camera takes for black, red and blue, without walking')
    parser.add_argument('--no-obstacles', action='store_true',
                        help='only follow the line; ignore the blue hurdle and the red stripes')
    args = parser.parse_args()
    obstacles_on = not args.no_obstacles

    from hiwonder.CalibrationConfig import *
    init()   
    
    #加载参数(load parameters)
    param_data = np.load(calibration_param_path + '.npz')
    mtx = param_data['mtx_array']
    dist = param_data['dist_array']
    newcameramtx, roi_area = cv2.getOptimalNewCameraMatrix(mtx, dist, (640, 480), 0, (640, 480))
    mapx, mapy = cv2.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (640, 480), 5)

    
    # the robot's own camera wrapper: brings every picture to 640 x 480 and flips it as /boot/camera_setting.yaml says.
    # A raw cv2.VideoCapture may deliver another size, and the lens correction below then warps it to a black picture.
    import hiwonder.Camera as Camera
    camera = Camera.Camera()
    camera.camera_open()
    raw_size_shown = False
    AGC.runAction('stand_slow')
    switch = not args.view      # the move thread only walks when switch is True
    tint = True
    if args.view:
        print('View mode: the robot stands still. Esc or q quits, m hides or shows the colours.')
    no_picture_since = time.time()
    try:
        while True:
            ret,img = camera.read()
            if ret:
                no_picture_since = None
                if not raw_size_shown:
                    print('camera picture: %d x %d' % (img.shape[1], img.shape[0]))
                    raw_size_shown = True
                frame = img.copy()
                if frame.shape[1] != 640 or frame.shape[0] != 480:
                    frame = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_NEAREST)
                frame = cv2.remap(frame.copy(), mapx, mapy, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
                
                Frame = run(frame)
                if args.view and tint:
                    colour_view(frame, Frame)
                cv2.imshow('Frame', Frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord('q')):
                    break
                if key == ord('m'):
                    tint = not tint
            else:
                if no_picture_since is not None and time.time() - no_picture_since > 5:
                    print('No picture from the camera. Is the app service still running?  sudo systemctl stop tonypi')
                    no_picture_since = None
                time.sleep(0.01)
    finally:
        switch = False
        camera.camera_close()
        cv2.destroyAllWindows()
