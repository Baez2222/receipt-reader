import math
import os
import re

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

# output folder for rectified images
BASE_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), "rectified_images")


def rectification(FILE_PATH):
    im = cv2.imread(FILE_PATH)
    try:
        points = find_largest_contour(im)
        lowx, lowy, hix, hiy = order_rectpts(points)
        
        if lowy > im.shape[0] * 0.2:
            im = im[int(lowy*0.7):, :]
            points = find_largest_contour(im)
        return crop_page(im, points, FILE_PATH)
    
    except:
        print("Wrong rectangle points. Retrying...")
        points = find_largest_contour(im, 2)
        lowx, lowy, hix, hiy = order_rectpts(points)
        
        if lowy > im.shape[0] * 0.2:
            im = im[int(lowy*0.7):, :]
            points = find_largest_contour(im)
        
        return crop_page(im, points, FILE_PATH)


def order_rectpts(points):
    x1 = int(points[0][0])
    x2 = 0
    y1 = int(points[0][1])
    y2 = 0
    for i in range(1, len(points)):
        if int(points[i][0]) > x1 and x2 == 0:
            x2 = int(points[i][0])
        if int(points[i][1]) > y1 and y2 == 0:
            y2 = int(points[i][1])
    if x1 < x2:
        lowx = x1
        hix = x2
    else:
        lowx = x2
        hix = x1
    if y1 < y2:
        lowy = y1
        hiy = y2
    else:
        lowy = y2
        hiy = y1
    print(lowx,lowy)
    print(hix,hiy)
    print()
    
    return lowx, lowy, hix, hiy


def find_largest_contour(im, method = 1):
    imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(imgray, 127, 255, 0)

    # kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (27, 27))
    # kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
    # thresh1 = cv2.morphologyEx(thresh, cv2.MORPH_ERODE, kernel1)
    # thresh1 = cv2.morphologyEx(thresh1, cvFILE_PATH2.MORPH_DILATE, kernel2)


    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


    pageContour = [""]
    max = 0
    for i in contours:
        if cv2.contourArea(i) > max:
            pageContour[0] = i
            max = cv2.contourArea(i)
    # x,y,w,h = cv.boundingRect(pageContour[0])
    # cv.rectangle(im, (x,y), (x+w, y+h), (0,255,0), 3)
    rectangle = cv2.minAreaRect(pageContour[0])
    ((x,y),(w,h),angle) = cv2.minAreaRect(pageContour[0])


    # convert the minAreaRect output to points
    pts = cv2.boxPoints(rectangle)
    # contours must be of shape (n, 1, 2) and dtype integer
    pts = pts.reshape(-1, 1, 2)
    pts = pts.astype(int)
    
    rectangle_pts = []
    for coord in pts:
        corner_pts = []
        for i in str(coord).split():
            _ = re.sub("[^0-9.]", "", i)
            if _ != "":
                corner_pts.append(_)
        rectangle_pts.append(corner_pts)
    print(rectangle_pts)
    if method == 2:
        rectangle_pts.append(rectangle_pts[0])
        rectangle_pts.remove(rectangle_pts[0])
        print(rectangle_pts)
    # crop_img = im[lowy:hiy, lowx:hix]
    
    # imgcont = cv2.drawContours(im, pageContour[0], -1, (0, 255, 0),3)
    # cv2.imshow("cropped contours 1", cv2.resize(imgcont, (750, 900)))
    # cv2.waitKey(0)
    
    # imgcont = cv2.drawContours(crop_img, pageContour[0], -1, (0, 255, 0),3)
    # cv2.imshow("cropped contours 2", cv2.resize(imgcont, (750, 900)))
    # cv2.waitKey(0)
    # cv2.rectangle(im, (int(rectangle_pts[3][0]), int(rectangle_pts[3][1])), (int(rectangle_pts[1][0]), int(rectangle_pts[1][1])), (0,255,0), 3)
    # cv2.imshow("cropped contours", cv2.resize(im, (900, 1000)))
    # cv2.waitKey(0)
    
    return rectangle_pts


def crop_page(img, rectangle_pts, FILE_PATH):
    # img = cv2.imread(FILE_PATH)
    rows,cols,ch = img.shape

    # pts1 = np.float32([[69,369], [1943, 415], [1887,2713], [13,2667]])
    pts1 = np.float32(rectangle_pts)
    # print(pts1)
    
    ratio=0.70710678118 # aspect ratio of document
    docH=math.sqrt((pts1[2][0]-pts1[1][0])*(pts1[2][0]-pts1[1][0])+(pts1[2][1]-pts1[1][1])*(pts1[2][1]-pts1[1][1]))
    docW=ratio*docH;
    pts2 = np.float32([[pts1[0][0],pts1[0][1]], [pts1[0][0]+docW, pts1[0][1]], [pts1[0][0]+docW, pts1[0][1]+docH], [pts1[0][0], pts1[0][1]+docH]])

    M = cv2.getPerspectiveTransform(pts1,pts2)

    offsetSize=500
    transformed = np.zeros((int(docW+offsetSize), int(docH+offsetSize)), dtype=np.uint8);
    dst = cv2.warpPerspective(img, M, transformed.shape, flags=cv2.INTER_NEAREST)
    
    im = dst
    imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(imgray, 127, 255, 0)

    # kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (27, 27))
    # kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
    # thresh1 = cv2.morphologyEx(thresh, cv2.MORPH_ERODE, kernel1)
    # thresh1 = cv2.morphologyEx(thresh1, cv2.MORPH_DILATE, kernel2)


    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    pageContour = [""]
    max = 0
    print(len(contours))
    for i in contours:
        if cv2.contourArea(i) > max:
            pageContour[0] = i
            max = cv2.contourArea(i)
    # print(pageContour[0])
    # x,y,w,h = cv.boundingRect(pageContour[0])
    # cv.rectangle(im, (x,y), (x+w, y+h), (0,255,0), 3)
    # print(pageContour[0])
    # imgcont = cv2.drawContours(im, pageContour[0], -1, (0, 255, 0),3)
    # cv2.imshow("cropped contours", cv2.resize(imgcont, (700, 800)))
    # cv2.waitKey(0)
    
    # print(M)
    rectangle = cv2.minAreaRect(pageContour[0])
    ((x,y),(w,h),angle) = cv2.minAreaRect(pageContour[0])
        
    # conver the minAreaRect output to points
    pts = cv2.boxPoints(rectangle)
    # contours must be of shape (n, 1, 2) and dtype integer
    pts = pts.reshape(-1, 1, 2)
    pts = pts.astype(int)
    print(pts)
    print((pts[0][0][0]))
    print(str(pts[1][0][0]))
    print(str(pts[0][0][1]))
    print(str(pts[2][0][1]))
    
    x1 = pts[0][0][0]
    x2 = 0
    y1 = pts[0][0][1]
    y2 = 0
    for i in range(1, len(pts)):
        if abs(pts[i][0][0] - x1) > pts[i][0][0]*.2 and x2 == 0:
            x2 = pts[i][0][0]
        if abs(pts[i][0][1] - y1) > pts[i][0][1]*.2 and y2 == 0:
            y2 = pts[i][0][1]
    if x1 < x2:
        lowx = x1
        hix = x2
    else:
        lowx = x2
        hix = x1
    if y1 < y2:
        lowy = y1
        hiy = y2
    else:
        lowy = y2
        hiy = y1
    print(lowx,lowy)
    print(hix,hiy)
    print()
    crop_img = im[lowy:hiy, lowx:hix]
    cv2.imshow("cropped", cv2.resize(crop_img, (800, 1000)))
    cv2.waitKey(0)
    # cv2.imshow("cropped", cv2.resize(im, (800, 1000)))
    crop_img_path = os.path.join(BASE_PATH, os.path.basename(FILE_PATH))
    print(crop_img_path)
    cv2.imwrite(crop_img_path, crop_img)
    
    
    im = Image.open(crop_img_path)
    # im = im.filter(ImageFilter.MedianFilter())
    enhancer = ImageEnhance.Contrast(im)
    im = enhancer.enhance(2)
    # im = im.convert('1')
    im.save(crop_img_path)
    return crop_img_path
