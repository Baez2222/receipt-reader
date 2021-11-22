import cv2
import numpy as np
import pytesseract
import argparse

import math
import re
import numpy as np
from matplotlib import pyplot as plt
import nltk
import string
import copy
from PIL import Image


parser = argparse.ArgumentParser(description='Locating bounding box in the image')
parser.add_argument('file', type=str,
                    help='path to file')
parser.add_argument('location_list', type=str,
                    help='list of coordinate % points of each box relative to the top right corner of the page')

args = parser.parse_args()

english_vocab = set(w.lower() for w in nltk.corpus.words.words())


# python3 test.py "/home/hbaez/Projects/detectron2/ticket_photos/20210308_235908.jpg" "[(.5, .2), (.3, .8)]"

def find_largest_contour(file_path):
    im = cv2.imread(file_path)
    imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(imgray, 127, 255, 0)

    # kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (27, 27))
    # kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
    # thresh1 = cv2.morphologyEx(thresh, cv2.MORPH_ERODE, kernel1)
    # thresh1 = cv2.morphologyEx(thresh1, cv2.MORPH_DILATE, kernel2)


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
    
    # rectangle_pts.append(rectangle_pts[0])
    # rectangle_pts.remove(rectangle_pts[0])
    
    print(rectangle_pts)
    
    imgcont = cv2.drawContours(im, pageContour[0], -1, (0, 255, 0),3)
    cv2.imshow("cropped contours", cv2.resize(imgcont, (900, 1000)))
    cv2.waitKey(0)
    # cv2.rectangle(im, (int(rectangle_pts[3][0]), int(rectangle_pts[3][1])), (int(rectangle_pts[1][0]), int(rectangle_pts[1][1])), (0,255,0), 3)
    # cv2.imshow("cropped contours", cv2.resize(im, (900, 1000)))
    # cv2.waitKey(0)
    
    return rectangle_pts
    
    
    
    
    
    
    
    
    contours
def crop_page(file_path, coords_list, rectangle_pts):
    img = cv2.imread(file_path)
    rows,cols,ch = img.shape

    # pts1 = np.float32([[69,369], [1943, 415], [1887,2713], [13,2667]])
    pts1 = np.float32(rectangle_pts)
    # print(pts1)
    
    ratio=0.70710678118
    docH=math.sqrt((pts1[2][0]-pts1[1][0])*(pts1[2][0]-pts1[1][0])+(pts1[2][1]-pts1[1][1])*(pts1[2][1]-pts1[1][1]))
    docW=ratio*docH;
    pts2 = np.float32([[pts1[0][0],pts1[0][1]], [pts1[0][0]+docW, pts1[0][1]], [pts1[0][0]+docW, pts1[0][1]+docH], [pts1[0][0], pts1[0][1]+docH]])

    M = cv2.getPerspectiveTransform(pts1,pts2)

    offsetSize=500
    transformed = np.zeros((int(docW+offsetSize), int(docH+offsetSize)), dtype=np.uint8);
    dst = cv2.warpPerspective(img, M, transformed.shape)
    
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
    # crop_img = im[686:2009,1619:2916]
    # crop_img = im[352:4000, 0:6101]
    cv2.imshow("cropped", cv2.resize(crop_img, (800, 1000)))
    cv2.waitKey(0)
    # cv2.imshow("cropped", cv2.resize(im, (800, 1000)))
    
    cv2.imwrite("rectified/4.jpg", crop_img)
    
    return crop_img
    
    




def template_matching(im, template):
    # template = cv2.imread('ticket_photos/20210308_234355_crop.jpg')
    # template = cv2.imread('ticket_photos/20210622_124500_crop.jpg')

    # im = cv2.imread('ticket_photos/20210308_235908.jpg')
    # im = cv2.imread('ticket_photos/20210622_123215.jpg')

    # pre process template

    imgray_ = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    ret_, thresh_ = cv2.threshold(imgray_, 127, 255, 0)

    # pre process image

    imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(imgray, 127, 255, 0)

    # img2 = thresh.copy()
    w, h = thresh_.shape[::-1]
    # All the 6 methods for comparison in a list
    methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
                'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']
    # methods = ['cv2.TM_CCOEFF_NORMED']
    correct_top, correct_bottom = 0, 0
    for meth in methods:
        img = copy.deepcopy(thresh)
        method = eval(meth)
        
        # Apply template Matching
        res = cv2.matchTemplate(img,thresh_,method)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
        if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
            top_left = min_loc
        else:
            top_left = max_loc
        bottom_right = (top_left[0] + w, top_left[1] + h)
        
        text = ocrtext(bottom_right, top_left, im)
        
        if text != "":
            correct_top = top_left
            correct_bottom = bottom_right
            # break
        # for word in text:
        #     if nltk.edit_distance(word, "ship") < 1:
        
        
        # cv2.rectangle(im,top_left, bottom_right, 255, 2)
        # # plt.subplot(121),
        # plt.imshow(res,cmap = 'gray')
        # plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
        # # plt.subplot(122),
        # plt.imshow(im,cmap = 'gray')
        # plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
        # plt.suptitle(meth)
        # # plt.savefig("rectified/13_template_matches.jpg")
        # plt.show()

    return correct_bottom, correct_top, text





def ocrtext(bottom_right, top_left, cropped):
    # Apply OCR on the cropped image
    crop_img = cropped[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
    # cv2.imshow("crop_img", crop_img)
    # cv2.waitKey(0)
    
    imgray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((1, 1), np.uint8)
    img = cv2.dilate(imgray, kernel, iterations=1)
    img = cv2.erode(img, kernel, iterations=1)
    ret, thresh = cv2.threshold(img, 127, 255, 0)
    
    # kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (27, 27))
    # kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
    # thresh1 = cv2.morphologyEx(thresh, cv2.MORPH_ERODE, kernel1)
    # thresh1 = cv2.morphologyEx(thresh1, cv2.MORPH_DILATE, kernel2)
    text = []
    text.append(pytesseract.image_to_string(crop_img))
    crop_img = cv2.threshold(cv2.GaussianBlur(img, (5, 5), 0), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    text.append(pytesseract.image_to_string(crop_img))

    crop_img = cv2.threshold(cv2.bilateralFilter(img, 5, 75, 75), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    text.append(pytesseract.image_to_string(crop_img))
    
    crop_img = cv2.threshold(cv2.medianBlur(img, 3), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    text.append(pytesseract.image_to_string(crop_img))
    
    crop_img = cv2.adaptiveThreshold(cv2.GaussianBlur(img, (5, 5), 0), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2)
    text.append(pytesseract.image_to_string(crop_img))
    
    crop_img = cv2.adaptiveThreshold(cv2.bilateralFilter(img, 9, 75, 75), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2)
    text.append(pytesseract.image_to_string(crop_img))
    
    crop_img = cv2.adaptiveThreshold(cv2.medianBlur(img, 3), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2)
    text.append(pytesseract.image_to_string(crop_img))
    
    correct_top, correct_bottom = 0, 0
    correct_text = ""
    print("!!!!!!!!!!!!!!!!!!!!!!")
    print(text)
    print("!!!!!!!!!!!!!!!!!!!!!!")
    for i in text:
        # print(i)
        # lower case
        curr = i.lower()
        
        # remove punctuation
        curr = "".join([char for char in curr if char not in string.punctuation])

        # tokenization
        words = nltk.word_tokenize(curr)
        
        words = [word for word in words if len(word) > 1]
        print(words)
        # spell = SpellChecker()
        # misspelled = spell.unknown(words)

        
        for word in words:
            if nltk.edit_distance(word, "ship") < 2:
                correct_text = i
        # for word in misspelled:
            # possible.append(spell.correction(word))
            # print(spell.candidates(word))
    
    return correct_text

def spell_checker(textlist):
    textlist = [['ship', 'to', 'ooo10', 'impression', 'homes', '6a1', 'antswood', 'dr', 'community', 'live', 'oak', 'cree', 'fort', 'worth', 'txxoooo0oo'],
                ['ship', 'to', '©0010', 'impression', 'homes', '641', 'antswood', 'dr', 'community', 'live', 'oak', 'cree', 'fort', 'worth', 'txxoo000'],
                ['ship', 'to', 'ooo10', 'impression', 'homes', '641', 'antswood', 'dr', 'community', 'live', 'oak', 'cree', 'fort', 'worth', 'txoocoo0o'],
                ['bhip', 'to', 'oo0o10', 'impression', 'homes', '641', 'antswood', 'dr', 'community', 'live', 'oak', 'cree', 'fort', 'worth', 'txxxoo0oo000'],
                ['ship', 'to', 'oo00o190', 'impression', 'homes', '642', '°antswood', 'dr', 'community', 'live', 'oak', 'cree', 'fort', 'worth', 'txxxoo0000°'],
                ['bhip', 'tos', 'oo001i10°', 'impression', 'homes', '641', 'antswood', 'odor', 'community', 'live', 'oak', 'cree', 'fort', 'worth', 'txo0000'],
                ['shap', 'to', '90010', 'impression', 'homes', '6421', 'antswood', 'dr', 'community', 'live', 'oak', 'cree', 'fort', 'worth', 'txxxo00000°']]
    word_graph = {}
    word_weight = {}
    
    # create graph
    for text in textlist: # text = tokenized address
        currWord = "graph start"
        for i in range(len(text) + 1): # i = word index in text
            
            # if i + 1== len(text): # if end of text list, change current word to current index (instead of )
            #     currWord = text[i]
                
            # existing graph node
            if currWord in word_graph: # if current word has already been seen before
                found = False
                for j in range(len(word_graph[currWord])):
                    if i == len(text):
                        word_graph[currWord][j][1] += 1
                    elif word_graph[currWord][j][0] == text[i]:
                        word_graph[currWord][j][1] += 1
                        found = True
                        break
                if not found:
                    word_graph[currWord].append([text[i], 1])
            # new word
            else:
                if i== len(text):
                    word_graph[currWord] = [["graph end", 1]]
                    continue
                else:
                    # print(currWord, text[i])
                    word_graph[currWord] = [[text[i], 1]]
            
            # word weight dict
            if text[i] in word_weight:
                    word_weight[text[i]] += 1
            else:
                word_weight[text[i]] = 1
            
            currWord = text[i]
        
    print(word_graph)
    print("\n\n\n\n\n\n")
    print(word_weight)
    
    # find path with largest weights
    currNode = "graph start"
    address = ""
    while currNode != "graph end":
        highCount = 0
        highWord = ""
        for word in word_graph[currNode]: # checks every outgoing edge of current node
            if word[0] == "graph end":
                currNode = "graph end"
                highWord = "graph end"
                break
            if word_weight[word[0]] > highCount:
                highCount = word_weight[word[0]]
                highWord = word[0]
        currNode = highWord
        if currNode != "graph end":
            address += highWord + " "
    print(address)

            
    
    
def get_item_box(im):
    # im = cv2.imread(file_path)
    imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(im, 127, 255, 0)

    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    imgcont = cv2.drawContours(im, contours, -1, (0, 255, 0),3)
    cv2.imshow("cropped contours", cv2.resize(imgcont, (900, 1000)))
    cv2.waitKey(0)



if __name__ == "__main__":
    # pts = find_largest_contour(args.file)

    # # cropped = crop_page(args.file, args.location_list, pts)
    # cropped = cv2.imread(args.file)
    # print(pytesseract.image_to_string(cropped))
    # bottom_right, top_left, text = template_matching(cropped, cv2.imread('rectified/22_crop.jpg'))
    
    
    
    
    spell_checker([])

    # imgcont = cv2.drawContours(cropped, contours, -1, (0, 255, 0),3)
    # cv2.imshow("cropped contours", cv2.resize(cropped, (900, 1000)))
    # cv2.waitKey(0)
    
    # im = cv2.imread(args.file)
    
    # imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    # ret, thresh = cv2.threshold(imgray, 127, 255, 0)

    # contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # imgcont = cv2.drawContours(im, contours, -1, (0, 255, 0),3)
    # cv2.imshow("contours", cv2.resize(imgcont, (900, 1000)))
    # cv2.waitKey(0)
    # cv2.imwrite("rectified/13_rectified_contours.jpg", imgcont)
    
    
    # Apply OCR on the cropped image
    # crop_img = cropped[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
    # cv2.imshow("crop_img", crop_img)
    # cv2.waitKey(0)
    
    # # get_item_box(cropped)
    
    # imgray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
    # kernel = np.ones((1, 1), np.uint8)
    # img = cv2.dilate(imgray, kernel, iterations=1)
    # img = cv2.erode(img, kernel, iterations=1)
    # ret, thresh = cv2.threshold(img, 127, 255, 0)
    
    # kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (27, 27))
    # kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
    # thresh1 = cv2.morphologyEx(thresh, cv2.MORPH_ERODE, kernel1)
    # thresh1 = cv2.morphologyEx(thresh1, cv2.MORPH_DILATE, kernel2)
    
    # crop_img = cv2.threshold(cv2.GaussianBlur(img, (5, 5), 0), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    # gbtext = pytesseract.image_to_string(crop_img)

    # crop_img = cv2.threshold(cv2.bilateralFilter(img, 5, 75, 75), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    # bftext = pytesseract.image_to_string(crop_img)
    
    # crop_img = cv2.threshold(cv2.medianBlur(img, 3), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    # mbtext = pytesseract.image_to_string(crop_img)
    
    # crop_img = cv2.adaptiveThreshold(cv2.GaussianBlur(img, (5, 5), 0), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2)
    # atgbtext = pytesseract.image_to_string(crop_img)
    
    # crop_img = cv2.adaptiveThreshold(cv2.bilateralFilter(img, 9, 75, 75), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2)
    # atbftext = pytesseract.image_to_string(crop_img)
    
    # crop_img = cv2.adaptiveThreshold(cv2.medianBlur(img, 3), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2)
    # atmbtext = pytesseract.image_to_string(crop_img)
    
    # contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # text = pytesseract.image_to_string(crop_img)
    
    # print(gbtext)
    # print(bftext)
    # print(mbtext)
    # print(atgbtext)
    # print(atbftext)
    # print(atmbtext)
    
    # f = open("ocrtext.txt", "a")
    # f.write(text)