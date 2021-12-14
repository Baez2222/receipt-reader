import cv2
import copy
from numpy.core.numeric import empty_like
import pytesseract
import nltk
import numpy as np
import string
import os

BASE_PATH = os.path.dirname(os.path.realpath(__file__))
textlist = []
f = open("templates.txt", "r")
templates = []
for line in f:
    templates.append(line.rstrip())


def template_matching(im):
    # template = cv2.imread('ticket_photos/20210308_234355_crop.jpg')
    # template = cv2.imread('ticket_photos/20210622_124500_crop.jpg')

    # im = cv2.imread('ticket_photos/20210308_235908.jpg')
    # im = cv2.imread('ticket_photos/20210622_123215.jpg')

    # pre process template
    for template in templates:
        im_ = cv2.imread(os.path.join(BASE_PATH, template))
        imgray_ = cv2.cvtColor(im_, cv2.COLOR_BGR2GRAY)
        ret_, thresh_ = cv2.threshold(imgray_, 127, 255, 0)

        # pre process image

        imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(imgray, 127, 255, 0)

        # img2 = thresh.copy()
        w, h = thresh_.shape[::-1]
        # All the 6 methods for comparison in a list
        methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
                    'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']
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
            
            lst = ocrtext(bottom_right, top_left, im)
            if lst:
                textlist.extend(lst)
                
        if textlist:
            correct_top = top_left
            correct_bottom = bottom_right
            break
    return correct_bottom, correct_top, textlist


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
    # print("!!!!!!!!!!!!!!!!!!!!!!")
    # print(text)
    # print("!!!!!!!!!!!!!!!!!!!!!!")
    temptext = []
    for i in text:
        # print(i)
        # lower case
        curr = i.lower()
        
        # remove punctuation
        curr = "".join([char for char in curr if char not in string.punctuation])

        # tokenization
        words = nltk.word_tokenize(curr)
        
        words = [word for word in words if len(word) > 1]
        # print(words)
        # spell = SpellChecker()
        # misspelled = spell.unknown(words)

        correct_text = False
        for word in words:
            if nltk.edit_distance(word, "ship") < 2:
                correct_text = True
        
        if correct_text:
            temptext.append(words)
        # print(text)
        # for word in misspelled:
            # possible.append(spell.correction(word))
            # print(spell.candidates(word))
    # print(temptext)
    return temptext
