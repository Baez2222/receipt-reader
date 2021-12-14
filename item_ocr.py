import argparse
import cv2
import pytesseract
import os
import numpy as np
from rectification import find_largest_contour, order_rectpts
import string
import nltk


parser = argparse.ArgumentParser(description='Locating bounding box in the image')
parser.add_argument('file', type=str,
                    help='path to file')
args = parser.parse_args()

# output folder for table cropped images
BASE_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), "table_images")

def get_item_box(FILE_PATH):
    img = cv2.imread(FILE_PATH)
    imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(imgray, 127, 255, 0)

    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
    # imgcont = cv2.drawContours(img, sorted_contours[1], -1, (0, 255, 0),3)
    # cv2.imshow("contours", cv2.resize(imgcont, (900, 1000)))
    # cv2.waitKey()
    
    get_table_contour(img, FILE_PATH)
    
    return get_msf(imgray)
    

def get_table_contour(im, FILE_PATH):
    try:
        points = find_largest_contour(im, largest=False)
        lowx, lowy, hix, hiy = order_rectpts(points)
        
        if lowy > im.shape[0] * 0.2:
            im = im[int(lowy*0.7):, :]
            points = find_largest_contour(im, largest=False)
        points = np.float32(points)
        return crop_table(im, points, FILE_PATH)
    
    except:
        print("Wrong rectangle points. Retrying...")
        points = find_largest_contour(im, method=2, largest=False)
        lowx, lowy, hix, hiy = order_rectpts(points)
        
        if lowy > im.shape[0] * 0.2:
            im = im[int(lowy*0.7):, :]
            points = find_largest_contour(im, largest=False)
        points = np.float32(points)
        return crop_table(im, points, FILE_PATH)


def crop_table(im, pts, FILE_PATH):
    # conver the minAreaRect output to points
    # pts = cv2.boxPoints(pts)
    # contours must be of shape (n, 1, 2) and dtype integer
    pts = pts.reshape(-1, 1, 2)
    pts = pts.astype(int)
    
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
    # print(lowx,lowy)
    # print(hix,hiy)
    # print()
    crop_img = im[lowy:hiy, lowx:hix]
    # cv2.imshow("cropped", cv2.resize(crop_img, (800, 1000)))
    # cv2.waitKey(0)
    # cv2.imshow("cropped", cv2.resize(im, (800, 1000)))
    crop_img_path = os.path.join(BASE_PATH, os.path.basename(FILE_PATH))
    # print(crop_img_path)
    # print(crop_img_path)
    cv2.imwrite(crop_img_path, crop_img)

def get_msf(im):
    msf = 0
    # Perform text extraction
    text = pytesseract.image_to_string(im, lang='eng', config='--psm 4')
    # print(text)
    # print(type(text))
    for line in text.splitlines():
        # print(i)
        # lower case
        curr = line.lower()
        
        # # remove punctuation
        # curr = "".join([char for char in curr if char not in string.punctuation])

        # tokenization
        words = nltk.word_tokenize(curr)
        
        words = [word for word in words if len(word) > 1]
        
        # print(words)
        try:
            if nltk.edit_distance(words[0], "drywall") < 2:
                for word in words:
                    word = word.replace(",", ".", 1)
                    try:
                        msf = float(word)
                        break
                    except ValueError:
                        continue
        except:
            continue
    return msf
    
if __name__ == "__main__":
    # pts = find_largest_contour(args.file)

    # # cropped = crop_page(args.file, args.location_list, pts)
    # cropped = cv2.imread(args.file)
    # print(pytesseract.image_to_string(cropped))
    # bottom_right, top_left, text = template_matching(cropped, cv2.imread('rectified/22_crop.jpg'))
    
    get_item_box(args.file)