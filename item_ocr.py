import argparse
import cv2
import pytesseract


parser = argparse.ArgumentParser(description='Locating bounding box in the image')
parser.add_argument('file', type=str,
                    help='path to file')
args = parser.parse_args()


def get_item_box(im):
    im = cv2.imread(im)
    imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(imgray, 127, 255, 0)

    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    imgcont = cv2.drawContours(im, contours, -1, (0, 255, 0),3)
    cv2.imshow("contours", cv2.resize(imgcont, (900, 1000)))
    cv2.waitKey()
    
    # # Grayscale, Gaussian blur, Otsu's threshold
    # image = cv2.imread(im)
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # blur = cv2.GaussianBlur(gray, (3,3), 0)
    # thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # # Morph open to remove noise and invert image
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    # opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    # invert = 255 - opening

    # Perform text extraction
    data = pytesseract.image_to_string(imgray, lang='eng', config='--psm 4')
    print(data)



if __name__ == "__main__":
    # pts = find_largest_contour(args.file)

    # # cropped = crop_page(args.file, args.location_list, pts)
    # cropped = cv2.imread(args.file)
    # print(pytesseract.image_to_string(cropped))
    # bottom_right, top_left, text = template_matching(cropped, cv2.imread('rectified/22_crop.jpg'))
    
    get_item_box(args.file)