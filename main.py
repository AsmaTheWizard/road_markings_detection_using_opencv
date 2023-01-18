import cv2
import numpy as np
import matplotlib.pyplot as plt

#------------ Image Helper Functions -------------

def load_img(img_path):
    img = cv2.imread(img_path)
    return img

def display_img(img_title, img):
    cv2.imshow(img_title, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def smooth_img(img):
    """ 
    Smooth image and remove noise to be able to detect sharp changes in image, for this, we use  Gaussian filter
    """
    blur_img = cv2.GaussianBlur(img, (5,5),0)
    display_img('smooth image',blur_img)
    return blur_img 

def canny_edge(img):
    canny_img = cv2.Canny(img, 50, 155)
    display_img('canny image',canny_img)
    return canny_img

def region_of_interest_masking(img):
    img_height = img.shape[0]
    shape = np.array([[(200,img_height),(1100, img_height),(550,250)]])
    img_mask = np.zeros_like(img)
    cv2.fillPoly(img_mask, shape, 255)
    display_img('ROI mask', img_mask)
    img_mask = cv2.bitwise_and(img,img_mask )
    display_img('bitwise img', img_mask)
    return img_mask

def show_lines(img, lines):
    img_lines = np.zeros_like(img)
    if lines is not None:
        for line in lines:
            x1,y1,x2,y2 = line.reshape(4)
            cv2.line(img_lines, (x1,y1),(x2,y2), (255,0,0), 10)
            return img_lines


#from the book
def make_coordinates(image, line_parameters):
          slope, intercept = line_parameters
          y1 = image.shape[0]
          y2 = int(y1*(3/5))
          x1 = int((y1- intercept)/slope)
          x2 = int((y2 - intercept)/slope)
          return np.array([x1, y1, x2, y2])

#from book  
def average_slope_intercept(image, lines):
          left_fit = []
          right_fit = []
          for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            parameter = np.polyfit((x1, x2), (y1, y2), 1)
            slope = parameter[0]
            intercept = parameter[1]
            if slope < 0:
              left_fit.append((slope, intercept))
            else:
              right_fit.append((slope, intercept))
          left_fit_average =np.average(left_fit, axis=0)
          right_fit_average = np.average(right_fit, axis =0)
          left_line =make_coordinates(image, left_fit_average)
          right_line = make_coordinates(image, right_fit_average)
  
          return np.array([left_line, right_line])


#------------ Color Functions -------------
def convert_to_grayscale(img):
    lane_lines = np.copy(img)
    gray_img = cv2.cvtColor(lane_lines, cv2.COLOR_RGB2GRAY)
    display_img('grayscale image',gray_img)
    return gray_img



def main():
    img = load_img('img/test_image.jpeg')
    lane_img = np.copy(img)
    display_img('original image',img)
    gray_img = convert_to_grayscale(lane_img)
    sm_img = smooth_img(gray_img)
    canny_img = canny_edge(sm_img)
    cropped_img = region_of_interest_masking(canny_img)
    lines = cv2.HoughLinesP(cropped_img, 2,  np.pi/180, 100, np.array([]),  minLineLength = 40, maxLineGap=5 )
    averaged_lines = average_slope_intercept(lane_img, lines)

    line_img = show_lines(lane_img, averaged_lines)
    display_img('lines image',lane_img)
    
    comb_img = cv2.addWeighted(lane_img, 0.8, line_img, 1,1 )
    display_img('combine image',comb_img)
    

if __name__ == '__main__':
    main()
