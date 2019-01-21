import cv2
import numpy as np
def detectred(image,hsv):
    #Mask for red planes
    min_red = np.array([0, 100, 80])
    max_red = np.array([10, 256, 256])
    mask1 = cv2.inRange(hsv, min_red, max_red)
    min_red2 = np.array([170, 100, 80])
    max_red2 = np.array([180, 256, 256])
    mask2 = cv2.inRange(hsv, min_red2, max_red2)
    mask0 = mask1 + mask2

    #CLeanse red mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
    kernel2 = np.ones((4,4),np.uint8)
    erosion = cv2.erode(mask0,kernel2,iterations = 2)
    dilation = cv2.dilate(erosion,kernel,iterations = 2)
    mask0=dilation
    mask_closed = cv2.morphologyEx(mask0, cv2.MORPH_CLOSE, kernel)
    mask_clean = cv2.morphologyEx(mask_closed, cv2.MORPH_OPEN, kernel)
    #frame1=cv2.bitwise_and(image,image,mask=mask0)
    #cv2.imshow("Debug",mask_clean)

    #Draw cirles around contours
    im2, contours, hierarchy = cv2.findContours(mask_clean, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #cv2.drawContours(image,contours,-1,(0,255,0),6)
    print("Number of red arrows detected=",len(contours))
    for i in contours:
        print(i)
        (x,y),radius = cv2.minEnclosingCircle(i)
        center = (int(x),int(y))
        radius = int(radius)
        cv2.circle(image,center,radius,(0,255,0),2)
    return image

def detectblack(image,hsv):
    #mask for black planes
    min_black0 = np.array([17, 15, 100])
    max_black0 = np.array([50, 56, 200])
    min_black1 = np.array([86, 31, 4])
    max_black1 = np.array([220, 88, 50])
    min_black2 = np.array([25, 146, 190])
    max_black2 = np.array([62, 174, 250])
    min_black3 = np.array([145, 133, 128])
    max_black3 = np.array([103, 86, 65])
    mask00 = cv2.inRange(hsv, min_black0, max_black0)
    mask11 = cv2.inRange(hsv, min_black1, max_black1)
    mask22 = cv2.inRange(hsv, min_black2, max_black2)
    mask33 = cv2.inRange(hsv, min_black3, max_black3)
    mask3=mask00+mask11+mask22+mask33


    #CLeanse the mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    kernel2 = np.ones((2,3),np.uint8)
    erosion = cv2.erode(mask3,kernel2,iterations = 1)
    dilation = cv2.dilate(erosion,kernel,iterations = 4)
    mask3=dilation
    mask_closed2 = cv2.morphologyEx(mask3, cv2.MORPH_CLOSE, kernel)
    mask_clean2 = cv2.morphologyEx(mask_closed2, cv2.MORPH_OPEN, kernel)
    #cv2.imshow("Degug",mask_clean2)

    #Draw contours
    im3, contours2, hierarchy2 = cv2.findContours(mask_clean2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #cv2.drawContours(image,contours2,-1,(0,255,0),6)
    print("Number of support aircraft detected=",len(contours2))
    for i in contours2:
        print(i)
        (x,y),radius = cv2.minEnclosingCircle(i)
        center = (int(x),int(y))
        radius = int(radius)
        cv2.circle(image,center,radius,(0,0,255),2)
    return image

def main():
    windowName1="Original Image"
    windowName2="Processed Image"
    image = cv2.imread('C:\\Users\\user\\Desktop\\Camera\\test5.jpg')
    #resize cause why not
    max_dimension = max(image.shape)
    scale = 700/max_dimension
    image = cv2.resize(image, None, fx=scale, fy=scale)
    hsv= cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    cv2.imshow(windowName1,image)
    image=detectred(image,hsv)
    image=detectblack(image,hsv)
    cv2.imshow(windowName2,image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__=="__main__":
    main()
