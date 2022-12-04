import cv2

filename = 'img.png'

def main():
    # create a HOGDescriptor object
    hog = cv2.HOGDescriptor()
    
    #Initialize the People Detector
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    
    #load an image 
    img = cv2.imread(filename)
    
    # Detect People
    (bounding_boxes, weights) = hog.detectMultiScale(img, 
                                                     winStride = (4,4),
                                                     padding = (8,8), 
                                                     scale = 1.05)
    #Draw bounding boxes on the image
    for (x,y, w, h) in bounding_boxes :
        cv2.rectangle(img , (x,y) , (x+w , y+h), (0,255,0), 4)
        
        
        
        cv2.imshow('img', img)
        
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    
main()

    
    
    
    