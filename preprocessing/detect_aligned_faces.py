import sys
import math
sys.path.append("./face_frontalization")
import frontalize
import facial_feature_detector as feature_detection
import camera_calibration as calib
import scipy.io as io
import cv2
import numpy as np
import os
import check_resources as check
import matplotlib.pyplot as plt
import dlib
import os


this_path = os.path.dirname(os.path.abspath(__file__))
image_path ="./test_converted_list.txt"


def face_detect(image_file_path,image_save_path):
    # check for dlib saved weights for face landmark detection
    # if it fails, dowload and extract it manually from
    # http://sourceforge.net/projects/dclib/files/dlib/v18.10/shape_predictor_68_face_landmarks.dat.bz2
    
    # load detections performed by dlib library on 3D model and Reference Image
    # load query image
    #print image_file_path
    img = cv2.imread(image_file_path, 1)

    (ori_w, org_h, org_channel) = img.shape
    
    check.check_dlib_landmark_weights() 
    lmarks = feature_detection.get_landmarks(img)

    
    
    if lmarks.shape[0] == 0:      
       return 0
    else:
        print image_save_path
	if not os.path.exists(image_save_path):
	    os.makedirs(image_save_path)
        for num_face in range(0,lmarks.shape[0]):
	    face_landmarks = lmarks[num_face,:,:]
	    print "Detect face %d out of total %d faces" %((num_face+1), lmarks.shape[0])
	    print "start to align face"
	    w = 256;
	    h = 256;
            
	    eyecornerDst = [ (np.int(0.25 * w ), np.int(h / 3.5)), (np.int(0.75 * w ), np.int(h / 3.5)), (np.int(0.5 * w ), np.int(h / 2)) ];
	    eyecornerSrc = [ (np.int(face_landmarks[36][0]), np.int(face_landmarks[36][1])),( np.int(face_landmarks[45][0]) , np.int(face_landmarks[45][1])), (np.int(face_landmarks[30][0]), np.int(face_landmarks[30][1]))] ;
	   
	    
              
            # Apply similarity transformation
	    tform = similarityTransform(eyecornerSrc, eyecornerDst);

	         
	    img_face_aligned = cv2.warpAffine(img, tform, (w,h));
            #note that x is horizontal, is width, while y is height, is vertical
            #print tform

	    width = np.int((face_landmarks[16][0]-face_landmarks[0][0])*1.3)
            height = np.int((face_landmarks[8][1]-face_landmarks[19][1])*1.3)
            if width<0 | height<0:
		print("face crooked")
		sys.exit()
           
            left_top_x = np.int(face_landmarks[0][0])-np.int((np.int(face_landmarks[16][0])-np.int(face_landmarks[0][0]))*0.15)
            left_top_y = np.int(face_landmarks[19][1])-np.int((np.int(face_landmarks[8][1])-np.int(face_landmarks[19][1]))*0.15)
            if left_top_x<0:
		left_top_x =0
	    if left_top_y<0:
	        left_top_y=0
	    image_new_name_affined = image_save_path + '/A_' + str(left_top_x) +'_'+str(left_top_y) +'_'+ str(width) +'_'+ str(height) +'.jpg'
                   
	    resized_image = cv2.resize(img_face_aligned, (256, 256))

	    cv2.imwrite(image_new_name_affined, resized_image)
            
	return 1

     
   
def similarityTransform(inPoints, outPoints) :
    s60 = math.sin(60*math.pi/180);
    c60 = math.cos(60*math.pi/180);  
  
    inPts = np.copy(inPoints).tolist();
    outPts = np.copy(outPoints).tolist();
    
    xin = c60*(inPts[0][0] - inPts[1][0]) - s60*(inPts[0][1] - inPts[1][1]) + inPts[1][0];
    yin = s60*(inPts[0][0] - inPts[1][0]) + c60*(inPts[0][1] - inPts[1][1]) + inPts[1][1];
    
    inPts.append([np.int(xin), np.int(yin)]);
    
    xout = c60*(outPts[0][0] - outPts[1][0]) - s60*(outPts[0][1] - outPts[1][1]) + outPts[1][0];
    yout = s60*(outPts[0][0] - outPts[1][0]) + c60*(outPts[0][1] - outPts[1][1]) + outPts[1][1];
    
    outPts.append([np.int(xout), np.int(yout)]);
    
    tform = cv2.estimateRigidTransform(np.array([inPts]), np.array([outPts]), False);
    
    return tform;


if __name__ == "__main__":

    f = open(image_path, 'r')
    f_lines = f.readlines()
    f.close()
    num_total_fail = 0
    undetected_images=[]
    detect_alignfail=0
    alignfail_images=[]
   
    for ix, line in enumerate(f_lines):
	image_file = line.split('\n')[0]
        print image_file
        gif_check = image_file[-3:];
        # This face detection method cannot detect faces from gif images, so need to check it first. 
        #print gif_check
        if gif_check == 'gif':	    
	    print "There is still a gif file, convert it to jpg first. "
            break
        image_name = image_file.split('/')[-1];
        image_file_path = os.pardir + image_file

        image_save_path = os.pardir + "/data/Test_faces_aligned/" +image_name[:-4]

        flag = face_detect(image_file_path,image_save_path)
        if flag==0:
            undetected_images.append(image_file)
	    print "No face detected "
            num_total_fail=num_total_fail+1


    print 'Total number of undetected images is %d' %num_total_fail
    print 'Images names are'
    print undetected_images

            
        
