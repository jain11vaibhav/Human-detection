from tensorflow.compat.v1 import InteractiveSession
from tensorflow.compat.v1 import ConfigProto
from imageai.Detection import ObjectDetection
from scipy.ndimage import zoom
import scipy.io
import numpy as np
import cv2
import os
import glob
import shutil
import pickle
import random
import threading
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

SAMPLE_DIR = "./sample/"
DATASET_DIR = "./dataset2/"
GROUND_TRUTH = "./ground_truth2/"
OLD_DATASET_DIR = "./tdataset/"
OUTPUT = './output2/'


config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

image_size = (384, 384, 3)

def cv2_clipped_zoom(img, zoom_factor):
    """
    Center zoom in/out of the given image and returning an enlarged/shrinked view of 
    the image without changing dimensions
    Args:
        img : Image array
        zoom_factor : amount of zoom as a ratio (0 to Inf)
    """
    height, width = img.shape[:2]  # It's also the final desired shape
    new_height, new_width = int(height * zoom_factor), int(width * zoom_factor)

    # Crop only the part that will remain in the result (more efficient)
    # Centered bbox of the final desired size in resized (larger/smaller) image coordinates
    y1, x1 = max(0, new_height - height) // 2, max(0, new_width - width) // 2
    y2, x2 = y1 + height, x1 + width
    bbox = np.array([y1, x1, y2, x2])
    # Map back to original image coordinates
    bbox = (bbox / zoom_factor).astype(np.int)
    y1, x1, y2, x2 = bbox
    cropped_img = img[y1:y2, x1:x2]

    # Handle padding when downscaling
    resize_height, resize_width = min(
        new_height, height), min(new_width, width)
    pad_height1, pad_width1 = (
        height - resize_height) // 2, (width - resize_width) // 2
    pad_height2, pad_width2 = (height - resize_height) - \
        pad_height1, (width - resize_width) - pad_width1
    pad_spec = [(pad_height1, pad_height2), (pad_width1,
                                             pad_width2)] + [(0, 0)] * (img.ndim - 2)

    result = cv2.resize(cropped_img, (resize_width, resize_height))
    result = np.pad(result, pad_spec, mode='constant')
    assert(result.shape[0] == height and result.shape[1] == width)
    return result


# def randomise_Dataset():
#     if os.path.exists(DATASET_DIR):
#         shutil.rmtree(DATASET_DIR)
#     os.mkdir(DATASET_DIR)
#     for folder in os.listdir(SAMPLE_DIR):
#         print(folder)
#         os.mkdir(DATASET_DIR + folder)
#         for f in os.listdir(SAMPLE_DIR + folder):
#             if f.endswith(".mat"):
#                 print(f)
#                 f1 = SAMPLE_DIR + folder + '/' + f.split('.')[0] + ".jpg"
#                 img = cv2.imread(f1)
#                 new_img = np.zeros(img.shape)
#                 for i in range(32):
#                     scale = random.uniform(0.2, 1.2)
#                     temp = cv2_clipped_zoom(img[(i*384)+0:(i*384)+384, :, :], scale)
#                     new_img[(i*384)+0:(i*384)+384, :, :] = temp
#                 cv2.imwrite(DATASET_DIR + folder + '/' + f.split('.')[0] + ".jpg", new_img)


# def prepare_Ground_Truth():
#     if os.path.exists(GROUND_TRUTH):
#         shutil.rmtree(GROUND_TRUTH)
#     os.mkdir(GROUND_TRUTH)
#     execution_path = os.getcwd()
#     image_size = (384, 384, 3)

#     print("Loading Model")
#     detector = ObjectDetection()
#     # detector.setModelTypeAsRetinaNet()
#     detector.setModelTypeAsYOLOv3()
#     detector.setModelPath(os.path.join(
#         execution_path, "trained models/yolo.h5"))
#     detector.loadModel()
#     custom = detector.CustomObjects(person=True)
#     print("Loaded Model")

#     for folder in os.listdir(DATASET_DIR):
#         print(folder)
#         os.mkdir(GROUND_TRUTH + folder)
#         for f in os.listdir(DATASET_DIR + folder):
#             print(f)
#             img = cv2.imread(DATASET_DIR + folder + '/' + f)
#             mat = scipy.io.loadmat(SAMPLE_DIR + folder + '/' + f.split('.')[0] + '.mat')
#             mat_data = np.array(mat['keypoints'])
#             data = []
#             for i in range(32):
#                 frame, detections = detector.detectCustomObjectsFromImage(custom_objects=custom, display_object_name=False, display_percentage_probability=False, input_image=img[(
#                     i*384)+0:(i*384)+384, :, :], input_type="array", output_type="array", minimum_percentage_probability=1)
#                 temp = {}
#                 if(len(detections) == 1):
#                     j = 8
#                     if(mat_data[i, j, 2] == 1 or mat_data[i, j, 2] == 0):
#                         center_x = int(mat_data[i, j, 0] * image_size[0])
#                         center_y = int(mat_data[i, j, 1] * image_size[1])
#                     else:
#                         print('error   ', mat_data[i, j, 2])
#                     # center_x = (detections[0]['box_points'][0] + detections[0]['box_points'][2])/2
#                     # center_y = (detections[0]['box_points'][1] + detections[0]['box_points'][3])/2
#                     height = detections[0]['box_points'][3] - \
#                         detections[0]['box_points'][1]
#                     width = detections[0]['box_points'][2] - \
#                         detections[0]['box_points'][0]
#                     # cv2.circle(frame, (int(center_x), int(center_y)), int(8), (255, 255, 255), -1)
#                     # cv2.imwrite('./output/'+str(i)+'.jpg',frame)
#                     temp['center'] = (center_x, center_y)
#                     temp['height'] = height
#                     temp['width'] = width
#                 else:
#                     j = 8
#                     if(mat_data[i, j, 2] == 1 or mat_data[i, j, 2] == 0):
#                         center_x = int(mat_data[i, j, 0] * image_size[0])
#                         center_y = int(mat_data[i, j, 1] * image_size[1])
#                         height = (mat_data[i, 9, 1] -
#                                   mat_data[i, 15, 1]) * image_size[1]
#                     else:
#                         print(mat_data[i, j, 2])
#                         print("error")
#                     temp['center'] = (center_x, center_y)
#                     temp['height'] = height
#                     temp['width'] = 0
#                 data.append(temp)
#             pickle_out = open(GROUND_TRUTH + folder + '/' +
#                               f.split('.')[0] + ".pickle", "wb")
#             pickle.dump(data, pickle_out)
#             pickle_out.close()


# def prepare_Ground_Truth(folder):
#     # if os.path.exists(GROUND_TRUTH):
#     #     shutil.rmtree(GROUND_TRUTH)
#     # os.mkdir(GROUND_TRUTH)

#     if(True):
#         print(folder)
#         os.mkdir(GROUND_TRUTH + folder)
#         files = os.listdir(DATASET_DIR + folder)
#         length = len(files)
#         count = 0
#         for f in files:
#             print(f, "   Completed -> ", count*100/length,"%")
#             count = count + 1 
#             img = cv2.imread(DATASET_DIR + folder + '/' + f)
#             mat = scipy.io.loadmat(SAMPLE_DIR + folder + '/' + f.split('.')[0] + '.mat')
#             mat_data = np.array(mat['keypoints'])
#             data = []
#             for i in range(32):
#                 temp = {}
#                 found_start = 0
#                 found_end = 0
#                 temp_image = img[(i*384)+0:(i*384)+384, :, :]
#                 spine = 26
#                 if(mat_data[i, spine, 2] == 1 or mat_data[i, spine, 2] == 0):
#                     center_x = int(mat_data[i, spine, 0] * image_size[0])
#                     center_y = int(mat_data[i, spine, 1] * image_size[1])
#                 else:
#                     print('error   ', mat_data[i, spine, 2])
                
#                 for row in range(image_size[0]):
#                     for col in range(image_size[1]):
#                         if(found_start == 0 and (temp_image[row, col, :] != [0, 0, 0]).all()):
#                             found_start = 1
#                             start = row
#                         if(found_end == 0 and (temp_image[image_size[0] - row - 1, image_size[1] - col - 1, :] != [0, 0, 0]).all()):
#                             found_end = 1
#                             end = image_size[0] - row - 1
#                         if(found_end !=0 and found_start!=0):
#                             break
#                 height = end - start
#                 temp['center'] = (center_x, center_y)
#                 temp['height'] = height
#                 data.append(temp)
#             pickle_out = open(GROUND_TRUTH + folder + '/' +
#                               f.split('.')[0] + ".pickle", "wb")
#             pickle.dump(data, pickle_out)
#             pickle_out.close()

def prepareDataset(folder):
    print(folder)
    # debug = 1
    os.mkdir(DATASET_DIR + folder)
    os.mkdir(GROUND_TRUTH + folder)
    files = os.listdir(OLD_DATASET_DIR + folder)
    length = len(files)
    count = 0

    for f in files:
        img = cv2.imread(SAMPLE_DIR + folder + '/' + f)
        mat = scipy.io.loadmat(SAMPLE_DIR + folder + '/' + f.split('.')[0] + '.mat')
        mat_data = np.array(mat['keypoints'])
        new_img = np.zeros(img.shape)
        # out_image = np.zeros(img.shape)
        data = []
        for i in range(32):
            scale = random.uniform(0.2, 1.2)
            temp_image = cv2_clipped_zoom(img[(i*384)+0:(i*384)+384, :, :], scale)
            new_img[(i*384)+0:(i*384)+384, :, :] = temp_image

            temp = {}
            found_start = 0
            found_end = 0
            spine = 26
            if(mat_data[i, spine, 2] == 1 or mat_data[i, spine, 2] == 0):
                center_x = mat_data[i, spine, 0] * image_size[0]
                center_y = mat_data[i, spine, 1] * image_size[1]
                center_x = int(image_size[0]/2 + (center_x - image_size[0]/2) * scale)
                center_y = int(image_size[1]/2 + (center_y - image_size[1]/2) * scale)
            else:
                print('error   ', mat_data[i, 25, 2])
            
            for row in range(image_size[0]):
                for col in range(image_size[1]):
                    if(found_start == 0 and (temp_image[row, col, :] != [0, 0, 0]).all()):
                        found_start = 1
                        start = row
                    if(found_end == 0 and (temp_image[image_size[0] - row - 1, image_size[1] - col - 1, :] != [0, 0, 0]).all()):
                        found_end = 1
                        end = image_size[0] - row - 1
                    if(found_end !=0 and found_start!=0):
                        break

            
            height = end - start
            temp['center'] = (center_x, center_y)
            temp['height'] = height
            data.append(temp)
            # if(debug == 1):
            #     out_image[(i*384)+0:(i*384)+384, :, :] = temp_image
            #     cv2.circle(out_image, (center_x, (i*384)+center_y), int(image_size[0]/32), (255, 255, 255), -1)
            #     cv2.line(out_image,(center_x,int((i*384)+center_y-(height/2))),(center_x,int((i*384)+center_y+(height/2))),(255,255,255),5)
        
        cv2.imwrite(DATASET_DIR + folder + '/' + f.split('.')[0] + ".jpg", new_img)
        # if(debug == 1):
        #     cv2.imwrite(OUTPUT + f.split('.')[0] + ".jpg", out_image)

        pickle_out = open(GROUND_TRUTH + folder + '/' + f.split('.')[0] + ".pickle", "wb")
        pickle.dump(data, pickle_out)
        pickle_out.close()
        count = count + 1 
        print(f, "   Completed -> ", count*100/length,"%")


prepareDataset('9')
