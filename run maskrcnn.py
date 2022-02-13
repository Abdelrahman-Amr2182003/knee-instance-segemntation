from tqdm import tqdm

import sys
sys.path.append("mrcnn")
from m_rcnn import *
import cv2
from visualize import *
import glob    
test_model, inference_config =load_inference_model(1,"mask_rcnn_object_0005.h5")
imgs=glob.glob('*.jpg')
for ii in imgs[0:30]:
    
    img = cv2.imread(ii)
    h,w,c=img.shape
    image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Detect results
    r = test_model.detect([image])[0]
    colors = random_colors(80)

    object_count = len(r["class_ids"])
    for i in range(object_count):
        mask = r["masks"][:, :, i]
        contours = get_mask_contours(mask)
        for cnt in contours:
            cv2.polylines(img, [cnt], True, colors[i], 2)#only the outlines      
    cv2.imshow('img',cv2.resize(img,(700,700)))
    cv2.waitKey(0)
    cv2.destroyAllWindows()