import cv2
import numpy as np

def preprocess_drawing(image_path):
    # 1) Load & upscale for quality
    orig = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    orig = cv2.resize(orig, (2048, 2048), interpolation=cv2.INTER_LANCZOS4)

    # 2) Crop away top 30% & bottom 25%
    h, w = orig.shape
    cropped = orig[int(h*0.30):int(h*0.75), int(w*0.05):int(w*0.95)]

    # 3) Further refine crop (optional)
    h2, w2 = cropped.shape
    cropped = cropped[int(h2*0.05):int(h2*0.80), int(w2*0.05):int(w2*0.95)]

    # 4) Binarize just to find shapes (lines→white, bg→black)
    _, bw = cv2.threshold(cropped, 200, 255, cv2.THRESH_BINARY_INV)

    # 5) Close gaps so main shape is one component
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel, iterations=2)

    # 6) Keep only the largest connected component mask
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(bw, 8)
    if num_labels <= 1:
        mask = bw.copy()
    else:
        areas = stats[1:, cv2.CC_STAT_AREA]
        largest = 1 + int(np.argmax(areas))
        mask = (labels == largest).astype(np.uint8) * 255

    # 7) Apply that mask to the original cropped grayscale, preserving thickness
    #    Everything outside mask → white (255), inside mask → original pixel
    fg = cv2.bitwise_and(cropped, cropped, mask=mask)
    bg = np.ones_like(cropped, dtype=np.uint8) * 255
    preserved = np.where(mask[...,None]==255, fg[...,None], bg[...,None])[:,:,0]

    # 8) Place onto a white A4 canvas
    a4 = 255 * np.ones((3508, 2480), np.uint8)
    mh, mw = preserved.shape
    yoff = (3508 - mh)//2
    xoff = (2480 - mw)//2
    a4[yoff:yoff+mh, xoff:xoff+mw] = preserved

    return a4