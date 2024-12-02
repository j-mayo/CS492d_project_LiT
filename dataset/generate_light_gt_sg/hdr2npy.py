import os
import re
import cv2
import numpy as np

from glob import glob

opj = os.path.join

def read_hdr_file(filepath):
    # Read the HDR image
    hdr_image = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)#, format='HDR-FI')
    if hdr_image is None:
        raise ValueError(f"Failed to read HDR image from {filepath}")
    
    # Convert from BGR to RGB if needed
    if hdr_image.ndim == 3 and hdr_image.shape[2] == 3:
        hdr_image = cv2.cvtColor(hdr_image, cv2.COLOR_BGR2RGB)
    
    # Print the array containing the image data
    #print(hdr_image)
    return hdr_image


if __name__ == "__main__":
    filedir = "./hdrs"
    savedir = "./hdrnpys"
    hdrs = glob(opj(filedir, "*"))
    os.makedirs(savedir, exist_ok=True)
    for fname in hdrs:
        hdr_array = read_hdr_file(fname)
        fsave = re.sub(r'/hdrs/(\d{3})\.hdr$', r'/hdrnpys/\1.npy', fname)
        np.save(fsave, hdr_array)