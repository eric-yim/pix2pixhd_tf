import numpy as np
def grid_imgs(imgs,cols=3,buffer=2):
    """
    Turns images into a single image grid
    imgs must be same size
    """
    if isinstance(imgs,list):
        imgs = np.array(imgs)
    shape = imgs[0].shape
    rows = len(imgs)//cols
    if len(imgs) % cols != 0:
        rows+=1
    img_new = np.zeros(((shape[0]+buffer)*rows-buffer,(shape[1]+buffer)*cols-buffer,shape[2]),dtype=imgs.dtype) + 255
    for i,img in enumerate(imgs):
        r = i//cols
        c = i % cols
        img_new[r*(shape[0]+buffer):r*(shape[0]+buffer)+shape[0],c*(shape[1]+buffer):c*(shape[1]+buffer)+shape[1],:]=img
    return img_new
