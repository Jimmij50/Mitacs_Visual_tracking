import cv2
import numpy as np
import math
from collections import namedtuple
import matplotlib.pyplot as plt
import copy
POS=namedtuple('POS',['x','y'])
ix1,iy1=-1,-1
ix2,iy2=-1,-1
def padding_ex(img,para=1.5):
    size_x=img.shape[1]
    size_y=img.shape[0]
    pad=np.zeros([int(size_y*para),int(size_x*para),3],dtype='uint8')
    center_x=int(size_x*para)//2
    center_y=int(size_y*para)//2
    left=center_x-size_x//2
    top=center_y-size_y//2
    pad[top:top+size_y,left:left+size_x,:]=img[:,:,:]
    for i in range(top):
        pad[i,left:left+size_x]=img[0,:]
    for i in range(top+size_y,pad.shape[0]):
        pad[i,left:left+size_x]=img[size_y-1,:]
    for i in range(left):
        pad[:,i]=pad[:,left]
    for i in range(left+size_x,pad.shape[1]):
        pad[:,i]=pad[:,left+size_x-1]
    return pad
# def it(re):
    
#     re[i]=i+20
def padding(x1,y1,x2,y2,para=1.5):
    center_x=(x1+x2)//2
    center_y=(y1+y2)//2
    size_x=int((x2-x1)*para)
    size_y=int((y2-y1)*para)
    x1_p=center_x-size_x//2
    x2_p=x1_p+size_x
    y1_p=center_y-size_y//2
    y2_p=y1_p+size_y
    return x1_p,y1_p,x2_p,y2_p

#不归一化
# def Gaussian2(img_in,sigma1=0.5,sigma2=0.5):
#     x=img_in.shape[0]
#     y=img_in.shape[1]
#     C_X=int(np.around(x/2))
#     C_Y=int(np.around(y/2))
#     filter_g=np.zeros([x,y])
#     for i in range(x):
#         for j in range(y):
#             filter_g[i,j]=np.around(1/(2*math.pi*sigma1*sigma2)*math.exp(-((i-C_X)**2+(j-C_Y)**2)/(2*sigma1*sigma2)),decimals=2)

   
#     return filter_g


#归一化
def Gaussian2(img_in,sigma1=0.5,sigma2=0.5):
    x=img_in.shape[0]
    y=img_in.shape[1]
    C_X=int(np.around(x/2))
    C_Y=int(np.around(y/2))
    filter_g=np.zeros([x,y])
    for i in range(x):
        for j in range(y):
            filter_g[i,j]=np.around(1/(2*math.pi*sigma1*sigma2)*math.exp(-((i-C_X)**2+(j-C_Y)**2)/(2*sigma1*sigma2)),decimals=2)
    max_index=np.argmax(filter_g)
    max_index=np.unravel_index(max_index,(filter_g.shape[0],filter_g.shape[1]))
    max=filter_g[max_index]
    #filter_g/max
    for i in range(x):
        for j in range(y):
            filter_g[i,j]=filter_g[i,j]/max
    #print(filter_g)
    #print(max)
    return filter_g


def window(img):
    '''
multiply a img with a hanning window funciton
    '''
    cos_window=np.outer(np.hanning(img.shape[0]),np.hanning(img.shape[1]))
    win_patch=np.multiply(img,cos_window[:, :, None])
    win_patch=win_patch
    max_i=np.argmax(win_patch)
    index=np.unravel_index(max_i,win_patch.shape)
    max=win_patch[index]
    win_patch=win_patch/max
    return win_patch
def get_subwindow_list(img,template):
    subwindow_list=[]
    size_x=template.shape[1]
    size_y=template.shape[0]
    subwindow=np.zeros([size_y,size_x,3],dtype='uint8')
    for i in range(10,img.shape[1],20):# x
        for j in range(10,img.shape[0],20):#y
            left=i-size_x//2 if (i-size_x//2)>0 else 0
            top=j-size_y//2 if (j-size_y//2)>0 else 0
            right=i+size_x//2-1 if(i+size_x//2-1)<img.shape[1]-1 else img.shape[1]-1
            down=j+size_y//2-1 if (j+size_y//2-1)<img.shape[0]-1 else img.shape[0]-1
            size_x_img=right-left
            size_y_img=down-top
            left_t=size_x//2-size_x_img//2
            top_t=size_y//2-size_y_img//2 
            right_t=left_t+size_x_img
            down_t=top_t+size_y_img
            # print(size_x_img)
            # print(left,right)
            # print(top_t,left_t)
            subwindow=subwindow*0
            subwindow[top_t:down_t,left_t:right_t]=img[top:down,left:right]
            for ii in range(top_t):
                subwindow[ii,left_t:right_t]=img[top,left:right]
            for ii in range(down_t,template.shape[0]):
                subwindow[ii,left_t:right_t]=img[down,left:right]
            for ii in range(left_t):
                subwindow[:,ii]=subwindow[:,left_t]
            for ii in range(right_t,template.shape[1]):
                subwindow[:,ii]=subwindow[:,right_t-1]
            subwindow_list.append(((subwindow),POS(i,j)))
    return subwindow_list

def calculate_K(sigma, x, z=None):
    """
    Gaussian Kernel with dense sampling.
    Evaluates a gaussian kernel with bandwidth SIGMA for all displacements
    between input images X and Y, which must both be MxN. They must also
    be periodic (ie., pre-processed with a cosine window). The result is
    an MxN map of responses.

    If X and Y are the same, ommit the third parameter to re-use some
    values, which is faster.
    :param sigma: feature bandwidth sigma
    :param x:
    :param y: if y is None, then we calculate the auto-correlation
    :return:
    """

    xf=np.fft.fft2(x,axes=(0,1))
    N = xf.shape[0] * xf.shape[1]
    xx = np.dot(x.flatten().transpose(), x.flatten())  # squared norm of x

    if z is None:
        # auto-correlation of x
        zf = xf
        zz = xx
    else:
        zz = np.dot(z.flatten().transpose(), z.flatten())  # squared norm of y
        zf=np.fft.fft2(z,axes=(0,1))

    xyf = np.multiply(zf, np.conj(xf))
    if len(xyf.shape) == 3:
        xyf_ifft = np.fft.ifft2(np.sum(xyf, axis=2))
    elif len(xyf.shape) == 2:
        xyf_ifft = np.fft.ifft2(xyf)
            # elif len(xyf.shape) == 4:
            #     xyf_ifft = np.fft.ifft2(np.sum(xyf, axis=3))

    #row_shift, col_shift = np.floor(np.array(xyf_ifft.shape) / 2).astype(int)
    #xy_complex = np.roll(xyf_ifft, row_shift, axis=0)
    #xy_complex = np.roll(xy_complex, col_shift, axis=1)
    c = np.real(xyf_ifft)
    d = np.real(xx) + np.real(zz) - 2 * c
    k = np.exp(-1. / sigma ** 2 * np.abs(d) / N)

    return k
def detect_init(gausian,win_patch):
    '''
    gausian is the two dimension gausian distributin matrix
    it is the desired output

    And win_patch is the image multiplied by the window function
    '''
    gausian_f=np.fft.fft2(gausian,axes=(0,1))
    k=calculate_K(0.2,win_patch)
    lambda_value=1e-4
    alphaf=np.divide(gausian_f,(np.fft.fft2(k,axes=(0,1))+lambda_value))
    return alphaf
def calculate_response(sub_w,win_patch,alphaf):
    '''
    sub_w a subwindow of the video
    patch is the image multiflied by window function
    '''
    
    #sub_w=sub_w-sub_w.mean() #不知道为什么要减去均值
   
    #sub_wf=np.fft.fft2(sub_w,axes=(0,1))
    K=calculate_K(0.2,win_patch,sub_w)
    KF=np.fft.fft2(K,axes=(0,1))
    response= np.real(np.fft.ifft2(np.multiply(alphaf, KF)))
    return response
def detect(subwindow_list,win_patch,alphaf):
    #response_list=[]
    max_list=[]
    max_dict={}

    for win in subwindow_list:
        win_mean=win[0]-win[0].mean()
        win_z=window(win_mean)
        response=calculate_response(win_z,win_patch,alphaf)
        max_ii=np.argmax(response)
        max_index=np.unravel_index(max_ii,response.shape)
        max_responce_in_sub=response[max_index]
        #print(max_responce_in_sub)
        center=win[1]
        left=center.x-response.shape[1]//2
        top=center.y-response.shape[0]//2
        new_POS=POS(left+max_index[1],top+max_index[0])
        max_list.append((max_responce_in_sub,new_POS))
        max_dict[max_responce_in_sub]=new_POS
    c=sorted(max_dict)[-1]
    finalpos=max_dict[c]
    #cv2.rectangle(frame1,(finalpos.x-response.shape[1]//3,finalpos.y-response.shape[0]//5),\
    #(finalpos.x+response.shape[1]//3,finalpos.y+response.shape[0]//3),(0,255,0),2)
    return finalpos



def draw_rect(event,x,y,flags,param):
    global ix1,iy1,ix2,iy2
    
    if (event==cv2.EVENT_LBUTTONDOWN ):
        ix1,iy1=x,y
    if(flags==cv2.EVENT_FLAG_LBUTTON ):
        ix2,iy2=x,y
 

capture = cv2.VideoCapture(1)
cv2.namedWindow('image') 
cv2.namedWindow('template') 
cv2.setMouseCallback('image',draw_rect)
frames=0
while(1):
    frames+=1
    print(frames)
    _,img=capture.read()
    img=cv2.resize(img,(img.shape[1]//2,img.shape[0]//2))
    if frames<80:
        cv2.rectangle(img,(ix1,iy1),(ix2,iy2),(0,255,0),2)
    if frames==80:
        x1_p,y1_p,x2_p,y2_p=padding(ix1,iy1,ix2,iy2)
        temp_pad=img[y1_p:y2_p,x1_p:x2_p]
        filter_g=Gaussian2(temp_pad)
        temp_patch=window(temp_pad-temp_pad.mean()) 
        alphaf=detect_init(filter_g,temp_patch)
    if frames>80:
        listw=get_subwindow_list(img,filter_g)
        final=detect(listw,temp_patch,alphaf)
        cv2.circle(img,(final.x,final.y),10,(0,0,213),-1)
        cv2.imshow('template',temp_pad)
    cv2.imshow('image',img)
    cv2.waitKey(10)

