import cv2
import numpy as np
import math
import matplotlib.pyplot as plt  


def captureImage():
    cap = cv2.VideoCapture(0)
    while True:
        ret_val, frame = cap.read()
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imshow('frame', image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            out = cv2.imwrite('screen_grab.png', frame)
            break
    cap.release()
    cv2.destroyAllWindows()
    return frame

def imageForceField(image):
    height, width = image.shape[:2]
    forces = np.zeros((height,width))
    offset = 2;

    for y in range(offset+1,height-offset):
        for x in range(offset+1, width-offset):
            force = np.array([0,0],dtype = 'float')
            for yy in range(y-offset, y+offset):
                for xx in range(x-offset, x+offset):
                    if (xx == x) and (yy == y):
                        continue
                    num = np.array([xx-x, yy-y],dtype = 'float')
                    den = np.power( np.linalg.norm(num),3)
                    dot = float (image[yy,xx])
                    force += num * (dot/den)
            forces[y,x] = np.linalg.norm(force)
    forces = np.int8(forces)
    return forces

def imageForceField2(image):
    height, width = image.shape[:2]
    sr = 2 * height
    sc = 2 * width
    forces = np.zeros((height,width),dtype=np.complex_)
    for y in range(0, height):
        for x in range(0, width):
            num = complex(height,width) - complex(y,x)
            den = pow(abs(num),3)
            forces[y,x] = num/den
    snd = np.fft.fft2(forces) * np.fft.fft2(image)
    snd = np.fft.fftshift(snd)
    fst = np.fft.ifftshift(snd)
    fst = np.fft.ifft2(snd)
    forces = (math.sqrt(height * width)) * fst
    forces = np.abs(forces)
    # forces = 20 * np.log(forces)
    return forces

def imageForceField3(image):
    height, width = image.shape[:2]
    sr = 2 * height-1
    sc = 2 * width-1
    r = height-1
    c = width-1
    t = 3*height-3
    u = 3*width-3

    upf = np.zeros((t,u),dtype=np.complex_)
    inp = np.zeros((t,u))
    for rr in range(0, sr):
        for cc in range(0, sc):
            

            num = complex(r,c) - complex(rr,cc) + 0j
            den = pow(abs(complex(r,c) - complex(rr,cc)),3)
            if den==0:
                upf[rr,cc] = 0j
            else: upf[rr,cc]= num/den

    for x in range(0, height):
        for y in range(0, width):
            inp[x,y]=image[x,y]

    oup = np.sqrt(t*u)*np.fft.ifft(np.fft.fft(upf)*np.fft.fft(inp))
    ff = oup[np.ix_([r,2*r],[c,2*c])]

    return ff

def forceFieldFeatureExtraction():
    pass 


if __name__ == "__main__":
    print("Launching webcam ...")
    #image = captureImage()
    image = cv2.imread('Ear_1.png',0) #screen_grab.png
    cv2.imshow('image',image)
    res = imageForceField(image)
    print(res)
    #plt.imshow(res, cmap = 'gray', interpolation = 'bicubic')
    #plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    #plt.show()
    cv2.imshow('image', res)
    cv2.waitKey(0)
    # cv2.destroyAllWindows()