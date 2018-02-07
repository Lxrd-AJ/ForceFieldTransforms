import cv2
import numpy as np
import math
import matplotlib.pyplot as plt  


def captureImage():
    cap = cv2.VideoCapture(0)
    while True:
        ret_val, frame = cap.read()
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        image = imageForceField4(image)
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
            force = np.array([0,0],dtype = 'float64')
            for yy in range(y-offset, y+offset):
                for xx in range(x-offset, x+offset):
                    if (xx == x) and (yy == y):
                        continue
                    # num = np.array([xx-x, yy-y],dtype = 'float64')
                    # # den = np.power( np.linalg.norm(num),3)
                    # den = np.linalg.norm(num,3)
                    # # dot = float (image[yy,xx])
                    # force += num * (image[yy,xx]/den)

                    num = np.array([xx-x, yy-y])
                    scalar = image[yy,xx] / math.sqrt(np.sum(np.power(num,2))) #pow(math.sqrt(np.sum(np.power(num,2))),3)
                    force = force + (num * scalar )
            forces[y,x] = math.sqrt(np.sum(np.power(force,2))) #np.linalg.norm(force)
    # forces = np.int8(forces)
    return forces #np.absolute(forces)

def imageForceField2(image):
    height, width = image.shape[:2]
    scale = 2
    forces = np.zeros((height*scale,width*scale),dtype=np.complex_)
    for y in range(0, height*scale):
        for x in range(0, width*scale):
            num = complex(height,width) - complex(y,x)
            den = pow(abs(num),3)
            if den==0:
                forces[y,x] = 0j
            else: forces[y,x] = num/den
    image = cv2.resize(image, (width*scale,height*scale))
    forces = np.sqrt(height*width) * np.fft.ifft2( np.fft.fft2(forces) * np.fft.fft2(image)) 
    forces = np.absolute(forces)
    print(forces.shape)
    forces *= 255.0/forces.max()
    forces = np.uint8(forces)
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

    oup = np.sqrt(t*u)*np.fft.ifft2(np.fft.fft2(upf)*np.fft.fft2(inp))
    # ff = oup[np.ix_([r,2*r],[c,2*c])]
    ff = oup 
    return np.absolute(ff)

def imageForceField4(image):
    height, width = image.shape[:2]
    sr = 2 * (height-1)
    sc = 2 * (width-1)
    r = height - 1
    c = width - 1
    upf = np.zeros((sr,sc), dtype=np.complex_)
    for rr in range(0,sr):
        for cc in range(0,sc):
            num = complex(r,c) - complex(rr,cc) + 0j
            den = pow(abs(complex(r,c) - complex(rr,cc)),3)
            if den==0:
                upf[rr,cc] = 0j
            else: upf[rr,cc]= num/den
    inp = image 
    # inp = cv2.resize(image, (sc,sr)) #TODO: Fix Bug here
    vertical = int((sr - height + 1) / 2)
    horizontal = int((sc - width + 1) / 2)
    inp = cv2.copyMakeBorder(image, vertical, vertical, horizontal, horizontal,cv2.BORDER_CONSTANT,value=255)

    oup = np.sqrt(height*width) * np.fft.ifft2( np.fft.fft2(upf) * np.fft.fft2(inp))
    return np.absolute(oup) 
            

def stream():
    cap = cv2.VideoCapture(0)
    frame_count = 1
    while True:
        ret_val, frame = cap.read()
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        image = imageForceField4(image)
        print("Processed frame", frame_count)
        frame_count += 1
        plt.imshow(image, cmap = 'gray', interpolation = 'bicubic')
        plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
        plt.show()

        if cv2.waitKey(1) & 0xFF == ord('q'):            
            break
    cap.release()
    cv2.destroyAllWindows()
    return frame


if __name__ == "__main__":
    print("Launching webcam ...")
    # image = captureImage()
    # stream()

    # image = cv2.imread('screen_grab.png',0)
    image = cv2.imread('Ear_1.png',0) 
    # height, width = image.shape[:2]
    # image = cv2.resize(image, (width,height))
    # cv2.imshow('image', image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    
    # cv2.imshow('image',image)
    res = imageForceField2(image)
    print(res)
    cv2.imshow('image', res)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # plt.imshow(res, cmap = 'gray', interpolation = 'bicubic')
    # plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    # plt.show()

    