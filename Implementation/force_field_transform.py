import cv2
import numpy as np
import math 


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
    for y in range(0,height):
        for x in range(0, width):
            current_pixel = (x,y)
            force = np.array([0,0])
            for yy in range(0, width):
                for xx in range(0, height):
                    if (xx == x) and (yy == y):
                        continue
                    num = np.array([xx-x, yy-y])
                    den = np.linalg.norm(num,3)
                    force += image[y,x] * np.array((num/den), dtype='int64')
            forces[y,x] = np.linalg.norm(force)
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
    snd = np.fft.fft(forces) * np.fft.fft(image)
    fst = np.fft.ifft(snd)
    forces = (math.sqrt(height * width)) * fst
    forces = np.absolute(forces)
    return forces

def forceFieldFeatureExtraction():
    pass 


if __name__ == "__main__":
    print("Launching webcam ...")
    #image = captureImage()
    image = cv2.imread('Ear_1.jpg',0)
    res = imageForceField(image)
    cv2.imshow('image', res)
    cv2.waitKey(0)
    cv2.destroyAllWindows()