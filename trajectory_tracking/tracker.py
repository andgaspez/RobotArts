"""
import cv2
import numpy as np
from detector.fgmodel.targetMagenta import targetMagenta

# Initialize the Magenta foreground target detector
magentaDetector = targetMagenta.build_model(25)

# Initialize video capture
cap = cv2.VideoCapture(0)

# Variables for drawing
color = (255, 0, 255)  # Magenta color
grosor = 3  # Thickness of the drawing line
x1, y1 = None, None
imAux = None

# Set up video writer for saving the output
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    if imAux is None:
        imAux = np.zeros(frame.shape, dtype=np.uint8)

    # Process the frame with the Magenta foreground target detector
    magentaDetector.process(frame)

    # Get the foreground mask
    fgmask = magentaDetector.getForeGround()

    # Find contours in the mask
    cnts, _ = cv2.findContours(fgmask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:1]

    for c in cnts:
        area = cv2.contourArea(c)
        if area > 1000:
            x, y2, w, h = cv2.boundingRect(c)
            x2 = x + w // 2

            if x1 is not None:
                if 0 < y2 < 60 or 0 < y1 < 60:
                    imAux = imAux
                else:
                    imAux = cv2.line(imAux, (x1, y1), (x2, y2), color, grosor)
            cv2.circle(frame, (x2, y2), grosor, color, 3)
            x1 = x2
            y1 = y2
        else:
            x1, y1 = None, None

    imAuxGray = cv2.cvtColor(imAux, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(imAuxGray, 10, 255, cv2.THRESH_BINARY)
    thInv = cv2.bitwise_not(th)
    frame = cv2.bitwise_and(frame, frame, mask=thInv)
    frame = cv2.add(frame, imAux)

    # Show the output frames
    cv2.imshow('maskMagenta', fgmask.astype(np.uint8) * 255)
    cv2.imshow('imAux', imAux)
    cv2.imshow('frame', frame)

    # Write the frame to the output video
    out.write(frame)

    k = cv2.waitKey(1)
    if k == 27:  # Press ESC to exit
        break

cap.release()
out.release()
cv2.destroyAllWindows()
"""
import cv2
import numpy as np
from detector.fgmodel.targetMagenta import targetMagenta

# Initialize the Magenta foreground target detector
magentaDetector = targetMagenta.build_model(25)

# Initialize video capture
cap = cv2.VideoCapture(2)

# Variables for drawing
color = (255, 0, 255)  # Magenta color
grosor = 3  # Thickness of the drawing line
x1, y1 = None, None
imAux = None

# Set up video writer for saving the output
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    if imAux is None:
        imAux = np.zeros(frame.shape, dtype=np.uint8)

    # Process the frame with the Magenta foreground target detector
    magentaDetector.process(frame)

    # Get the foreground mask
    fgmask = magentaDetector.getForeGround()

    # Find contours in the mask
    cnts, _ = cv2.findContours(fgmask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:1]

    for c in cnts:
        area = cv2.contourArea(c)
        if area > 100:
            # Calculate moments for each contour
            M = cv2.moments(c)
            # Calculate the centroid coordinates
            if M["m00"] != 0:
                x2 = int(M["m10"] / M["m00"])
                y2 = int(M["m01"] / M["m00"])
            else:
                x2, y2 = 0, 0

            # if x1 is not None:
            #     if 0 < y2 < 60 or 0 < y1 < 60:
            #         imAux = imAux
            #     else:
            #         imAux = cv2.line(imAux, (x1, y1), (x2, y2), color, grosor)
            # cv2.circle(frame, (x2, y2), grosor, color, 3)
            # x1 = x2
            # y1 = y2
            if x1 is not None:
                imAux = cv2.line(imAux, (x1, y1), (x2, y2), color, grosor)
            cv2.circle(frame, (x2, y2), grosor, color, 3)
            x1 = x2
            y1 = y2

        else:
            x1, y1 = None, None

    imAuxGray = cv2.cvtColor(imAux, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(imAuxGray, 10, 255, cv2.THRESH_BINARY)
    thInv = cv2.bitwise_not(th)
    frame = cv2.bitwise_and(frame, frame, mask=thInv)
    frame = cv2.add(frame, imAux)

    # Show the output frames
    cv2.imshow('maskMagenta', fgmask.astype(np.uint8) * 255)
    cv2.imshow('imAux', imAux)
    cv2.imshow('frame', frame)

    # Write the frame to the output video
    out.write(frame)

    k = cv2.waitKey(1)
    if k == 27:  # Press ESC to exit
        break

cap.release()
out.release()
cv2.destroyAllWindows()