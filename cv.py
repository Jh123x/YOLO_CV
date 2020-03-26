import cv2 #For computer vision capabilities
import numpy as np #For calculations

#Video capture object (Change input based on your configuration)
capture = cv2.VideoCapture(0)

# Load Yolo (Image recognition) weights and configurations
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")

#Open the names of the associated weights
with open("coco.names", "r") as f:

    #Put the names into a list
    classes = [line.strip() for line in f.readlines()]

#Get the names of the layers
layer_names = net.getLayerNames()

#Get the outer layers
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

#Generate the color
colors = np.random.uniform(0, 255, size=(len(classes), 3))

#Loop to show the frames
while True:

    #Getting the frame in the capture
    ret, frame = capture.read()

    #Resize the frame based on fx ratio and fy ratio
    frame = cv2.resize(frame, None, fx=1.0, fy=1.0)

    #Get the dimensions of the frame
    height, width, channels = frame.shape

    #Preprocesses the frame (Mean subtraction, Scaling, optional channel swapping)
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

    #Set the inut for the neural network
    net.setInput(blob)

    #Compute the output of the layer an store them in a list
    outs = net.forward(output_layers)

    #Define the accumulators to store the result for use later
    class_ids = []
    confidences = []
    boxes = []

    #Iterating through the result of the output layers
    for out in outs:

        #Iterate through all the possible detections detected
        for detection in out:
            scores = detection[5:]

            #Get the maximum value of the score
            class_id = np.argmax(scores)

            #Obtain the confidence level of the object
            confidence = scores[class_id]

            #If below 50% it is unlikely to be the object
            if confidence > 0.5:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                #Append the rectangular coordinates to a list
                boxes.append([x, y, w, h])

                #Append the confidence level to the box
                confidences.append(float(confidence))

                #Append the class id of the object detected
                class_ids.append(class_id)

    #Filter the boxes for those who meet the criteria
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    #Create a font to be used to print the names of the objects detected
    font = cv2.FONT_HERSHEY_PLAIN

    #Iterate through all the bounding boxes
    for i in range(len(boxes)):

        #If the index is found inside the filtered list
        if i in indexes:

            #Unpack the dimensions
            x, y, w, h = boxes[i]

            #Get the label for the identified object
            label = str(classes[class_ids[i]])

            #Get the color to be used for the box
            color = colors[i]

            #Draw the rectangular bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

            #Put the text about the box
            cv2.putText(frame, label, (x, y + 30), font, 3, color, 3)

    #Showing the frame in the window
    cv2.imshow("Image",frame)

    #Check if the Q key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):

        #If so break the loop
        break 


#Close the capture
capture.release()

#Destroy all the windows
cv2.destroyAllWindows()