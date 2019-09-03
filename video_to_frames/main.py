import cv2
vidcap = cv2.VideoCapture('GH010005.MP4')
success,image = vidcap.read()
count = 0
count_valids = 0
while count != 840:
    if success == True:
        num_frame = str(count_valids).zfill(4)
        name_frame = "frame" + num_frame + ".jpg"
        cv2.imwrite(name_frame, image)     # save frame as JPEG file
        count_valids += 1
    success,image = vidcap.read()
    print('Read a new frame: ', success)
    count += 1
    
