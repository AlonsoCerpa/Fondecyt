import cv2

vidcap = cv2.VideoCapture('GH010023.MP4')
success,image = vidcap.read()
count = 0
count_valids = 0
fps = 60
seconds = 37
step = 30
count_step = 1
while count < fps * seconds:
    if count_step % step == 0:
        while success == False:
            success,image = vidcap.read()
            count += 1
        num_frame = str(count_valids).zfill(4)
        name_frame = "frame" + num_frame + ".jpg"
        print('Saved frame: ', name_frame)
        cv2.imwrite(name_frame, image)     # save frame as JPEG file
        count_valids += 1
        count_step = 1
    else:
        count_step += 1
        
    success,image = vidcap.read()
    print('Read frame ', count, ': ', success)
    count += 1
    
