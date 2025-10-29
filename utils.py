import cv2

def video_writer(frames, path, fps=30):
    '''
    Writes video `frames` to `path`. Expect `frames` to be shape [num_frames, height, width, 3]
    
    TODO: for some reason this saves an .mp4 but I can't preview play it in VSCode on my ubuntu partition? 
    don't have this problem with the built in torchrl VideoWriter even though that one also saves an mp4. 
    can play the video4 by opening thru file explorer though. 
    '''
    height, width = frames.shape[1], frames.shape[2]

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(path, fourcc, fps, (width, height), isColor=True)

    for frame in frames:
        video.write(frame.numpy())

    video.release()
    print(f"Video should now be saved in: {path}")