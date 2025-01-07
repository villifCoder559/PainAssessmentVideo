import os
import time
import custom.faceExtractor as extractor
import cv2

face_extractor = extractor.FaceExtractor()
path_video_input = os.path.join("partA", "video", "video", "112016_m_25", "112016_m_25-PA4-064.mp4")
path_video_output = os.path.join("partA", "video", "extracted_video", "112016_m_25", "112016_m_25-PA4-064.mp4")
start = time.time()
# face_extractor.generate_face_oval_video(path_video_input=path_video_input,
#                                         path_video_output=path_video_output,
#                                         align=True)
list_annotated_image = face_extractor.get_frames_annotated(path_video_input=path_video_output)
# plot list_annotated_image[0]
print(f'Total time: {time.time()-start:.2f} s')
cv2.imshow('Landmarks', list_annotated_image[0])
cv2.waitKey(0)
cv2.destroyAllWindows()