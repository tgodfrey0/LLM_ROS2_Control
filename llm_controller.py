import cv2
import apriltag

at_options = apriltag.DetectorOptions(families="tag36h11")
tag_width = 10

def entry():
  cv2.namedWindow("Stream")
  vc = cv2.VideoCapture(0)
  
  at_detector = apriltag.Detector(at_options)

  if vc.isOpened():
      rval, frame = vc.read()
  else:
      rval = False

  while rval:
      grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
      
      tags = at_detector.detect(grayscale)
      n_tags = len(tags)
      
      print(f"{n_tags} tags found")
      
      for tag in tags:
        (ptA, ptB, ptC, ptD) = tag.corners
        ptB = (int(ptB[0]), int(ptB[1]))
        ptC = (int(ptC[0]), int(ptC[1]))
        ptD = (int(ptD[0]), int(ptD[1]))
        ptA = (int(ptA[0]), int(ptA[1]))

        cv2.line(frame, ptA, ptB, (0, 255, 0), 2)
        cv2.line(frame, ptB, ptC, (0, 255, 0), 2)
        cv2.line(frame, ptC, ptD, (0, 255, 0), 2)
        cv2.line(frame, ptD, ptA, (0, 255, 0), 2)
        
        (cX, cY) = (int(tag.center[0]), int(tag.center[1]))
        cv2.circle(frame, (cX, cY), 5, (0, 0, 255), -1)
        
        tagFamily = tag.tag_family.decode("utf-8")
        cv2.putText(frame, tagFamily, (ptA[0], ptA[1] - 15),
          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
          
      cv2.imshow("Stream", frame)
      rval, frame = vc.read()
      key = cv2.waitKey(20)
      if key == 27: # exit on ESC
          break
  
  cv2.destroyWindow("Stream")
  vc.release()

if __name__=="__main__":
  entry()