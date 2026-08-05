import cv2
import pigpio
import os

pi = pigpio.pi()
if not pi.connected:
    print("Error: pigpio not running. Run 'cd pigpiod-79' then 'sudo pigpiod' in terminal first.")
    exit(1)

SERVO_PAN_PIN = 14  
SERVO_TILT_PIN = 15  

pi.set_mode(SERVO_PAN_PIN, pigpio.OUTPUT)
pi.set_mode(SERVO_TILT_PIN, pigpio.OUTPUT)

current_pan = 1500 
min_pan, max_pan = 1000, 2000

current_tilt = 1500 
min_tilt, max_tilt = 1000, 2000

STEP_TILT = 50
STEP_PAN = 50 

def move_camera_left():
    global current_pan
    current_pan = min(max_pan, current_pan - STEP_PAN)
    pi.set_servo_pulsewidth(SERVO_PAN_PIN, current_pan)
    print("Moving Left. Current pos: " + str(current_pan))

def move_camera_right():
    global current_pan
    current_pan = max(min_pan, current_pan + STEP_PAN)
    pi.set_servo_pulsewidth(SERVO_PAN_PIN, current_pan)
    print("Moving Right. Current pos: " + str(current_pan))

def move_camera_up():
    global current_tilt
    current_tilt = max(min_tilt, current_tilt - STEP_TILT)
    pi.set_servo_pulsewidth(SERVO_TILT_PIN, current_tilt)
    print("Moving Up. Current pos: " + str(current_tilt))

def move_camera_down():
    global current_tilt
    current_tilt = min(max_tilt, current_tilt + STEP_TILT)
    pi.set_servo_pulsewidth(SERVO_TILT_PIN, current_tilt)
    print("Moving Down. Current pos: " + str(current_tilt))

pi.set_servo_pulsewidth(SERVO_PAN_PIN, current_pan)
pi.set_servo_pulsewidth(SERVO_TILT_PIN, current_tilt)

cascade_path = '/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml'
if not os.path.exists(cascade_path):
    cascade_path = 'haarcascade_frontalface_default.xml'
  
face_detector = cv2.CascadeClassifier(cascade_path)

camera = cv2.VideoCapture(0, cv2.CAP_V4L2)
camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

frame_count = 0
SKIP_FRAMES = 2

try:
    while True:
        ret, frame = camera.read()
        if not ret:
            continue
            
        frame_count += 1
        if frame_count % SKIP_FRAMES != 0:
            continue

        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        frame_h, frame_w = frame.shape[:2]
        center_x = frame_w // 2
        center_y = frame_h // 2
        deadzone_x = int(frame_w * 0.12)
        deadzone_y = int(frame_h * 0.12)
        
        faces = face_detector.detectMultiScale(
            gray, 
            scaleFactor=1.3, 
            minNeighbors=3, 
            minSize=(30, 30)
        )
        
        if len(faces) > 0:
            x, y, w, h = faces[0]
            
            face_x = x + (w >> 1)
            face_y = y + (h >> 1)

            print(f"Detected Face Center -> X: {face_x}, Y: {face_y}") 
                                                                                                 
            if face_x < (center_x - deadzone_x):
                move_camera_left()
            elif face_x > (center_x + deadzone_x):
                move_camera_right()
                
            if face_y < (center_y - deadzone_y):
                move_camera_up()
            elif face_y > (center_y + deadzone_y):
                move_camera_down()
        else:
            print("No face detected")

finally:
    pi.set_servo_pulsewidth(SERVO_PAN_PIN, 0)
    pi.set_servo_pulsewidth(SERVO_TILT_PIN, 0)
    pi.stop()
    camera.release()