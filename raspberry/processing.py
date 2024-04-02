from gpiozero import MotionSensor
from picamera2 import PiCamera2
from datetime import datetime
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.models import load_model

pir = MotionSensor(4)
camera = PiCamera2()
model = load_model('human-or-not-vgg16.keras')

while True:
    pir.wait_for_motion()
    print("Motion detected!")
    filename = "home/pi/{0:%Y}-{0:%m}-{0:%d}.png".format(datetime.now())
    camera.start_preview()
    camera.capture(filename)

    img = load_img(filename, target_size=(224, 224))
    img_array = img_to_array(img) / 255.0
    prediction = model.predict(img_array)
    is_positive_class = prediction[0][0] > 0.9
    print("Is human", is_positive_class)

    pir.wait_for_no_motion()
    camera.stop_preview()
