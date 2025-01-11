# import kivy dependencies
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
# Import kivy ux components
from kivy.uix.image import Image
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.logger import Logger

# import other dependencies
import cv2
from layers import L1Dist
import tensorflow as tf
import os
import numpy as np

# Build app amd layout
class CamApp(App):
    def build(self):
        # Main Layout Components
        self.web_cam = Image(size_hint = (1,.8))
        self.button = Button(text = 'Verify', on_press = self.verify, size_hint = (1,.1))
        self.verification_label = Label(text = 'Verification Uninitiated', size_hint = (1,.1))

        # Add items to layout
        layout = BoxLayout(orientation='vertical')
        layout.add_widget(self.web_cam)
        layout.add_widget(self.button)
        layout.add_widget(self.verification_label)

        # Load tensorflow/keras model
        self.model = tf.keras.models.load_model('siamesemodelv2.h5', custom_objects={'L1Dist':L1Dist})

        # Setup Video Capture Device
        self.capture = cv2.VideoCapture(0)
        Clock.schedule_interval(self.update, 1.0/33.0)

        return layout
    
    # Run continiously to get webcam feed
    def update(self, *args):
        # Read frame from opencv
        ret, frame = self.capture.read()
        frame = frame[120:120+250,200:200+250,:]
        # Flip horizontal and convert image into texture
        buf = cv2.flip(frame, 0).tostring()
        img_texture = Texture.create(size=(frame.shape[1],frame.shape[0]),colorfmt='bgr')
        img_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        self.web_cam.texture = img_texture

    # Load image from the file and convert it into 100x100 px
    def preprocess(self, file_path):
        # Read in image from file path
        byte_img = tf.io.read_file(file_path)
        # Load in the image
        img = tf.io.decode_jpeg(byte_img)
        # Preprocessing steps - resizing the image to be 100x100x3
        img = tf.image.resize(img,(100,100))
        # scale image to be in between 0 to 1
        img = img/255.0
        return img
    
    # Verification function to verify the person
    def verify(self, *args):
        # Specify thresholds
        detection_threshold =0.5 # 0.6 # 0.5
        verification_threshold =0.8 # 0.7 # 0.8

        # Capture input image from our webcam
        SAVE_PATH = os.path.join('application_data','input_image','input_image.jpg')
        ret, frame = self.capture.read()
        frame = frame[120:220+250,200:300+250,:]
        cv2.imwrite(SAVE_PATH, frame)

        # Build result array
        results = []
        for image in os.listdir(os.path.join('application_data','verification_images')):
            input_img = self.preprocess(os.path.join('application_data','input_image','input_image.jpg'))
            validation_img = self.preprocess(os.path.join('application_data','verification_images',image))


            # Make Predictions
            result = self.model.predict(list(np.expand_dims([input_img, validation_img], axis = 1)))
            results.append(result)

        # Detection Threshold: matric above which a prediction is considered positive 
        detection = np.sum(np.array(results) > detection_threshold)
        # verification_threshold: proportion of a positive prediction / total positive samples
        verification = detection/len(os.listdir(os.path.join('application_data','verification_images')))
        verified = verification > verification_threshold
        # Set Verification text
    #    self.verification_label.text = 'Verified' if verification == True else 'Unverified'

        # Set Verification text and background color
        if verified:
            self.verification_label.text = 'Verified'
            self.verification_label.canvas.before.clear()
            with self.verification_label.canvas.before:
                from kivy.graphics import Color, Rectangle
                Color(0, 1, 0, 1)  # Green
                Rectangle(size=self.verification_label.size, pos=self.verification_label.pos)
        elif verification == 0:
            self.verification_label.text = 'Unverified'
            self.verification_label.canvas.before.clear()
            with self.verification_label.canvas.before:
                from kivy.graphics import Color, Rectangle
                Color(1, 0, 0, 1)  # Red
                Rectangle(size=self.verification_label.size, pos=self.verification_label.pos)
        else:
            self.verification_label.text = 'Verification Uninitiated'
            self.verification_label.canvas.before.clear()
            with self.verification_label.canvas.before:
                from kivy.graphics import Color, Rectangle
                Color(1, 1, 1, 1)  # Default (White)
                Rectangle(size=self.verification_label.size, pos=self.verification_label.pos)

        # Log out details
        Logger.info(results)
        Logger.info(detection)
        Logger.info(verification)
        Logger.info(verified) 

        return results, verified


if __name__ == '__main__':
    CamApp().run()