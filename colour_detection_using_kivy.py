import cv2
import numpy as np
from kivy.app import App
from kivy.uix.image import Image
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.core.window import Window
from scipy.spatial import KDTree


class VideoApp(App):
    def build(self):
        # Maximize the window for full-screen experience
        Window.maximize()
        Window.clearcolor = (0, 0, 0, 1)
        Window.bind(on_resize=self.on_window_resize)

        # Main layout
        self.layout = BoxLayout(orientation='vertical', padding=20, spacing=20)

        # Video display widget
        self.image = Image(size_hint=(1, 0.75))  # Larger camera display
        self.layout.add_widget(self.image)

        # Color label
        self.color_label = Label(
            text="Detected Color: None",
            size_hint=(1, 0.1),
            font_size=28,  # Increased font size
            bold=True,
            color=(1, 1, 1, 1)  # White text
        )
        self.layout.add_widget(self.color_label)

        # Control buttons
        btn_layout = BoxLayout(size_hint=(1, 0.15), spacing=30, padding=20)

        self.start_button = Button(
            text="Start Video",
            font_size=28,  # Larger font size
            background_color=(0, 1, 0, 1),  # Green
            color=(1, 1, 1, 1),  # White text
            size_hint=(0.5, 1)
        )
        self.start_button.bind(on_press=self.start_video)
        btn_layout.add_widget(self.start_button)

        self.stop_button = Button(
            text="Stop Video",
            font_size=28,  # Larger font size
            background_color=(1, 0, 0, 1),  # Red
            color=(1, 1, 1, 1),  # White text
            size_hint=(0.5, 1)
        )
        self.stop_button.bind(on_press=self.stop_video)
        btn_layout.add_widget(self.stop_button)

        self.layout.add_widget(btn_layout)

        self.capture = None
        self.event = None

        # Load colors from dataset
        self.color_data = self.load_colors()

        return self.layout

    def on_window_resize(self, window, width, height):
        """Resize image display dynamically when the window resizes."""
        self.image.size = (width, height * 0.75)

    def start_video(self, instance):
        """Start capturing video from the webcam."""
        if self.capture is None:
            self.capture = cv2.VideoCapture(0)

        if not self.capture.isOpened():
            print("Error: Could not access the webcam.")
            return

        self.event = Clock.schedule_interval(self.update_video, 1.0 / 30.0)

    def stop_video(self, instance):
        """Stop video capture and release resources."""
        if self.event:
            self.event.cancel()
            self.event = None

        if self.capture:
            self.capture.release()
            self.capture = None

    def update_video(self, dt):
        """Capture video frame, detect color, and display it."""
        if self.capture is None:
            return

        ret, frame = self.capture.read()
        if not ret:
            return

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.flip(frame, 0)

        # Get center pixel color
        h, w, _ = frame.shape
        center_pixel = frame[h // 2, w // 2]

        # Get color name
        detected_color = self.get_closest_color(center_pixel)
        self.color_label.text = f"Detected Color: {detected_color}"

        # Convert to Kivy texture
        texture = Texture.create(size=(w, h), colorfmt='rgb')
        texture.blit_buffer(frame.tobytes(), colorfmt='rgb', bufferfmt='ubyte')
        texture.wrap = 'repeat'

        self.image.texture = texture
        self.image.size = Window.size

    def load_colors(self):
        """Load a professional set of colors."""
        color_dict = {
            "Red": (255, 0, 0),
            "Green": (0, 255, 0),
            "Blue": (0, 0, 255),
            "Yellow": (255, 255, 0),
            "Cyan": (0, 255, 255),
            "Magenta": (255, 0, 255),
            "White": (255, 255, 255),
            "Black": (0, 0, 0),
            "Gray": (128, 128, 128),
            "Orange": (255, 165, 0),
            "Pink": (255, 192, 203),
            "Purple": (128, 0, 128),
            "Brown": (139, 69, 19),
            "Lime": (50, 205, 50),
            "Teal": (0, 128, 128),
            "Maroon": (128, 0, 0),
            "Olive": (128, 128, 0),
            "Navy": (0, 0, 128),
            "Gold": (255, 215, 0),
            "Beige": (245, 245, 220),
            "Crimson": (220, 20, 60),
            "Azure": (0, 127, 255),
            "Burgundy": (128, 0, 32),
            "Charcoal": (54, 69, 79),
            "Champagne": (247, 231, 206),
            "Emerald": (80, 200, 120),
            "Fuchsia": (255, 0, 255),
            "Indigo": (75, 0, 130),
            "Lavender": (230, 230, 250),
            "Mustard": (255, 219, 88),
            "Peach": (255, 229, 180),
            "Salmon": (250, 128, 114),
            "Sapphire": (15, 82, 186),
            "Silver": (192, 192, 192),
            "Tan": (210, 180, 140),
            "Turquoise": (64, 224, 208)
        }

        # Convert RGB to NumPy array
        colors_list = np.array(list(color_dict.values()))
        names_list = list(color_dict.keys())

        # Use KDTree for fast nearest neighbor search
        return {"tree": KDTree(colors_list), "names": names_list, "colors": colors_list}

    def get_closest_color(self, rgb):
        """Find the closest color using KDTree."""
        _, index = self.color_data["tree"].query(rgb)
        return self.color_data["names"][index]


if __name__ == "__main__":
    VideoApp().run()
