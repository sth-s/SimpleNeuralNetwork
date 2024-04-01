import tkinter as tk
from tkinter import Canvas
import numpy as np
from PIL import Image,  ImageTk
from NeuralNetwork import NeuralNetwork
class Painter(tk.Tk):
    def __init__(self, nn: NeuralNetwork):
        super().__init__()
        self.nn = nn
        self.w = 28
        self.h = 28
        self.scale = 32
        self.colors = np.zeros((self.w, self.h))

        #Create the window
        self.title("Digit Recognizer")
        self.geometry(f"{self.w * self.scale + 200}x{self.h * self.scale}")
        self.canvas = Canvas(self, width=self.w * self.scale + 200, height=self.h * self.scale)
        self.canvas.pack()

        # Bind functions
        self.bind("<B1-Motion>", self.paint)
        self.bind("<B3-Motion>", self.erase)
        self.bind("<space>", self.clear)

        # Initialize mouse position
        self.mx, self.my = 0, 0  

        self.update()

    # Add wide effect
    def apply_brush_effect(self, x, y, increase=True):
        radius = 2
        strength = 0.3
        for i in range(max(0, x-radius), min(self.w, x+radius+1)):
            for j in range(max(0, y-radius), min(self.h, y+radius+1)):
                dist = np.sqrt((i - x)**2 + (j - y)**2)
                if dist <= radius:
                    effect = strength * (1 - dist/radius)
                    self.colors[i, j] += effect if increase else -effect
                    self.colors[i, j] = np.clip(self.colors[i, j], 0, 1)


    def paint(self, event):
        x, y = event.x // self.scale, event.y // self.scale
        self.apply_brush_effect(x, y, increase=True)
        self.update()

    def erase(self, event):
        x, y = event.x // self.scale, event.y // self.scale
        self.apply_brush_effect(x, y, increase=False)
        self.update()

    def clear(self, event):
        self.colors = np.zeros((self.w, self.h))
        self.update()

    def update(self):
        # Desk update
        img = Image.new("RGB", (self.w, self.h), "black")
        pixels = img.load()
        inputs = np.zeros(784)
        for i in range(self.w):
            for j in range(self.h):
                val = int(self.colors[i, j] * 255)
                pixels[i, j] = (val, val, val)
                inputs[i + j * 28] = self.colors[i, j]

        img = img.resize((self.w * self.scale, self.h * self.scale), Image.NEAREST)
        photo =  ImageTk.PhotoImage(image=img)

        self.canvas.create_image(0, 0, anchor=tk.NW, image=photo)
        self.canvas.image = photo

        # NeutalNetwork analyse
        outputs = self.nn.feed_forward(inputs)
        predicted_digit = np.argmax(outputs)

        self.canvas.delete("output_text")

        for i, output in enumerate(outputs):
            x_position = self.w * self.scale + 10
            y_position = i * 20 + 20
            probability_text = f"{i}: {output:.2f}"
        
            color = "red" if i == predicted_digit else "black"
            self.canvas.create_text(x_position, y_position, text=probability_text, tag="output_text", fill=color)