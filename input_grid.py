import tkinter
from tkinter import *

import numpy as np
from PIL import ImageGrab
from keras.models import load_model
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg)
from matplotlib.figure import Figure

canvas_size = 3 * 28
resized_img_size = (28, 28)
nn_model = load_model('neural_network_model.h5')


class DigitClassifier:
    def __init__(self, master, canvas_dim, image_size):
        self.master = master
        descriptor = Label(self.master, text="Press and Drag the mouse to draw")
        descriptor.pack(side=BOTTOM)

        self.image_size = image_size
        self.master.title('Interactive Digit Classifier')

        self.canvas = Canvas(master,
                             width=canvas_dim,
                             height=canvas_dim,
                             background='white',
                             highlightthickness=0
                             )
        self.canvas.pack()

        self.canvas.bind('<B1-Motion>', self.paint)

        classify_button = Button(master, text='Classify Digit', command=self.classify_digit)
        classify_button.pack()
        clear_button = Button(master, text='Clear', command=self.clear_canvas)
        clear_button.pack()

        # Figure to display bar graph of probabilities
        self.fig = Figure(figsize=(5, 4), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.plot = FigureCanvasTkAgg(self.fig, master=master)

    def clear_canvas(self):
        self.clear_plot()
        self.canvas.delete('all')

    def paint(self, event):
        x1, y1 = (event.x - 2), (event.y - 2)
        x2, y2 = (event.x + 2), (event.y + 2)
        self.canvas.create_oval(x1, y1, x2, y2, fill='black')

    def capture_canvas(self):
        x = self.master.winfo_rootx() + self.canvas.winfo_x()
        y = self.master.winfo_rooty() + self.canvas.winfo_y()
        x1 = x + self.canvas.winfo_width()
        y1 = y + self.canvas.winfo_height()
        im = ImageGrab.grab((x, y, x1, y1))
        im.save("trial.jpg")
        return im.resize(self.image_size)

    def clear_plot(self):
        self.ax.clear()
        self.ax.set_xticks(list(range(10)))
        self.plot.draw()


    def plot_bar(self, pred_prob):
        self.ax.set_xticks(list(range(10)))
        self.ax.bar(np.arange(0, 10, 1), pred_prob)
        self.plot.draw()
        self.plot.get_tk_widget().pack(side=tkinter.TOP, fill=tkinter.BOTH, expand=1)

    def classify_digit(self):
        canvas = self.capture_canvas()
        canvas_array = np.reshape((255 - np.array(canvas.convert("L"))) / 255, (1, 28, 28, 1))
        assert canvas_array.shape == (1, 28, 28, 1)
        pred_prob = nn_model.predict_proba(canvas_array)[0]
        self.clear_plot()
        self.plot_bar(pred_prob)


master = Tk()
input_grid = DigitClassifier(master, canvas_size, resized_img_size)
master.mainloop()
