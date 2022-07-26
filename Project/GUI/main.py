from tkinter import *
from tkinter import filedialog
from PIL import Image, ImageTk
import os
import math

class GUI(Frame):

    def __init__(self, master=None):
        Frame.__init__(self, master)
        w, h = 1280, 720
        master.minsize(width=w, height=h)
        master.maxsize(width=w, height=h)

        self.pack()
        self.path = "pic1.jpg"

        self.file = Button(self, text='Browse Image', command=self.choose)
        self.cuda = Button(self, text='Run', command=self.run_cuda)
        self.load_adjusted = Button(self, text='Show Adjusted Image', command=self.load_adjusted)
        self.load_edge_detected = Button(self, text='Show Edge Detected Image', command=self.load_edge_detected)
        
        
        alpha_sub = Frame(self)
        self.alpha_label = Label(alpha_sub, text="Contrast")
        self.alpha_slider = Scale(alpha_sub, from_=0, to=100, length=150, orient=HORIZONTAL)
        self.alpha_slider.set(50)
        
        beta_sub = Frame(self)
        self.beta_label = Label(beta_sub, text="Brightness")
        self.beta_slider = Scale(beta_sub, from_=0, to=100, length=150, orient=HORIZONTAL)
        self.beta_slider.set(50)

        # Replace with your image
        img = Image.open("pic1.jpg").resize((1024, 768))
        self.image = ImageTk.PhotoImage(img)
        self.canvas = Canvas(root, width=1024, height=768)
        self.image_container = self.canvas.create_image(0, 0, image=self.image, anchor=NW)
        self.canvas.pack(side=BOTTOM)

        self.file.pack(side=LEFT)
        self.cuda.pack(side=LEFT)
        self.load_adjusted.pack(side=LEFT)
        self.load_edge_detected.pack(side=LEFT)
        alpha_sub.pack(side=LEFT, padx=10)
        beta_sub.pack(side=LEFT, padx=10)

        
        self.alpha_label.pack(side=BOTTOM)
        self.alpha_slider.pack(side=BOTTOM)
        
        self.beta_label.pack(side=BOTTOM)
        self.beta_slider.pack(side=BOTTOM)
        

    def choose(self):
        try:
            ifile = filedialog.askopenfile(parent=self, mode='rb', title='Choose a file')
            path = Image.open(ifile).resize((1024, 768))
            self.path = ifile.name

            self.image2 = ImageTk.PhotoImage(path)
            self.canvas.itemconfig(self.image_container, image=self.image2)
        except:
            pass
        # self.label.image=self.image2

    def run_cuda(self):
        print("Running CUDA")
        print(self.path)
        alpha = self.get_alpha()
        beta = self.get_beta()
        print(alpha, beta)
        os.system("./main.out " + self.path + " " + str(alpha) + " " + str(beta))
        print("Done")

    def load_adjusted(self):
        print("Loading Adjusted")
        self.path = "./sobel-out-adjust.jpg"
        im = open(self.path, "rb")
        path = Image.open(im).resize((1024, 768))
        self.image2 = ImageTk.PhotoImage(path)
        self.canvas.itemconfig(self.image_container, image=self.image2)
        print("Done")

    def load_edge_detected(self):
        print("Loading Edge Detected")
        self.path = "./sobel-out.jpg"
        im = open(self.path, "rb")
        path = Image.open(im).resize((1024, 768))
        self.image2 = ImageTk.PhotoImage(path)
        self.canvas.itemconfig(self.image_container, image=self.image2)
        print("Done")

    def get_alpha(self):
        raw_alpha = self.alpha_slider.get()

        if raw_alpha <= 50:
            alpha = 1.0 * (raw_alpha - 0) / (50 - 0) # [0, 1]
            alpha = math.sqrt(alpha) 
        else:
            alpha = 1.0 * (raw_alpha - 50) / (100 - 50) # [0, 1]
            alpha = 1.1 ** (alpha*30) # exponential
            alpha += 1

        return alpha

    def get_beta(self):
        raw_beta = self.beta_slider.get()

        beta = 1.0 * (raw_beta - 0) / (100 - 0) # [0, 1]
        beta *= 300 # [0, 300]
        beta -= 150 # [-150, 150]

        return beta


root = Tk()

app = GUI(master=root)
app.mainloop()
root.destroy()
