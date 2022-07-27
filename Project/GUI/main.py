from tkinter import *
from tkinter import filedialog
from PIL import Image, ImageTk
import os
import math

class GUI(Frame):

    def __init__(self, master=None):
        Frame.__init__(self, master)
        self.winfo_toplevel().title("Edge Detector")

        w, h = 1280, 800
        master.minsize(width=w, height=h)
        master.maxsize(width=w, height=h)

        self.pack()
        self.path = "default.jpg"

        ### Menu
        menu_sub_frame = Frame(self)
        self.file = Button(menu_sub_frame, text='Browse Image', command=self.choose)
        self.cuda = Button(menu_sub_frame, text='Run', command=self.run_cuda)
        self.load_adjusted = Button(menu_sub_frame, text='Show Adjusted Image', command=self.load_adjusted)
        self.load_edge_detected = Button(menu_sub_frame, text='Show Edge Detected Image', command=self.load_edge_detected)
        
        alpha_sub_frame = Frame(menu_sub_frame)
        self.alpha_label = Label(alpha_sub_frame, text="Contrast")
        self.alpha_slider = Scale(alpha_sub_frame, from_=0, to=100, length=150, orient=HORIZONTAL)
        self.alpha_slider.set(50)
        self.alpha_label.pack(side=BOTTOM)
        self.alpha_slider.pack(side=BOTTOM)
        

        beta_sub_frame = Frame(menu_sub_frame)
        self.beta_label = Label(beta_sub_frame, text="Brightness")
        self.beta_slider = Scale(beta_sub_frame, from_=0, to=100, length=150, orient=HORIZONTAL)
        self.beta_slider.set(50)
        self.beta_label.pack(side=BOTTOM)
        self.beta_slider.pack(side=BOTTOM)

        threshold_sub_frame = Frame(menu_sub_frame)
        self.threshold_label = Label(threshold_sub_frame, text="Threshold")
        self.threshold_slider = Scale(threshold_sub_frame, from_=0, to=255, length=150, orient=HORIZONTAL)
        self.threshold_slider.set(0)
        self.threshold_label.pack(side=BOTTOM)
        self.threshold_slider.pack(side=BOTTOM)


        self.file.pack(side=LEFT)
        self.cuda.pack(side=LEFT)
        self.load_adjusted.pack(side=LEFT)
        self.load_edge_detected.pack(side=LEFT)
        alpha_sub_frame.pack(side=LEFT, padx=10)
        beta_sub_frame.pack(side=LEFT, padx=10)
        threshold_sub_frame.pack(side=LEFT, padx=10)
        
        ### Canvas
        self.canvas_width = 1024
        self.canvas_height = 700

        canvas_sub_frame = Frame(self)
        img = Image.open(self.path)
        img = self.fit_image_to_canves(img)
        self.image = ImageTk.PhotoImage(img)
        self.canvas = Canvas(canvas_sub_frame, width=self.canvas_width, height=self.canvas_height)
        self.image_container = self.canvas.create_image(self.canvas_width/2, 
                                                        self.canvas_height/2, 
                                                        image=self.image, 
                                                        anchor=CENTER)
        self.canvas.pack(side=TOP)

        
        ### Root Frame
        
        canvas_sub_frame.pack(side=TOP)
        menu_sub_frame.pack(side=BOTTOM)


    def choose(self):
        try:
            ifile = filedialog.askopenfile(parent=self, mode='rb', title='Choose a file')
            img = Image.open(ifile)
            img = self.fit_image_to_canves(img)
            self.path = ifile.name

            self.image2 = ImageTk.PhotoImage(img)
            self.canvas.itemconfig(self.image_container, image=self.image2)
        except:
            pass
        # self.label.image=self.image2

    def run_cuda(self):
        print("Running CUDA")
        print(self.path)
        alpha = self.get_alpha()
        beta = self.get_beta()
        thresh = self.get_threshhold()
        print(alpha, beta)
        os.system("./main.out " + self.path + " " + str(alpha) + " " + str(beta) + " " + str(thresh))
        print("Done")

    def load_adjusted(self):
        print("Loading Adjusted")
        self.path = "./sobel-out-adjust.jpg"
        im = open(self.path, "rb")
        img = self.fit_image_to_canves(Image.open(im))
        self.image2 = ImageTk.PhotoImage(img)
        self.canvas.itemconfig(self.image_container, image=self.image2)
        print("Done")

    def load_edge_detected(self):
        print("Loading Edge Detected")
        self.path = "./sobel-out.jpg"
        im = open(self.path, "rb")
        img = self.fit_image_to_canves(Image.open(im))
        self.image2 = ImageTk.PhotoImage(img)
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

    def get_threshhold(self):
        raw_thresh = self.threshold_slider.get()
        return raw_thresh

    def fit_image_to_canves(self, img):
        img_width = img.size[0] * 1.0
        img_height = img.size[1] * 1.0
        ratio = img_width / img_height

        if img_height > self.canvas_height:
            img_height = self.canvas_height
            img_width = ratio * img_height
        
        if img_width > self.canvas_width:
            img_width = self.canvas_width
            img_height = img_width / ratio
        
        return img.resize((int(img_width), int(img_height)))
        

root = Tk()

app = GUI(master=root)
app.mainloop()
root.destroy()
