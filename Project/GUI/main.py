from tkinter import *
from tkinter import filedialog
from PIL import Image, ImageTk
import os


class GUI(Frame):

    def __init__(self, master=None):
        Frame.__init__(self, master)
        w, h = 1280, 720
        master.minsize(width=w, height=h)
        master.maxsize(width=w, height=h)
        self.pack()
        self.path = "pic1.jpg"

        self.file = Button(self, text='Browse', command=self.choose)
        self.cuda = Button(self, text='Run Cuda', command=self.run_cuda)
        self.load_adjusted = Button(self, text='Load Adjusted', command=self.load_adjusted)
        self.load_edge_detected = Button(self, text='Load Edge Detected', command=self.load_edge_detected)
        self.alpha = Text(self,
                          height=1,
                          width=5)
        self.alpha_label = Label(self, text="Alpha")
        self.beta_label = Label(self, text="Beta")
        self.alpha.insert(END, "1")

        self.beta = Text(self,
                         height=1,
                         width=5)
        self.beta.insert(END, "0")
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
        self.alpha_label.pack(side=LEFT)
        self.alpha.pack(side=LEFT)
        self.beta_label.pack(side=LEFT)
        self.beta.pack(side=LEFT)

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
        alpha = self.alpha.get("1.0", 'end-1c')
        beta = self.beta.get("1.0", 'end-1c')
        os.system("./main.out " + self.path + " " + alpha + " " + beta)
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

        print("Done")


root = Tk()

app = GUI(master=root)
app.mainloop()
root.destroy()
