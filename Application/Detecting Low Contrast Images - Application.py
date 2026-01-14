import tkinter as tk
from tkinter import filedialog
import cv2
from skimage import exposure
from PIL import Image, ImageTk
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

class LowContrastDetectionApp:
    def __init__(self, root):#Design'y barnama
        self.root = root
        
        self.root.title("Low Contrast Image Detection")
        self.root.configure(bg="#ffffff")  
        self.custom_font = ("Helvetica", 12)
        self.main_frame = tk.Frame(self.root, bg="#ffffff")
        self.main_frame.pack(expand=True, fill=tk.BOTH, padx=20, pady=20)
        self.image_frame = tk.Frame(self.main_frame, bg="#ffffff")
        self.image_frame.grid(row=0, column=0, padx=20, pady=20)
        self.original_image_label = tk.Label(self.image_frame, bg="#f0f0f0", borderwidth=2, relief="groove")
        self.original_image_label.grid(row=0, column=0, padx=20)
        self.adjusted_image_label = tk.Label(self.image_frame, bg="#f0f0f0", borderwidth=2, relief="groove")
        self.adjusted_image_label.grid(row=0, column=1, padx=20)

        #Design'y Histogram
        self.histogram_frame_low = tk.Frame(self.main_frame, bg="#ffffff")
        self.histogram_frame_low.grid(row=0, column=1, padx=20, pady=20)
        self.histogram_frame_high = tk.Frame(self.main_frame, bg="#ffffff")
        self.histogram_frame_high.grid(row=0, column=2, padx=20, pady=20)

        

        #Design w Eshkrdny Button'akan
        self.button_frame = tk.Frame(self.main_frame, bg="#ffffff")
        self.button_frame.grid(row=1, column=0, columnspan=3, pady=10)
        self.load_button = tk.Button(self.button_frame, text="Load Image", command=self.load_image, bg="#007bff", fg="white", relief=tk.FLAT, padx=10, pady=5, font=self.custom_font)
        self.load_button.pack(side=tk.LEFT, padx=10)
        self.detect_button = tk.Button(self.button_frame, text="Detect Low Contrast", command=self.detect_low_contrast, bg="#28a745", fg="white", relief=tk.FLAT, padx=10, pady=5, font=self.custom_font)
        self.detect_button.pack(side=tk.LEFT, padx=10)
        self.reload_button = tk.Button(self.button_frame, text="Reload", command=self.reload_page, bg="#dc3545", fg="white", relief=tk.FLAT, padx=10, pady=5, font=self.custom_font)
        self.reload_button.pack(side=tk.LEFT, padx=10)
        
        #Label'y Anjam
        self.caption_label = tk.Label(self.main_frame, text="", pady=10, bg="#ffffff", font=self.custom_font)
        self.caption_label.grid(row=2, column=0, columnspan=3)

        #Qabaray Wenakan bo nishandan
        self.max_width = 300
        self.max_height = 300

    def load_image(self):#Rakeshany wena la folder
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg; *.jpeg; *.png; *.bmp")])#Snurdarkrdny type'y wena
        if file_path:
            self.original_image = cv2.imread(file_path)
            self.original_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
            self.display_image(self.original_image, self.original_image_label)

    def display_image(self, image, label):#Nishandanaway wena la folder
        #Goriny shape w size'y wena gar zyatrbw
        height, width = image.shape[:2]
        if height > self.max_height or width > self.max_width:
            scale = min(self.max_width/width, self.max_height/height)
            image = cv2.resize(image, (int(width*scale), int(height*scale)))

        image = Image.fromarray(image)  
        image = ImageTk.PhotoImage(image)  
        label.config(image=image)
        label.image = image

    def display_histogram(self, image, frame, contrast_type):#Nishandanaway histogram
        #7isabaty histogram
        hist = cv2.calcHist([cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)], [0], None, [256], [0,256])
        fig, ax = plt.subplots(figsize=(3, 2))

        #Design'y result histogram
        ax.plot(hist, color='green')
        ax.set_title('Histogram')
        ax.set_xlabel('Pixel value')
        ax.set_ylabel('Frequency')
        ax.text(0.9, 1.0605, f"{contrast_type} Contrast)", transform=ax.transAxes, ha='center', fontsize=10)
        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas.draw()
        canvas.get_tk_widget().pack()

    def detect_low_contrast(self):#Agar contrast kambu
        if hasattr(self, 'original_image'):

            hsv_image = cv2.cvtColor(self.original_image, cv2.COLOR_RGB2HSV)
            v_channel = hsv_image[:,:,2]
            #Garanaway 7iabat bo histogram
            hist = cv2.calcHist([v_channel], [0], None, [256], [0,256])
            #7isabaty contrast
            contrast = hist[-1] - hist[0]
            if contrast < 1000: 
                p2, p98 = np.percentile(v_channel, (2, 98))
                v_channel = exposure.rescale_intensity(v_channel, in_range=(p2, p98))

                enhanced_hsv_image = hsv_image.copy()
                enhanced_hsv_image[:,:,2] = v_channel
                enhanced_image_rgb = cv2.cvtColor(enhanced_hsv_image, cv2.COLOR_HSV2RGB)
                #Nishandanaway hardw wena (low and after high)
                self.display_image(self.original_image, self.original_image_label)
                self.display_image(enhanced_image_rgb, self.adjusted_image_label)
                self.caption_label.config(text="Image has low contrast", fg="#dc3545")
                #Nishandanaway histogram bo high w low
                self.display_histogram(self.original_image, self.histogram_frame_low, "(Low")
                self.display_histogram(enhanced_image_rgb, self.histogram_frame_high, " (High")
            else:
                # Agar high contrast bw, Tanha Nishandanaway original image
                self.display_image(self.original_image, self.original_image_label)
                self.adjusted_image_label.config(image=None)
                self.caption_label.config(text="Image has high contrast", fg="#28a745")
                # Tanha Nishandanaway histogram'y original image
                self.display_histogram(self.original_image, self.histogram_frame_low, "Low")
                for widget in self.histogram_frame_high.winfo_children():
                    widget.destroy()

    def reload_page(self):#Nwekrdnaway barnama
        self.root.destroy()  
        root = tk.Tk()
        app = LowContrastDetectionApp(root)
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        x = (screen_width - root.winfo_reqwidth()) // 4
        y = (screen_height - root.winfo_reqheight()) // 4
        root.geometry(f"1200x600+{x}+{y}")
        root.mainloop()

def main():
    root = tk.Tk()
    app = LowContrastDetectionApp(root)
    # Dyary krdny size'y window
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    x = (screen_width - root.winfo_reqwidth()) // 4
    y = (screen_height - root.winfo_reqheight()) // 4
    root.geometry(f"1200x600+{x}+{y}")
    root.mainloop()

if __name__ == "__main__":#Standardy run krdny python GUI
    main()
