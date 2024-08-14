import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import subprocess
import os
import sys
import threading
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

from data import get_split_dataset  # Correct import statement

class PixelNerfGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("PixelNeRF GUI")
        self.root.geometry("650x750")  # Set window size to 800x600

        self.current_frame = None
        self.selected_images = []
        self.chosen_images = []
        self.image_thumbnails = []
        self.show_selection_screen()

    def show_selection_screen(self):
        if self.current_frame:
            self.current_frame.destroy()

        self.current_frame = tk.Frame(self.root)
        self.current_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        # Dataset selection
        n_frame = tk.Frame(self.current_frame)
        n_frame.pack(pady=10)
        n_label = tk.Label(n_frame, text="Select Dataset:")
        n_label.pack(side=tk.LEFT)
        self.n_var = tk.StringVar(value="dtu")  # Default value to dtu
        self.n_dtu = tk.Radiobutton(n_frame, text="DTU", variable=self.n_var, value="dvr_dtu")
        self.n_dtu.pack(side=tk.LEFT, padx=5)
        self.n_sn64 = tk.Radiobutton(n_frame, text="ShapeNet Multiple Categories", variable=self.n_var, value="multi_obj")
        self.n_sn64.pack(side=tk.LEFT, padx=5)
        self.n_srn_car = tk.Radiobutton(n_frame, text="ShapeNet Single-Category", variable=self.n_var, value="srn")
        self.n_srn_car.pack(side=tk.LEFT, padx=5)

        # Split selection
        split_frame = tk.Frame(self.current_frame)
        split_frame.pack(pady=10)
        split_label = tk.Label(split_frame, text="Select Split:")
        split_label.pack(side=tk.LEFT)
        self.split_var = tk.StringVar(value="val")
        self.split_train = tk.Radiobutton(split_frame, text="train", variable=self.split_var, value="train")
        self.split_train.pack(side=tk.LEFT, padx=5)
        self.split_val = tk.Radiobutton(split_frame, text="val", variable=self.split_var, value="val")
        self.split_val.pack(side=tk.LEFT, padx=5)
        self.split_test = tk.Radiobutton(split_frame, text="test", variable=self.split_var, value="test")
        self.split_test.pack(side=tk.LEFT, padx=5)

        # Next button
        next_button = tk.Button(self.current_frame, text="Next", command=self.show_details_screen)
        next_button.pack(pady=20)

    def show_details_screen(self):
        if self.current_frame:
            self.current_frame.destroy()

        self.current_frame = tk.Frame(self.root)
        self.current_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        # Data Root selection
        data_root_frame = tk.Frame(self.current_frame)
        data_root_frame.pack(pady=10, fill=tk.X)
        self.data_root_label = tk.Label(data_root_frame, text="Data Root:")
        self.data_root_label.pack(side=tk.LEFT)
        self.data_root_button = tk.Button(data_root_frame, text="Select Folder", command=self.select_data_root)
        self.data_root_button.pack(side=tk.LEFT, padx=5)
        self.data_root_path = tk.StringVar()
        self.data_root_display = tk.Label(data_root_frame, textvariable=self.data_root_path)
        self.data_root_display.pack(side=tk.LEFT, padx=5)

        # Scene selection
        scene_frame = tk.Frame(self.current_frame)
        scene_frame.pack(pady=10, fill=tk.X)
        self.scene_label = tk.Label(scene_frame, text="Select Scene:")
        self.scene_label.pack(side=tk.LEFT)
        self.scene_listbox = tk.Listbox(scene_frame, selectmode=tk.SINGLE, width=50, height=10)
        self.scene_listbox.pack(side=tk.LEFT, padx=5)
        self.scene_listbox.bind('<<ListboxSelect>>', self.update_images)

        # Add vertical scrollbar to scene_listbox
        self.scene_scrollbar = tk.Scrollbar(scene_frame, orient="vertical", command=self.scene_listbox.yview)
        self.scene_scrollbar.pack(side=tk.LEFT, fill=tk.Y)
        self.scene_listbox.config(yscrollcommand=self.scene_scrollbar.set)

        # GPU ID and Scale
        gpu_frame = tk.Frame(self.current_frame)
        gpu_frame.pack(pady=10, fill=tk.X)
        self.gpu_label = tk.Label(gpu_frame, text="GPU ID:")
        self.gpu_label.pack(side=tk.LEFT)
        self.gpu_entry = tk.Entry(gpu_frame)
        self.gpu_entry.pack(side=tk.LEFT, padx=5)
        
        self.scale_label = tk.Label(gpu_frame, text="Scale:")
        self.scale_label.pack(side=tk.LEFT, padx=10)
        self.scale_entry = tk.Entry(gpu_frame)
        self.scale_entry.pack(side=tk.LEFT, padx=5)

        # Encoder Mode Selection
        encoder_frame = tk.Frame(self.current_frame)
        encoder_frame.pack(pady=10, fill=tk.X)
        encoder_label = tk.Label(encoder_frame, text="Select Encoder-Decoder Mode:")
        encoder_label.pack(side=tk.LEFT)
        self.encoder_mode_var = tk.StringVar(value="passive")  # Default value to passive
        self.encoder_mode_active = tk.Radiobutton(encoder_frame, text="Active", variable=self.encoder_mode_var, value="active")
        self.encoder_mode_active.pack(side=tk.LEFT, padx=5)
        self.encoder_mode_passive = tk.Radiobutton(encoder_frame, text="Passive", variable=self.encoder_mode_var, value="passive")
        self.encoder_mode_passive.pack(side=tk.LEFT, padx=5)

        # Images in Scene
        image_frame = tk.Frame(self.current_frame)
        image_frame.pack(pady=10, fill=tk.BOTH, expand=True)
        self.image_label = tk.Label(image_frame, text="Images in Scene:")
        self.image_label.pack(pady=5)
        self.image_canvas = tk.Canvas(image_frame, width=550, height=300)
        self.image_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.scrollbar = tk.Scrollbar(image_frame, orient="vertical", command=self.image_canvas.yview)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.image_canvas.configure(yscrollcommand=self.scrollbar.set)

        # Frame for buttons
        button_frame = tk.Frame(self.current_frame)
        button_frame.pack(side=tk.BOTTOM, pady=10)

        # Button to go back to the previous screen
        back_button = tk.Button(button_frame, text="Back", command=self.show_selection_screen)
        back_button.pack(side=tk.LEFT, padx=5)
        
        # Button to run the script
        self.run_button = tk.Button(button_frame, text="Run Script", command=self.run_script)
        self.run_button.pack(side=tk.LEFT, padx=5)   

    def select_data_root(self):
        folder_path = filedialog.askdirectory(title="Select Data Root Folder")
        if folder_path:
            self.data_root_path.set(folder_path)
            self.full_data_root_path = folder_path
            print(f"Selected data root: {folder_path}")
            self.update_scenes()

    def update_scenes(self):
        dataset_type = self.n_var.get()
        split = self.split_var.get()
        data_root_display = self.data_root_path.get()

        try:
            print(f"Updating scenes for dataset: {dataset_type}, split: {split}, data root: {data_root_display}")
            scenes = get_split_dataset(dataset_type, data_root_display, split, training=False)
            self.scene_listbox.delete(0, tk.END)
            for scene in scenes:
                self.scene_listbox.insert(tk.END, os.path.basename(os.path.normpath(scene["path"])))
                print(f"Inserted scene: {os.path.basename(os.path.normpath(scene['path']))}")
            print(f"Found {len(scenes)} scenes.")
        except IndexError:
            messagebox.showerror("Error", "Data root path is not in the expected format. Please select a valid data root folder.")
            print(f"Data root display value: {data_root_display}")
        except Exception as e:
            messagebox.showerror("Error", f"An unexpected error occurred: {str(e)}")
            print(f"Unexpected error: {str(e)}")

    def update_images(self, event):
        selection = self.scene_listbox.curselection()
        if selection:
            selected_scene = self.scene_listbox.get(selection[0])
            print(f"Selected scene: {selected_scene}")
            scene_path = os.path.join(self.data_root_path.get(), "DTU", selected_scene, "image")
            self.selected_images = self.get_images_in_scene(scene_path)
            self.display_images()

    def get_images_in_scene(self, scene):
        # List all image files in the selected scene directory
        image_files = [os.path.join(scene, file) for file in os.listdir(scene) if file.endswith(('.png', '.jpg', '.jpeg'))]
        return image_files
    
    def display_images(self):
        self.image_thumbnails = []
        self.selection_rects = {}  # Dictionary to keep track of selection rectangles
        self.image_canvas.delete("all")  # Clear the canvas

        x, y = 10, 10
        max_x = 0
        max_y = 0
        row_height = 0

        for i, file_path in enumerate(self.selected_images):
            img = Image.open(file_path)
            img.thumbnail((100, 100))
            img_tk = ImageTk.PhotoImage(img)
            self.image_thumbnails.append(img_tk)

            # Display the image on the canvas
            image_id = self.image_canvas.create_image(x, y, anchor=tk.NW, image=img_tk)

            # Extract the base name (e.g., "000041.png" -> "41")
            base_name = os.path.splitext(os.path.basename(file_path))[0]
            stripped_name = str(int(base_name))  # Convert to int and back to string to strip leading zeros

            # Bind click event to toggle image selection
            self.image_canvas.tag_bind(image_id, '<Button-1>', lambda event, img=stripped_name, image_id=image_id, x=x, y=y: self.toggle_image_selection(img, image_id, x, y))

            row_height = max(row_height, img_tk.height())
            x += img_tk.width() + 10  # Move to the next position horizontally

            # Wrap to the next row after a certain number of images (e.g., 5 images)
            if (i + 1) % 5 == 0:
                x = 10
                y += row_height + 10
                row_height = 0

            max_x = max(max_x, x)
            max_y = max(max_y, y + row_height)

        # Update the scroll region to include the new images
        self.image_canvas.config(scrollregion=(0, 0, max_x, max_y))
    
    def toggle_image_selection(self, img, image_id, x, y):
        if img in self.chosen_images:
            self.chosen_images.remove(img)
            # Remove the rectangle
            if image_id in self.selection_rects:
                self.image_canvas.delete(self.selection_rects[image_id])
                del self.selection_rects[image_id]
        else:
            self.chosen_images.append(img)
            # Draw a red rectangle behind the image
            rect_id = self.image_canvas.create_rectangle(x-2, y-2, x+102, y+102, outline='red', width=2)
            self.selection_rects[image_id] = rect_id
        print(self.chosen_images)

    def get_selected_scene_index(self):
        selection = self.scene_listbox.curselection()
        if selection:
            return selection[0]  # Return the first selected index (assuming single selection)
        return None

    def run_script(self):
        n_param = self.n_var.get()

        # Replace "dvr_dtu" with "dtu"
        if n_param == "dvr_dtu":
            n_param = "dtu"
        split_param = self.split_var.get()
        gpu_id = self.gpu_entry.get()
        data_root_display = self.data_root_path.get()
        data_root_parts = os.path.normpath(data_root_display).split(os.sep)
        data_root = os.path.join(data_root_parts[-2], data_root_parts[-1])
        scale = self.scale_entry.get()
        encoder_mode = self.encoder_mode_var.get()

        # Use only the indices (already stripped and stored in self.chosen_images)
        images = ' '.join(self.chosen_images)

        scene_index = self.get_selected_scene_index()

        # Ensure all required fields are filled out
        if not (n_param and split_param and gpu_id and data_root and scale and images):
            messagebox.showerror("Error", "All fields and image selection are required.")
            return

        # Construct the command string with just the image indices, data root, and encoder mode
        command = f'python eval/gen_video.py -n {n_param} --gpu_id={gpu_id} --split {split_param} -P "{images}" -D "{data_root}" -S {scene_index} --scale {scale} --encoder_mode "{encoder_mode}"'

        try:
            # Run the command using subprocess
            subprocess.run(command, shell=True, check=True)
            messagebox.showinfo("Success", "Script executed successfully.")
        except subprocess.CalledProcessError as e:
            # Display an error message if the command fails
            messagebox.showerror("Error", f"Script execution failed: {e}")

if __name__ == "__main__":
    root = tk.Tk()
    app = PixelNerfGUI(root)
    root.mainloop()
