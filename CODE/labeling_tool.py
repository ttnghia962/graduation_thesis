import cv2
import tkinter as tk
from tkinter import filedialog
from pathlib import Path
from PIL import Image, ImageTk

class SimpleFlowchartLabeler:
    def __init__(self):
        self.current_file = None
        self.image = None
        self.drawing = False
        self.shapes = []  # List to store: [(class_id, [x1,y1,x2,y2]), ...]
        self.current_shape = None
        self.start_point = None
        self.current_class = 0  # Default to shape class
        self.labeled_files = set()
        self.file_list = []
        self.current_index = -1
        
        # Set up directories
        self.setup_directories()
        
        # Create window and setup UI
        self.setup_ui()
        
        # Scan files
        self.scan_files()

    def scan_files(self):
        """Scan source directory for images"""
        self.source_dir = Path("source_images")
        
        # Get all image files
        image_extensions = ('.png', '.jpg', '.jpeg', '.bmp')
        self.file_list = [
            f for f in self.source_dir.glob('*')
            if f.suffix.lower() in image_extensions
        ]
        
        # Check which files already have labels
        for img_file in self.file_list:
            label_file = self.labels_dir / f"{img_file.stem}.txt"
            if label_file.exists():
                self.labeled_files.add(img_file.stem)
                
        # Update status
        self.update_status()
        self.update_file_list()

    def update_status(self):
        """Update status display"""
        total = len(self.file_list)
        labeled = len(self.labeled_files)
        status_text = f"Progress: {labeled}/{total} files labeled"
        if hasattr(self, 'status_label'):  # Check if status_label exists
            self.status_label.config(text=status_text)

    def set_class(self, class_id):
        """Set current class for drawing"""
        self.current_class = class_id
        class_name = "Shape" if class_id == 0 else "Direction"
        self.class_label.config(text=f"Current: {class_name}")
        
    def setup_directories(self):
        """Create necessary directories for YOLO format"""
        self.output_dir = Path("dataset")
        self.images_dir = self.output_dir / "images"
        self.labels_dir = self.output_dir / "labels"
        
        # Tạo thư mục nếu chưa tồn tại
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.labels_dir.mkdir(parents=True, exist_ok=True)
            
    def setup_ui(self):
        """Setup user interface"""
        self.window = tk.Tk()
        self.window.title("Flowchart Box Labeler")
        
        # Control Panel
        control_panel = tk.Frame(self.window)
        control_panel.pack(side=tk.LEFT, fill=tk.Y)
        
        # Add file listbox with scrollbar
        listbox_frame = tk.Frame(control_panel)
        listbox_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        scrollbar = tk.Scrollbar(listbox_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.file_listbox = tk.Listbox(listbox_frame, yscrollcommand=scrollbar.set, height=10)
        self.file_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.file_listbox.yview)
        
        # Bind listbox selection
        self.file_listbox.bind('<<ListboxSelect>>', self.on_select_file)
        
        # Class selection buttons with colors
        class_frame = tk.Frame(control_panel)
        class_frame.pack(fill=tk.X, padx=5, pady=5)
        
        tk.Button(class_frame, 
                 text="Shape (1)", 
                 command=lambda: self.set_class(0),
                 bg='lightblue',
                 width=10).pack(side=tk.LEFT, padx=2)
        
        tk.Button(class_frame,
                 text="Direction (2)", 
                 command=lambda: self.set_class(1),
                 bg='pink',
                 width=10).pack(side=tk.LEFT, padx=2)
        
        # Current selection indicator
        self.class_label = tk.Label(control_panel, 
                                  text="Current: Shape",
                                  font=('Arial', 10, 'bold'))
        self.class_label.pack(fill=tk.X, padx=5, pady=5)
        
        # Status label
        self.status_label = tk.Label(control_panel, text="No files loaded")
        self.status_label.pack(fill=tk.X, padx=5, pady=5)
        
        # Action buttons
        tk.Button(control_panel, text="Load Image", 
                 command=self.load_image).pack(fill=tk.X, padx=5, pady=2)
        tk.Button(control_panel, text="Save Labels", 
                 command=self.save_labels).pack(fill=tk.X, padx=5, pady=2)
        tk.Button(control_panel, text="Clear All", 
                 command=self.clear_shapes).pack(fill=tk.X, padx=5, pady=2)
        tk.Button(control_panel, text="Undo Last", 
                 command=self.undo_last).pack(fill=tk.X, padx=5, pady=2)
        
        # Navigation buttons
        nav_frame = tk.Frame(control_panel)
        nav_frame.pack(fill=tk.X, padx=5, pady=5)
        
        tk.Button(nav_frame, text="← Prev",
                 command=self.prev_file).pack(side=tk.LEFT, padx=2)
        tk.Button(nav_frame, text="Next →",
                 command=self.next_file).pack(side=tk.LEFT, padx=2)
        
        # Instructions
        instructions = """
        Instructions:
        1. Select file from list or Load Image
        2. Select type (Shape/Direction)
        3. Draw boxes around areas
        4. Save Labels
        
        Shortcuts:
        1 - Select Shape
        2 - Select Direction
        Z - Undo last
        C - Clear all
        S - Save
        """
        tk.Label(control_panel, 
                text=instructions,
                justify=tk.LEFT,
                bg='white',
                padx=10,
                pady=10).pack(padx=5, pady=10)
        
        # Canvas for image
        self.canvas = tk.Canvas(self.window, cursor="cross")
        self.canvas.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Bind events
        self.canvas.bind("<ButtonPress-1>", self.start_shape)
        self.canvas.bind("<B1-Motion>", self.draw_shape)
        self.canvas.bind("<ButtonRelease-1>", self.end_shape)
        
        # Keyboard shortcuts
        self.window.bind("1", lambda e: self.set_class(0))
        self.window.bind("2", lambda e: self.set_class(1))
        self.window.bind("z", lambda e: self.undo_last())
        self.window.bind("c", lambda e: self.clear_shapes())
        self.window.bind("s", lambda e: self.save_labels())

    def update_file_list(self):
        """Update the file listbox"""
        self.file_listbox.delete(0, tk.END)
        for file_path in self.file_list:
            # Mark labeled files with ✓
            prefix = "✓ " if file_path.stem in self.labeled_files else "□ "
            self.file_listbox.insert(tk.END, prefix + file_path.name)

    def on_select_file(self, event):
        """Handle file selection from listbox"""
        selection = self.file_listbox.curselection()
        if selection:
            index = selection[0]
            self.load_file_by_index(index)

    def prev_file(self):
        """Load previous file"""
        if self.current_index > 0:
            self.load_file_by_index(self.current_index - 1)

    def next_file(self):
        """Load next file"""
        if self.current_index < len(self.file_list) - 1:
            self.load_file_by_index(self.current_index + 1)

    def load_file_by_index(self, index):
        """Load file by index with duplicate check"""
        if 0 <= index < len(self.file_list):
            file_path = self.file_list[index]
            
            # Check if file is already labeled
            if file_path.stem in self.labeled_files:
                response = tk.messagebox.askyesno(
                    "File Already Labeled",
                    f"{file_path.name} is already labeled. Do you want to edit existing labels?"
                )
                if not response:
                    return
                    
            self.current_index = index
            self.current_file = str(file_path)
            self.image = cv2.imread(self.current_file)
            self.display_image()
            self.shapes = []
            self.load_existing_labels()
            self.update_status()

    def load_image(self):
        """Load an image file with duplicate checking"""
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp")],
            initialdir=str(Path(self.current_file).parent) if self.current_file else None
        )
        
        if file_path:
            file_name = Path(file_path).stem
            
            # Kiểm tra xem file đã được label chưa
            label_file = self.labels_dir / f"{file_name}.txt"
            if label_file.exists():
                response = tk.messagebox.askyesno(
                    "File Already Labeled",
                    f"This file ({file_name}) already has labels. Do you want to edit existing labels?",
                    icon='warning'
                )
                if not response:
                    return
            
            # Load image và check kích thước
            img = cv2.imread(file_path)
            if img is None:
                tk.messagebox.showerror(
                    "Error",
                    f"Cannot read image file: {file_path}"
                )
                return
                
            # Update current file và image
            self.current_file = file_path
            self.image = img
            self.shapes = []  # Clear existing shapes
            
            # Display image và load labels nếu có
            self.display_image()
            if label_file.exists():
                self.load_existing_labels()
                
            # Update window title với tên file
            self.window.title(f"Flowchart Labeler - {Path(file_path).name}")
            
            # Print status
            print(f"Loaded image: {file_path}")
            if self.shapes:
                print(f"Found {len(self.shapes)} existing labels")

    def display_image(self):
        """Display the image on canvas"""
        if self.image is not None:
            image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
            height, width = image.shape[:2]
            
            # Resize if too large
            max_size = 800
            if width > max_size or height > max_size:
                scale = max_size / max(width, height)
                width = int(width * scale)
                height = int(height * scale)
                image = cv2.resize(image, (width, height))
            
            self.photo = ImageTk.PhotoImage(Image.fromarray(image))
            self.canvas.config(width=width, height=height)
            self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
            self.redraw_shapes()

    def start_shape(self, event):
        """Start drawing a shape"""
        self.drawing = True
        self.start_point = (event.x, event.y)
        color = 'blue' if self.current_class == 0 else 'red'
        self.current_shape = self.canvas.create_rectangle(
            event.x, event.y, event.x, event.y, outline=color, width=2)

    def draw_shape(self, event):
        """Update shape while drawing"""
        if self.drawing:
            self.canvas.coords(self.current_shape,
                             self.start_point[0], self.start_point[1],
                             event.x, event.y)

    def end_shape(self, event):
        """Finish drawing a shape"""
        if self.drawing:
            self.drawing = False
            x1, y1 = self.start_point
            x2, y2 = event.x, event.y
            
            # Ensure coordinates are properly ordered
            min_x = min(x1, x2)
            max_x = max(x1, x2)
            min_y = min(y1, y2)
            max_y = max(y1, y2)
            
            # Add shape with class_id
            self.shapes.append((
                self.current_class,
                [min_x, min_y, max_x, max_y]
            ))
            
            self.redraw_shapes()

    def redraw_shapes(self):
        """Redraw all shapes"""
        if self.image is None:
            return
            
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
        
        colors = {0: 'blue', 1: 'red'}  # shape: blue, direction: red
        labels = {0: 'Shape', 1: 'Direction'}
        
        for class_id, box in self.shapes:
            x1, y1, x2, y2 = box
            color = colors.get(class_id, 'black')
            label = labels.get(class_id, 'Unknown')
            
            self.canvas.create_rectangle(
                x1, y1, x2, y2, 
                outline=color,
                width=2
            )
            self.canvas.create_text(
                x1, y1-5,
                text=label,
                fill=color,
                anchor=tk.SW
            )

    def convert_to_yolo(self, box, image_size):
        """Convert box coordinates to YOLO format"""
        try:
            # Đảm bảo box có đủ 4 giá trị
            if len(box) != 4:
                print(f"Invalid box format: {box}")
                return None
                
            x1, y1, x2, y2 = box
            width = image_size[1]
            height = image_size[0]
            
            # Convert to YOLO format (center coordinates and dimensions)
            x_center = (x1 + x2) / (2 * width)
            y_center = (y1 + y2) / (2 * height)
            box_width = abs(x2 - x1) / width
            box_height = abs(y2 - y1) / height
            
            # Ensure values are within [0, 1]
            x_center = max(0, min(1, x_center))
            y_center = max(0, min(1, y_center))
            box_width = max(0, min(1, box_width))
            box_height = max(0, min(1, box_height))
            
            return [x_center, y_center, box_width, box_height]
        except Exception as e:
            print(f"Error converting box {box}: {str(e)}")
            return None

    def save_labels(self):
        """Save labels with proper error handling"""
        if not self.current_file or not self.shapes:
            tk.messagebox.showwarning("Warning", "No image or shapes to save!")
            return
            
        filename = Path(self.current_file).stem
        label_path = self.labels_dir / f"{filename}.txt"
        
        try:
            # Convert boxes to YOLO format and save
            with open(label_path, 'w') as f:
                for class_id, box in self.shapes:
                    yolo_box = self.convert_to_yolo(box, self.image.shape[:2])
                    if yolo_box:
                        f.write(f"{class_id} {' '.join(map(str, yolo_box))}\n")
            
            # Update tracking
            self.labeled_files.add(filename)
            self.update_file_list()
            self.update_status()
            
            tk.messagebox.showinfo("Success", f"Labels saved for {filename}")
            
            # Automatically move to next file
            if self.current_index < len(self.file_list) - 1:
                self.next_file()
                
        except Exception as e:
            tk.messagebox.showerror("Error", f"Failed to save labels: {str(e)}")

    def load_existing_labels(self):
        """Load existing labels if available"""
        if not self.current_file:
            return
            
        filename = Path(self.current_file).stem
        label_path = self.labels_dir / f"{filename}.txt"
        
        if label_path.exists():
            self.shapes = []
            try:
                with open(label_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) == 5:  # Ensure we have all 5 values
                            class_id = int(parts[0])
                            x_center, y_center, width, height = map(float, parts[1:])
                            
                            # Convert YOLO format back to pixel coordinates
                            img_width = self.image.shape[1]
                            img_height = self.image.shape[0]
                            
                            x1 = int((x_center - width/2) * img_width)
                            y1 = int((y_center - height/2) * img_height)
                            x2 = int((x_center + width/2) * img_width)
                            y2 = int((y_center + height/2) * img_height)
                            
                            self.shapes.append((class_id, [x1, y1, x2, y2]))
            except Exception as e:
                messagebox.showerror("Error", f"Error loading labels: {str(e)}")
            
            self.redraw_shapes()

    def undo_last(self):
        """Remove the last drawn box"""
        if self.shapes:
            self.shapes.pop()
            self.redraw_shapes()
            
    def clear_shapes(self):
        """Clear all boxes"""
        self.shapes = []
        self.redraw_shapes()
        
    def run(self):
        """Start the labeling tool"""
        self.window.mainloop()

def main():
    labeler = SimpleFlowchartLabeler()
    labeler.run()

if __name__ == "__main__":
    main()