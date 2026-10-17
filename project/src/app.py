import os
import queue
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from PIL import Image, ImageTk

# Reuse the inference code, in UI and infer.py
from infer import identify_image, MODELS_FOLDER

PREVIEW_SIZE = 360   # the image preview box is 360 x 360 pixels


def find_available_species():
    """Looks in the models folder and returns the species that have a trained model."""
    if not os.path.isdir(MODELS_FOLDER):
        return []
    species_names = []
    for file_name in os.listdir(MODELS_FOLDER):
        if file_name.startswith("model_b_") and file_name.endswith(".pt"):
            species_names.append(file_name[len("model_b_"):-len(".pt")])
    return sorted(species_names)


class WildlifeApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Model B - Individual Animal Identifier")
        self.image_path = None
        self.preview_photo = None             # we must keep a reference or Tkinter drops the picture
        self.result_queue = queue.Queue()     # the worker thread passes its answer back through this

        self.build_layout()
        self.root.after(100, self.check_for_result)   # start polling for results

    # layout
    def build_layout(self):
        main_frame = ttk.Frame(self.root, padding=12)
        main_frame.grid(row=0, column=0)

        # Left side: image preview
        preview_frame = ttk.Frame(main_frame, width=PREVIEW_SIZE, height=PREVIEW_SIZE, relief="sunken")
        preview_frame.grid(row=0, column=0, rowspan=2, padx=(0, 12), sticky="n")
        preview_frame.pack_propagate(False)   # keep the box at a fixed size
        self.preview_label = ttk.Label(preview_frame, text="No image selected", anchor="center")
        self.preview_label.pack(expand=True)

        # Right side: controls
        controls = ttk.Frame(main_frame)
        controls.grid(row=0, column=1, sticky="nw")

        ttk.Button(controls, text="1. Choose image...", command=self.choose_image).grid(
            row=0, column=0, sticky="w")
        self.file_name_label = ttk.Label(controls, text="", width=40)
        self.file_name_label.grid(row=1, column=0, sticky="w", pady=(2, 10))

        ttk.Label(controls, text="2. Species:").grid(row=2, column=0, sticky="w")
        self.species_box = ttk.Combobox(controls, values=find_available_species(), state="readonly", width=20)
        self.species_box.grid(row=3, column=0, sticky="w", pady=(2, 10))
        available = find_available_species()
        if available:
            self.species_box.current(0)

        ttk.Label(controls, text="Unknown threshold (optional, e.g. 0.6):").grid(row=4, column=0, sticky="w")
        self.threshold_entry = ttk.Entry(controls, width=10)
        self.threshold_entry.grid(row=5, column=0, sticky="w", pady=(2, 10))

        self.run_button = ttk.Button(controls, text="3. Run", command=self.run_identification)
        self.run_button.grid(row=6, column=0, sticky="w")

        self.status_label = ttk.Label(controls, text="Pick an image and a species, then press Run.")
        self.status_label.grid(row=7, column=0, sticky="w", pady=(10, 0))

        # Bottom: results
        results_frame = ttk.LabelFrame(main_frame, text="Result", padding=10)
        results_frame.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(12, 0))

        self.species_result = ttk.Label(results_frame, text="Species: -", font=("Segoe UI", 11))
        self.species_result.grid(row=0, column=0, sticky="w")
        self.individual_result = ttk.Label(results_frame, text="Individual: -", font=("Segoe UI", 12, "bold"))
        self.individual_result.grid(row=1, column=0, sticky="w")
        self.similarity_result = ttk.Label(results_frame, text="Similarity: -")
        self.similarity_result.grid(row=2, column=0, sticky="w")

        self.top_matches_box = tk.Text(results_frame, height=5, width=70, state="disabled")
        self.top_matches_box.grid(row=3, column=0, pady=(8, 0), sticky="w")

    # actions
    def choose_image(self):
        chosen = filedialog.askopenfilename(
            title="Choose an animal image",
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.webp"), ("All files", "*.*")],
        )
        if not chosen:
            return   # the user cancelled the dialog

        try:
            image = Image.open(chosen).convert("RGB")
        except Exception as error:
            messagebox.showerror("Cannot open image", str(error))
            return

        self.image_path = chosen
        self.file_name_label.config(text=os.path.basename(chosen))

        # Shrink a copy of the image to fit the preview box (the original file is untouched)
        image.thumbnail((PREVIEW_SIZE, PREVIEW_SIZE))
        self.preview_photo = ImageTk.PhotoImage(image)
        self.preview_label.config(image=self.preview_photo, text="")

    def run_identification(self):
        # Check the inputs before starting
        if self.image_path is None:
            messagebox.showwarning("Missing image", "Please choose an image first.")
            return
        species_name = self.species_box.get()
        if not species_name:
            messagebox.showwarning("Missing species", "No species selected (is the models folder empty?).")
            return

        threshold_text = self.threshold_entry.get().strip()
        threshold = None
        if threshold_text:
            try:
                threshold = float(threshold_text)
            except ValueError:
                messagebox.showwarning("Bad threshold", "The threshold must be a number such as 0.6.")
                return

        self.run_button.config(state="disabled")
        self.status_label.config(text=f"Running the {species_name} model... (the first run loads the model)")

        # Run the model in a background thread so the window does not freeze
        worker = threading.Thread(
            target=self.worker_job, args=(species_name, self.image_path, threshold), daemon=True
        )
        worker.start()

    def worker_job(self, species_name, image_path, threshold):
        """Runs in the background thread."""
        try:
            result = identify_image(species_name, image_path, top_k=5, threshold=threshold)
            self.result_queue.put(("ok", result))
        except Exception as error:
            self.result_queue.put(("error", str(error)))

    def check_for_result(self):
        """Runs on the main thread every 100 ms and looks for a finished answer."""
        try:
            kind, content = self.result_queue.get_nowait()
        except queue.Empty:
            self.root.after(100, self.check_for_result)
            return

        self.run_button.config(state="normal")
        if kind == "ok":
            self.show_result(content)
        else:
            self.status_label.config(text="Something went wrong.")
            messagebox.showerror("Error", content)
        self.root.after(100, self.check_for_result)

    def show_result(self, result):
        self.status_label.config(text="Done.")
        self.species_result.config(text=f"Species: {result['species']}")
        self.individual_result.config(text=f"Individual: {result['individual']}")

        # How far ahead the best match is from the second best: a bigger gap means more confidence
        matches = result["top_matches"]
        similarity_text = f"Similarity of best match: {result['similarity']}"
        if len(matches) >= 2:
            gap = round(matches[0]["similarity"] - matches[1]["similarity"], 4)
            similarity_text += f"   (gap over 2nd match: {gap})"
        self.similarity_result.config(text=similarity_text)

        # Fill the top-matches box (it is read-only, so unlock it while writing)
        self.top_matches_box.config(state="normal")
        self.top_matches_box.delete("1.0", "end")
        for position, match in enumerate(matches, start=1):
            self.top_matches_box.insert("end", f"{position}. {match['individual']}   {match['similarity']}\n")
        self.top_matches_box.config(state="disabled")


if __name__ == "__main__":
    root = tk.Tk()
    WildlifeApp(root)
    root.mainloop()