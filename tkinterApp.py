# Author: Austin Snyder
# Date: October 13 2024

# TO START: 1st run "pip install tkinter" & "pip install numpy"
import tkinter as tk
import numpy as np

class DrawingApp:
    def __init__(self, master, n, file_name):
        self.saved_matrices = []
        self.master = master
        self.n = n
        self.cell_size = 100
        self.file_name = file_name

        # Create a canvas for drawing
        self.canvas = tk.Canvas(master, width=n * self.cell_size, height=n * self.cell_size, bg='white')
        self.canvas.pack()

        # Create a 2D array to store pixel data
        self.pixels = np.zeros((n, n), dtype=int)

        # Bind drawing to clicking for the mean time
        self.canvas.bind("<Button-1>", self.draw)

        # Button to save the drawn pixels to a file
        self.save_button = tk.Button(master, text="Save", command=self.save_to_file)
        self.clear_file = tk.Button(master, text="Clear File", command=self.clear_file)
        self.save_button.pack()
        self.clear_file.pack()

    def draw(self, event):
        # Calculate the grid position
        x = event.x // self.cell_size
        y = event.y // self.cell_size

        # Check boundaries
        if 0 <= x < self.n and 0 <= y < self.n:
            # Draw a filled rectangle
            if self.pixels[y,x] == 1:
                self.canvas.create_rectangle(x * self.cell_size, y * self.cell_size,
                                         (x + 1) * self.cell_size, (y + 1) * self.cell_size,
                                         fill='white', outline='black')
                self.pixels[y, x] = 0
            else:
                # Mark the pixel as drawn (1)
                self.canvas.create_rectangle(x * self.cell_size, y * self.cell_size,
                                             (x + 1) * self.cell_size, (y + 1) * self.cell_size,
                                             fill='black', outline='black')
                self.pixels[y, x] = 1

    def save_pixels(self, event):
        print("Pixels updated.")

    def save_to_file(self):
        # Flatten the current pixel matrix and append to the saved matrices
        new_vector = self.pixels.flatten()
        vector_string = ''.join(map(str, new_vector)) + '\n'

        if not self.vector_is_unique(vector_string):
            print("This was a duplicate matrix. Not Saving.")
            return

        # Save all matrices to a file
        with open(self.file_name, "a") as f:
            f.write(vector_string)

        print("All matrices saved to pixels.txt.")

    def clear_file(self):
        print("File was cleared")
        with open(self.file_name, 'w') as file:
            pass

    def vector_is_unique(self, new_vector):
        # Read existing entries from the file character by character
        try:
            with open(self.file_name, "r") as f:
                current_entry = ""
                while True:
                    # Read 1 char at a time
                    char = f.read(1)
                    if not char:
                        break

                    # add to current string
                    current_entry += char
                    if char == '\n':  # Entry completed
                        # check if the entry is equal to the one we try to add
                        print(f"Comparing: '{current_entry}' with '{new_vector}'")

                        if current_entry == new_vector:
                            return False

                        # reset for the next time around
                        current_entry = ""
        except FileNotFoundError:
            pass

        return True

if __name__ == "__main__":
    # CHANGE THIS TO CHANGE BETWEEN 3x3 OR 5x5
    n = 5

    # CHANGE THIS TO CHANGE THE FILE YOU WRITE TO
    file_name = "data5l.txt"

    # CLICK ON SQUARES, ONCE FILLED IN THE CORRECT ONES, CLICK SAVE AND IT WILL
    # APPEND IT TO THE FILE YOU HAVE ABOVE, IT WILL ALSO CHECK FOR DUPLICATE VALUES

    root = tk.Tk()
    # file name to save to
    app = DrawingApp(root, n, file_name)
    root.title("Draw n x n. Save to flattened array.")
    root.mainloop()