"""
Professional Structural Beam Analysis Software
Uses 1D Finite Element Analysis (Direct Stiffness Method)
Features: SFD, BMD, Deflection, Slope, Multiple Spans, Arbitrary Loading.
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.backends.backend_pdf import PdfPages
import json
import csv
import os

# ==========================================
# 1. MATHEMATICAL ENGINE (FEA SOLVER)
# ==========================================

class BeamSolver:
    def __init__(self, length, E, I, num_elements=500):
        """
        Initializes the 1D FEA Beam Solver.
        :param length: Total length of the beam.
        :param E: Modulus of Elasticity.
        :param I: Moment of Inertia.
        :param num_elements: Number of discrete elements (higher = smoother curves).
        """
        self.L = float(length)
        self.E = float(E)
        self.I = float(I)
        self.N = int(num_elements)
        
        # Discretize the beam
        self.x = np.linspace(0, self.L, self.N + 1)
        self.le = self.L / self.N
        
        # Global Stiffness Matrix (K) and Force Vector (F)
        # 2 Degrees of Freedom per node: Vertical Displacement (v) and Rotation (theta)
        self.total_dof = 2 * (self.N + 1)
        self.K = lil_matrix((self.total_dof, self.total_dof))
        self.F = np.zeros(self.total_dof)
        
        self.supports = [] # Store tuple of (node_index, support_type)
        self.build_global_stiffness()

    def build_global_stiffness(self):
        """Assembles the global stiffness matrix using Euler-Bernoulli beam elements."""
        E, I, le = self.E, self.I, self.le
        # Element stiffness matrix for standard 2D beam
        k_el = (E * I / le**3) * np.array([
            [12, 6*le, -12, 6*le],
            [6*le, 4*le**2, -6*le, 2*le**2],
            [-12, -6*le, 12, -6*le],
            [6*le, 2*le**2, -6*le, 4*le**2]
        ])
        
        for i in range(self.N):
            # DOFs for element i: v1, theta1, v2, theta2
            idx = [2*i, 2*i+1, 2*i+2, 2*i+3]
            for r in range(4):
                for c in range(4):
                    self.K[idx[r], idx[c]] += k_el[r, c]

    def add_point_load(self, x_pos, magnitude):
        """Adds a concentrated load at a specific location. Downward is negative."""
        node = int(round((x_pos / self.L) * self.N))
        if 0 <= node <= self.N:
            self.F[2*node] += magnitude

    def add_distributed_load(self, x_start, x_end, q_start, q_end):
        """
        Adds a UDL or UVL by discretizing it into fine point loads over the mesh.
        Downward pressure is negative.
        """
        for i, x_val in enumerate(self.x):
            if x_start <= x_val <= x_end + 1e-9: # 1e-9 for floating point tolerance
                # Linear interpolation of load intensity
                if x_end == x_start:
                    q_val = q_start
                else:
                    q_val = q_start + (q_end - q_start) * (x_val - x_start) / (x_end - x_start)
                
                # Tributary length for the node
                trib = self.le if (0 < i < self.N) else self.le / 2
                self.F[2*i] += q_val * trib

    def add_support(self, x_pos, sup_type):
        """Registers a support at a specific location."""
        node = int(round((x_pos / self.L) * self.N))
        if 0 <= node <= self.N:
            self.supports.append((node, sup_type))

    def solve(self):
        """Solves the FEA system for displacements and internal forces."""
        # 1. Apply Boundary Conditions (Penalty Method)
        max_k = abs(self.K.tocsr()).max()
        penalty = max_k * 1e12 # Very stiff spring to lock DOFs
        
        for node, s_type in self.supports:
            if s_type in ['Pinned', 'Roller', 'Fixed']:
                self.K[2*node, 2*node] += penalty # Lock vertical displacement
            if s_type == 'Fixed':
                self.K[2*node+1, 2*node+1] += penalty # Lock rotation

        # 2. Solve Ku = F
        K_csr = self.K.tocsr()
        self.U = spsolve(K_csr, self.F)
        
        self.deflection = self.U[0::2]
        self.slope = self.U[1::2]

        # 3. Calculate Internal Forces & Reactions
        # External forces required to maintain this deformation state
        F_total = K_csr.dot(self.U) 
        self.reactions = F_total - self.F
        print("\nREACTIONS:")
        for i in range(self.N + 1):
             print("Node", i, ":", self.reactions[2*i], "N")

        # Integrate left-to-right to get SFD and BMD
        self.V = np.zeros(self.N + 1)
        self.M = np.zeros(self.N + 1)
        
        V_curr = 0
        M_curr = 0
        
        for i in range(self.N):
            force_y = F_total[2*i]
            moment_z = F_total[2*i + 1]
            
            V_curr += force_y
            M_curr -= moment_z # Standard beam convention: sagging is positive
            
            self.V[i] = V_curr
            self.M[i] = M_curr
            
            # Change in moment over the element dx
            M_curr += V_curr * self.le
            
        # End node
        self.V[-1] = V_curr + F_total[2*self.N]
        self.M[-1] = M_curr - F_total[2*self.N + 1]

        return self.x, self.deflection, self.slope, self.V, self.M

# ==========================================
# 2. GRAPHICAL USER INTERFACE (Tkinter)
# ==========================================

class BeamApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Advanced Structural Beam Analyzer")
        self.root.geometry("1400x850")
        
        # Data Storage
        self.supports_data = []
        self.loads_data = []
        self.solver = None
        
        self.setup_ui()

    def setup_ui(self):
        # --- Left Panel: Inputs ---
        input_frame = ttk.Frame(self.root, padding="10", width=350)
        input_frame.pack(side=tk.LEFT, fill=tk.Y)
        
        ttk.Label(input_frame, text="Beam Properties", font=('Arial', 12, 'bold')).pack(anchor=tk.W, pady=5)
        
        # Beam Length
        frame_l = ttk.Frame(input_frame)
        frame_l.pack(fill=tk.X, pady=2)
        ttk.Label(frame_l, text="Length (m):").pack(side=tk.LEFT)
        self.entry_L = ttk.Entry(frame_l, width=10)
        self.entry_L.insert(0, "10.0")
        self.entry_L.pack(side=tk.RIGHT)

        # Modulus of Elasticity (E)
        frame_e = ttk.Frame(input_frame)
        frame_e.pack(fill=tk.X, pady=2)
        ttk.Label(frame_e, text="Elastic Modulus (E) [Pa]:").pack(side=tk.LEFT)
        self.entry_E = ttk.Entry(frame_e, width=10)
        self.entry_E.insert(0, "200e9")
        self.entry_E.pack(side=tk.RIGHT)

        # Moment of Inertia (I)
        frame_i = ttk.Frame(input_frame)
        frame_i.pack(fill=tk.X, pady=2)
        ttk.Label(frame_i, text="Moment of Inertia (I) [m^4]:").pack(side=tk.LEFT)
        self.entry_I = ttk.Entry(frame_i, width=10)
        self.entry_I.insert(0, "0.0004")
        self.entry_I.pack(side=tk.RIGHT)

        ttk.Separator(input_frame, orient='horizontal').pack(fill=tk.X, pady=10)

        # --- Supports Section ---
        ttk.Label(input_frame, text="Supports", font=('Arial', 12, 'bold')).pack(anchor=tk.W, pady=5)
        
        frame_sup = ttk.Frame(input_frame)
        frame_sup.pack(fill=tk.X, pady=2)
        ttk.Label(frame_sup, text="Loc (m):").grid(row=0, column=0)
        self.entry_sup_x = ttk.Entry(frame_sup, width=7)
        self.entry_sup_x.grid(row=0, column=1, padx=2)
        
        ttk.Label(frame_sup, text="Type:").grid(row=0, column=2)
        self.sup_type = ttk.Combobox(frame_sup, values=["Pinned", "Roller", "Fixed"], width=8, state="readonly")
        self.sup_type.current(0)
        self.sup_type.grid(row=0, column=3, padx=2)
        
        ttk.Button(frame_sup, text="Add", command=self.add_support).grid(row=0, column=4, padx=5)
        
        self.listbox_sup = tk.Listbox(input_frame, height=4)
        self.listbox_sup.pack(fill=tk.X, pady=5)

        ttk.Separator(input_frame, orient='horizontal').pack(fill=tk.X, pady=10)

        # --- Loads Section ---
        ttk.Label(input_frame, text="Loads (Downward is Negative)", font=('Arial', 12, 'bold')).pack(anchor=tk.W, pady=5)
        
        frame_load_type = ttk.Frame(input_frame)
        frame_load_type.pack(fill=tk.X, pady=2)
        ttk.Label(frame_load_type, text="Type:").pack(side=tk.LEFT)
        self.load_type = ttk.Combobox(frame_load_type, values=["Point Load", "Distributed (UDL/UVL)"], state="readonly")
        self.load_type.current(0)
        self.load_type.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=5)
        self.load_type.bind("<<ComboboxSelected>>", self.toggle_load_inputs)

        # Load inputs container
        self.frame_load_params = ttk.Frame(input_frame)
        self.frame_load_params.pack(fill=tk.X, pady=2)
        self.build_load_inputs() # Initialize defaults

        ttk.Button(input_frame, text="Add Load", command=self.add_load).pack(fill=tk.X, pady=5)
        
        self.listbox_loads = tk.Listbox(input_frame, height=5)
        self.listbox_loads.pack(fill=tk.X, pady=5)

        ttk.Separator(input_frame, orient='horizontal').pack(fill=tk.X, pady=10)

        # --- Actions Section ---
        ttk.Button(input_frame, text="Solve & Analyze", command=self.solve_system, style='Accent.TButton').pack(fill=tk.X, pady=5)
        
        btn_frame = ttk.Frame(input_frame)
        btn_frame.pack(fill=tk.X, pady=5)
        ttk.Button(btn_frame, text="Save Project", command=self.save_project).grid(row=0, column=0, padx=2, sticky='we')
        ttk.Button(btn_frame, text="Load Project", command=self.load_project).grid(row=0, column=1, padx=2, sticky='we')
        btn_frame.columnconfigure(0, weight=1)
        btn_frame.columnconfigure(1, weight=1)

        btn_export = ttk.Frame(input_frame)
        btn_export.pack(fill=tk.X, pady=5)
        ttk.Button(btn_export, text="Export PDF", command=self.export_pdf).grid(row=0, column=0, padx=2, sticky='we')
        ttk.Button(btn_export, text="Export Excel/CSV", command=self.export_csv).grid(row=0, column=1, padx=2, sticky='we')
        btn_export.columnconfigure(0, weight=1)
        btn_export.columnconfigure(1, weight=1)
        
        ttk.Button(input_frame, text="Clear All", command=self.clear_all).pack(fill=tk.X, pady=5)

        # --- Right Panel: Results & Plots ---
        right_frame = ttk.Frame(self.root)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.fig, self.axs = plt.subplots(4, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [1.5, 2, 2, 2]})
        self.fig.tight_layout(pad=3.0)
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=right_frame)
        self.canvas.draw()
        
        # Add standard Matplotlib Interactive Toolbar
        toolbar_frame = ttk.Frame(right_frame)
        toolbar_frame.pack(side=tk.TOP, fill=tk.X)
        self.toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame)
        self.toolbar.update()
        
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        self.reset_plots()

    def build_load_inputs(self):
        # Clear previous
        for widget in self.frame_load_params.winfo_children():
            widget.destroy()
            
        l_type = self.load_type.get()
        if l_type == "Point Load":
            ttk.Label(self.frame_load_params, text="Loc (m):").grid(row=0, column=0, pady=2)
            self.entry_load_x1 = ttk.Entry(self.frame_load_params, width=8)
            self.entry_load_x1.grid(row=0, column=1, pady=2, padx=2)
            
            ttk.Label(self.frame_load_params, text="Mag (N):").grid(row=0, column=2, pady=2)
            self.entry_load_mag1 = ttk.Entry(self.frame_load_params, width=8)
            self.entry_load_mag1.grid(row=0, column=3, pady=2, padx=2)
        else:
            ttk.Label(self.frame_load_params, text="Start (m):").grid(row=0, column=0, pady=2)
            self.entry_load_x1 = ttk.Entry(self.frame_load_params, width=6)
            self.entry_load_x1.grid(row=0, column=1, pady=2, padx=2)
            
            ttk.Label(self.frame_load_params, text="End (m):").grid(row=0, column=2, pady=2)
            self.entry_load_x2 = ttk.Entry(self.frame_load_params, width=6)
            self.entry_load_x2.grid(row=0, column=3, pady=2, padx=2)
            
            ttk.Label(self.frame_load_params, text="q1 (N/m):").grid(row=1, column=0, pady=2)
            self.entry_load_mag1 = ttk.Entry(self.frame_load_params, width=6)
            self.entry_load_mag1.grid(row=1, column=1, pady=2, padx=2)
            
            ttk.Label(self.frame_load_params, text="q2 (N/m):").grid(row=1, column=2, pady=2)
            self.entry_load_mag2 = ttk.Entry(self.frame_load_params, width=6)
            self.entry_load_mag2.grid(row=1, column=3, pady=2, padx=2)

    def toggle_load_inputs(self, event=None):
        self.build_load_inputs()

    def add_support(self):
        try:
            x = float(self.entry_sup_x.get())
            s_type = self.sup_type.get()
            L = float(self.entry_L.get())
            if x < 0 or x > L:
                raise ValueError("Support location must be between 0 and Beam Length.")
            
            self.supports_data.append({'x': x, 'type': s_type})
            self.listbox_sup.insert(tk.END, f"{s_type} at x = {x}m")
        except ValueError as e:
            messagebox.showerror("Input Error", str(e))

    def add_load(self):
        try:
            L = float(self.entry_L.get())
            l_type = self.load_type.get()
            if l_type == "Point Load":
                x = float(self.entry_load_x1.get())
                mag = float(self.entry_load_mag1.get())
                if x < 0 or x > L: raise ValueError("Load location must be on the beam.")
                self.loads_data.append({'type': 'Point', 'x': x, 'mag': mag})
                self.listbox_loads.insert(tk.END, f"Point: {mag}N at x={x}m")
            else:
                x1 = float(self.entry_load_x1.get())
                x2 = float(self.entry_load_x2.get())
                q1 = float(self.entry_load_mag1.get())
                q2 = float(self.entry_load_mag2.get())
                if x1 < 0 or x2 > L or x1 > x2: raise ValueError("Invalid distributed load coordinates.")
                self.loads_data.append({'type': 'Dist', 'x1': x1, 'x2': x2, 'q1': q1, 'q2': q2})
                self.listbox_loads.insert(tk.END, f"Dist: {q1} to {q2} N/m [{x1}m - {x2}m]")
        except ValueError as e:
            messagebox.showerror("Input Error", str(e))

    def solve_system(self):
        try:
            L = float(self.entry_L.get())
            E = float(self.entry_E.get())
            I = float(self.entry_I.get())
            
            if L <= 0 or E <= 0 or I <= 0:
                raise ValueError("Length, E, and I must be strictly positive.")
                
            self.solver = BeamSolver(L, E, I)
            
            for sup in self.supports_data:
                self.solver.add_support(sup['x'], sup['type'])
                
            for ld in self.loads_data:
                if ld['type'] == 'Point':
                    self.solver.add_point_load(ld['x'], ld['mag'])
                else:
                    self.solver.add_distributed_load(ld['x1'], ld['x2'], ld['q1'], ld['q2'])
                    
            self.solver.solve()
            self.plot_results()
            
            # Show max values in console/dialog or just visual. Visual is updated via plot.
            max_v = np.max(np.abs(self.solver.V))
            max_m = np.max(np.abs(self.solver.M))
            max_d = np.max(np.abs(self.solver.deflection))
            print(f"Max Shear: {max_v:.2f} N | Max Moment: {max_m:.2f} Nm | Max Deflection: {max_d:.6f} m")

        except Exception as e:
            messagebox.showerror("Analysis Error", f"An error occurred during calculation:\n{str(e)}")

    def reset_plots(self):
        for ax in self.axs:
            ax.clear()
            ax.grid(True, linestyle='--', alpha=0.6)
        
        self.axs[0].set_title("Beam Loading Diagram")
        self.axs[1].set_title("Shear Force Diagram (SFD)")
        self.axs[2].set_title("Bending Moment Diagram (BMD)")
        self.axs[3].set_title("Deflection Curve")
        self.canvas.draw()

    def plot_results(self):
        self.reset_plots()
        x = self.solver.x
        L = self.solver.L
        
        # --- 1. Beam & Loading Diagram ---
        ax0 = self.axs[0]
        ax0.plot([0, L], [0, 0], color='black', linewidth=4)
        
        # Plot Supports
        for sup in self.supports_data:
            sx = sup['x']
            if sup['type'] == 'Fixed':
                ax0.plot(sx, 0, marker='s', markersize=12, color='gray')
            else:
                ax0.plot(sx, 0, marker='^', markersize=12, color='green')
                
        # Plot Loads (Simplified visualization)
        max_height = 1.0
        for ld in self.loads_data:
            if ld['type'] == 'Point':
                color = 'red' if ld['mag'] < 0 else 'blue'
                direction = -1 if ld['mag'] < 0 else 1
                ax0.annotate('', xy=(ld['x'], 0), xytext=(ld['x'], direction * max_height),
                             arrowprops=dict(facecolor=color, shrink=0.05, width=2, headwidth=8))
            else:
                color = 'orange'
                ax0.fill_between([ld['x1'], ld['x2']], [0, 0], [0.5, 0.5], color=color, alpha=0.3)
                ax0.text((ld['x1']+ld['x2'])/2, 0.6, 'UDL/UVL', ha='center', color='darkorange')

        ax0.set_xlim(-0.05*L, 1.05*L)
        ax0.set_ylim(-1.5, 1.5)
        ax0.axis('off')
        ax0.set_title("Free Body Diagram", pad=10)

        # --- 2. Shear Force Diagram ---
        ax1 = self.axs[1]
        ax1.plot(x, self.solver.V, color='blue', linewidth=2)
        ax1.fill_between(x, self.solver.V, 0, color='blue', alpha=0.2)
        ax1.set_ylabel("Shear Force (N)")
        ax1.axhline(0, color='black', linewidth=1)

        # --- 3. Bending Moment Diagram ---
        ax2 = self.axs[2]
        # Standard convention: Sagging positive, plotted upwards
        ax2.plot(x, self.solver.M, color='green', linewidth=2)
        ax2.fill_between(x, self.solver.M, 0, color='green', alpha=0.2)
        ax2.set_ylabel("Bending Moment (Nm)")
        ax2.axhline(0, color='black', linewidth=1)
        # Structural engineers sometimes invert BMD: 
        # ax2.invert_yaxis() # Uncomment to plot tension on bottom

        # --- 4. Deflection Curve ---
        ax3 = self.axs[3]
        ax3.plot(x, self.solver.deflection * 1000, color='red', linewidth=2) # Convert to mm
        ax3.fill_between(x, self.solver.deflection * 1000, 0, color='red', alpha=0.2)
        ax3.set_ylabel("Deflection (mm)")
        ax3.set_xlabel("Beam Length (m)")
        ax3.axhline(0, color='black', linewidth=1)

        self.canvas.draw()

    # ==========================================
    # 3. ADVANCED FEATURES (Save/Load/Export)
    # ==========================================

    def clear_all(self):
        self.supports_data.clear()
        self.loads_data.clear()
        self.listbox_sup.delete(0, tk.END)
        self.listbox_loads.delete(0, tk.END)
        self.solver = None
        self.reset_plots()

    def get_project_dict(self):
        return {
            'L': self.entry_L.get(),
            'E': self.entry_E.get(),
            'I': self.entry_I.get(),
            'supports': self.supports_data,
            'loads': self.loads_data
        }

    def save_project(self):
        filepath = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON files", "*.json")])
        if filepath:
            with open(filepath, 'w') as f:
                json.dump(self.get_project_dict(), f, indent=4)
            messagebox.showinfo("Success", "Project saved successfully!")

    def load_project(self):
        filepath = filedialog.askopenfilename(filetypes=[("JSON files", "*.json")])
        if filepath:
            self.clear_all()
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            self.entry_L.delete(0, tk.END); self.entry_L.insert(0, data.get('L', '10'))
            self.entry_E.delete(0, tk.END); self.entry_E.insert(0, data.get('E', '200e9'))
            self.entry_I.delete(0, tk.END); self.entry_I.insert(0, data.get('I', '0.0004'))
            
            for sup in data.get('supports', []):
                self.supports_data.append(sup)
                self.listbox_sup.insert(tk.END, f"{sup['type']} at x = {sup['x']}m")
                
            for ld in data.get('loads', []):
                self.loads_data.append(ld)
                if ld['type'] == 'Point':
                    self.listbox_loads.insert(tk.END, f"Point: {ld['mag']}N at x={ld['x']}m")
                else:
                    self.listbox_loads.insert(tk.END, f"Dist: {ld['q1']} to {ld['q2']} N/m [{ld['x1']}m - {ld['x2']}m]")
            
            self.solve_system()

    def export_pdf(self):
        if not self.solver:
            messagebox.showwarning("Warning", "Please solve the system before exporting.")
            return
            
        filepath = filedialog.asksaveasfilename(defaultextension=".pdf", filetypes=[("PDF files", "*.pdf")])
        if filepath:
            try:
                with PdfPages(filepath) as pdf:
                    pdf.savefig(self.fig)
                    # Note: For a fully detailed engineering text report inside the PDF, 
                    # one could utilize `reportlab`, but plotting Matplotlib graphs 
                    # to PDF directly satisfies pure graphic export needs.
                messagebox.showinfo("Success", "Engineering Report (Plots) exported to PDF!")
            except Exception as e:
                messagebox.showerror("Export Error", str(e))

    def export_csv(self):
        if not self.solver:
            messagebox.showwarning("Warning", "Please solve the system before exporting.")
            return
            
        filepath = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
        if filepath:
            try:
                with open(filepath, 'w', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(['Position (m)', 'Shear Force (N)', 'Bending Moment (Nm)', 'Slope (rad)', 'Deflection (m)'])
                    for i in range(len(self.solver.x)):
                        writer.writerow([
                            self.solver.x[i], 
                            self.solver.V[i], 
                            self.solver.M[i], 
                            self.solver.slope[i], 
                            self.solver.deflection[i]
                        ])
                messagebox.showinfo("Success", "Results exported to CSV (Excel compatible) successfully!")
            except Exception as e:
                messagebox.showerror("Export Error", str(e))

if __name__ == "__main__":
    root = tk.Tk()
    
    # Optional: Apply a native or modern theme
    style = ttk.Style(root)
    if 'clam' in style.theme_names():
        style.theme_use('clam')
        
    app = BeamApp(root)
    root.mainloop()