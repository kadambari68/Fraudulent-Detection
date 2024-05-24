import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import matplotlib.backends.backend_tkagg as tkagg

class FraudDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Fraud Detection App")
        self.root.geometry("800x600")

        # Menu Bar
        menubar = tk.Menu(root)
        filemenu = tk.Menu(menubar, tearoff=0)
        filemenu.add_command(label="Load Transaction Data", command=self.load_data)
        filemenu.add_separator()
        filemenu.add_command(label="Exit", command=root.quit)
        menubar.add_cascade(label="File", menu=filemenu)
        self.root.config(menu=menubar)

        # Project Information
        project_info = """
                                   Welcome to Fraud Detection App!
        
        This application helps you detect fraudulent transactions in your dataset.
        """

        self.info_label = tk.Label(self.root, text=project_info, wraplength=600, justify='left', font=("Arial", 12), fg="blue")
        self.info_label.pack(pady=10)

        # Status Bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        self.status_bar = tk.Label(self.root, textvariable=self.status_var, bd=1, relief=tk.SUNKEN, anchor=tk.W, font=("Arial", 10, "italic"), bg="lightgrey")
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

        # Plot Frame
        self.plot_frame = tk.Frame(self.root)
        self.plot_frame.pack(expand=True, fill=tk.BOTH)

        # Analyze Button
        self.analyze_button = tk.Button(self.root, text="Analyze Data", command=self.analyze_data, state=tk.DISABLED, bg="green", fg="white", font=("Arial", 12, "bold"))
        self.analyze_button.pack(side=tk.TOP, pady=10)

    def load_data(self):
        self.filename = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if self.filename:
            self.analyze_button.config(state=tk.NORMAL)

    def analyze_data(self):
        try:
            self.status_var.set("Analyzing data...")
            
            # Load the transaction data
            transaction_data = pd.read_csv(self.filename)

            # Convert 'Time' column to seconds elapsed from the start
            transaction_data['Time'] = pd.to_datetime(transaction_data['Time'])
            transaction_data['Time'] = (transaction_data['Time'] - transaction_data['Time'].min()).dt.total_seconds()

            # Select relevant features for fraud detection
            features = ['Amount', 'Time']

            # Preprocess the data
            scaler = StandardScaler()
            data_scaled = scaler.fit_transform(transaction_data[features])

            # Apply anomaly detection using Isolation Forest
            isolation_forest = IsolationForest(contamination=0.01, random_state=42)
            isolation_forest.fit(data_scaled)

            # Predict the anomalies
            anomaly_predictions = isolation_forest.predict(data_scaled)

            # Add the anomaly predictions to the transaction data
            transaction_data['IsFraud'] = anomaly_predictions

            # Visualize the anomalies
            fraudulent_transactions = transaction_data[transaction_data['IsFraud'] == -1]
            non_fraudulent_transactions = transaction_data[transaction_data['IsFraud'] == 1]

            plt.figure(figsize=(8, 6))
            plt.scatter(non_fraudulent_transactions['Time'], non_fraudulent_transactions['Amount'], color='blue', label='Non-Fraudulent')
            plt.scatter(fraudulent_transactions['Time'], fraudulent_transactions['Amount'], color='red', label='Fraudulent')
            plt.xlabel('Time', fontdict={"fontsize": 12, "fontweight": "bold"})
            plt.ylabel('Amount', fontdict={"fontsize": 12, "fontweight": "bold"})
            plt.title('Fraud Detection', fontdict={"fontsize": 14, "fontweight": "bold"})
            plt.legend()
            plt.tight_layout()

            # Clear previous plot (if any) and display new plot
            for widget in self.plot_frame.winfo_children():
                widget.destroy()
            self.canvas = plt.gcf()
            self.canvas.set_size_inches(8, 6)
            self.canvas = tkagg.FigureCanvasTkAgg(self.canvas, master=self.plot_frame)
            self.canvas.draw()
            self.canvas.get_tk_widget().pack(expand=True, fill=tk.BOTH)

            self.status_var.set("Analysis complete")

        except Exception as e:
            messagebox.showerror("Error", str(e))
            self.status_var.set("Error")

root = tk.Tk()
app = FraudDetectionApp(root)
root.mainloop()
