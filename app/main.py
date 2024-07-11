import tkinter as tk
from tkinter import filedialog, messagebox
import os
import joblib
import datetime

import numpy as np
import pandas as pd

from data_processing import preprocess_data
from utilities import load_data_file, display_data


class HousingPriceApp:
    def __init__(self, root):
        self.root = root
        self.root.title('Bachelor thesis | KPI')
        self.root.configure(bg='light green')

        title_label = tk.Label(root, text='Прогнозування цін на купівлю житла', bg='light green')
        title_label.pack(pady=10)

        self.data_text = tk.Text(root, height=20, width=60)
        self.data_text.pack(pady=10)

        self.scrollbar = tk.Scrollbar(root, command=self.data_text.yview)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.data_text['yscrollcommand'] = self.scrollbar.set

        show_data_button = tk.Button(root, text='Відобразити дані', command=self.load_data)
        show_data_button.pack(pady=5)

        load_analysis_button = tk.Button(root, text='Завантажити дані для аналізу', command=self.load_analysis_data)
        load_analysis_button.pack(pady=5)
        
        save_result_button = tk.Button(root, text='Зберегти результат', command=self.save_result)
        save_result_button.pack(pady=5)

        self.load_models()

    def load_data(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if file_path:
            display_data(self.data_text, file_path)

    def load_analysis_data(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if file_path:
            try:
                data = pd.read_csv(file_path)
                messagebox.showinfo("Success", "Файл успішно завантажено для аналізу.")
                self.process_and_predict(data)
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load file: {e}")

    def load_models(self):
        try:
            current_file_path = os.path.realpath(__file__)
            current_directory = os.path.dirname(current_file_path)
            os.chdir(current_directory)

            self.ridge_model_full_data = joblib.load('../models/ridge_model_full_data.pkl')
            self.svr_model_full_data = joblib.load('../models/svr_model_full_data.pkl')
            self.gbr_model_full_data = joblib.load('../models/gbr_model_full_data.pkl')
            self.xgb_model_full_data = joblib.load('../models/xgb_model_full_data.pkl')
            self.lgb_model_full_data = joblib.load('../models/lgb_model_full_data.pkl')
            self.rf_model_full_data = joblib.load('../models/rf_model_full_data.pkl')
            self.stack_gen_model = joblib.load('../models/stack_gen_model.pkl')
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load models: {e}")

    def blended_predictions(self, X):
        return ((0.1 * self.ridge_model_full_data.predict(X)) + 
                (0.2 * self.svr_model_full_data.predict(X)) + 
                (0.1 * self.gbr_model_full_data.predict(X)) + 
                (0.1 * self.xgb_model_full_data.predict(X)) + 
                (0.1 * self.lgb_model_full_data.predict(X)) + 
                (0.05 * self.rf_model_full_data.predict(X)) + 
                (0.35 * self.stack_gen_model.predict(np.array(X))))

    def process_and_predict(self, data):
        try:
            data = preprocess_data(data)
            self.predictions = self.blended_predictions(data)
            self.data_text.delete(1.0, tk.END)
            self.data_text.insert(tk.END, "Id,SalePrice\n")
            for i, prediction in enumerate(self.predictions):
                self.data_text.insert(tk.END, f"{i+1},{np.floor(np.expm1(prediction))}\n")
        except Exception as e:
            messagebox.showerror("Error", f"Помилка під час обробки даних: {e}")

    def save_result(self):
        content = self.data_text.get(1.0, tk.END).strip()
        if not content or content == "Id,SalePrice":
            messagebox.showerror("Error", "Немає результату для збереження. Спочатку зробіть прогноз.")
            return
        
        try:
            filename = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".csv"
            file_path = os.path.join('..', 'data', filename)
            with open(file_path, 'w', newline='') as file:
                file.write(content)
            messagebox.showinfo("Save Result", f"Result saved as {file_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save result: {e}")


if __name__ == "__main__":
    root = tk.Tk()
    app = HousingPriceApp(root)
    root.mainloop()
