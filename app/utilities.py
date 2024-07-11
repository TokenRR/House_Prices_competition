import pandas as pd


def load_data_file(file_path):
    try:
        data = pd.read_csv(file_path)
        return data
    except Exception as e:
        print(f"Failed to load file: {e}")
        return None


def display_data(text_widget, file_path):
    try:
        data = pd.read_csv(file_path)
        text_widget.delete(1.0, 'end')
        text_widget.insert('end', data.to_string())
    except Exception as e:
        text_widget.delete(1.0, 'end')
        text_widget.insert('end', f"Failed to load file: {e}")
