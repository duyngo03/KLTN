import pickle
import tkinter as tk
from tkinter import messagebox, Listbox, Scrollbar, ttk

def load_model(filename):
    """Load the trained model from a file."""
    with open(filename, 'rb') as file:
        model = pickle.load(file)
    print(f"Model loaded from {filename}")
    return model

def suggest_next_word(model, lambdas, input_words, top_n=5):

    input_words = input_words.split()

    w3 = input_words[-3] if len(input_words) > 2 else ""
    w2 = input_words[-2] if len(input_words) > 1 else ""
    w1 = input_words[-1] if len(input_words) > 0 else ""

    probabilities = {}

    for w in model['unigram']:
        # Lấy xác suất cho từng n-gram từ model
        unigram_prob = model['unigram'].get(w, 0)
        bigram_prob = model['bigram'].get((w1, w[0]), 0)
        trigram_prob = model['trigram'].get((w2, w1, w[0]), 0)
        fourgram_prob = model['fourgram'].get((w3, w2, w1, w[0]), 0)
        # Tính tổng xác suất bằng cách kết hợp các xác suất với trọng số lambda
        total_prob = (
            lambdas[0] * unigram_prob + 
            lambdas[1] * bigram_prob + 
            lambdas[2] * trigram_prob + 
            lambdas[3] * fourgram_prob
        )
        
        # Lưu xác suất của từ vào dictionary
        probabilities[w[0]] = total_prob

    # top_suggestions = sorted(probabilities.items(), key=lambda item: item[1], reverse=True)[:top_n]

    top_suggestions = sorted(probabilities.keys(), key=lambda w: probabilities[w], reverse=True)[:top_n]

    return top_suggestions
#Build UI
def on_suggest_click():
    input_words = entry.get()  # Lấy từ nhập vào từ ô input
    if not input_words.strip():  # Kiểm tra nếu không có từ nhập vào
        messagebox.showwarning("Warning", "Vui lòng nhập từ để gợi ý.")
        return
    
    top_suggestions = suggest_next_word(model, lambdas, input_words)
    
    # Xóa bảng cũ trước khi hiển thị kết quả mới
    for row in suggestions_table.get_children():
        suggestions_table.delete(row)

    # Thêm các kết quả mới vào bảng
    for idx, word in enumerate(top_suggestions, start=1):
        suggestions_table.insert("", "end", values=(idx, word))

# Tải model
model = load_model("model_suggest_in_line.pkl")

# Định nghĩa trọng số lambda
lambdas = [0.0145, 0.0551, 0.2664, 0.664]

# Khởi tạo cửa sổ Tkinter
root = tk.Tk()
root.title("Gợi ý Từ Tiếp Theo")
root.geometry("400x350")  # Kích thước cửa sổ

# Tạo ô nhập liệu cho người dùng nhập từ
entry_label = tk.Label(root, text="Nhập từ:", font=("Helvetica", 14))
entry_label.pack(pady=10)

entry = tk.Entry(root, width=40, font=("Helvetica", 12))
entry.pack(pady=10)

# Tạo nút "Gợi ý"
suggest_button = tk.Button(root, text="Gợi ý", font=("Helvetica", 12), command=on_suggest_click, fg="black")
suggest_button.pack(pady=10)

# Tạo bảng Treeview để hiển thị kết quả
columns = ("STT", "Từ Gợi Ý")
suggestions_table = ttk.Treeview(root, columns=columns, show="headings", height=10)

# Định nghĩa tiêu đề cho từng cột
suggestions_table.heading("STT", text="STT")
suggestions_table.heading("Từ Gợi Ý", text="Từ Gợi Ý")

# Định nghĩa kích thước từng cột
suggestions_table.column("STT", width=50, anchor="center")
suggestions_table.column("Từ Gợi Ý", width=300, anchor="center")

suggestions_table.pack(pady=10)

# Chạy ứng dụng Tkinter
root.mainloop()