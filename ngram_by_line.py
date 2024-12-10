import pickle
import tkinter as tk
import re
from tkinter import messagebox, Listbox, Scrollbar, ttk
from collections import defaultdict

special_patterns = ['<string>', '<file>', '<directory>', '<number>', '<URL>']
skip_prefixes = {"exit", "clear", "ssh"}
special_words = ['|', '||', '&&', '&', '>', '>>', ';' ]

def load_model(filename):
    """Load the trained model from a file."""
    with open(filename, 'rb') as file:
        model = pickle.load(file)
    print(f"Model loaded from {filename}")
    return model

def merge_similar_keys(sorted_commands):
    merged_commands = {}
    
    for command, prob in sorted_commands:
        # Kiểm tra nếu command có thể gộp vào key đã có trong merged_commands
        merged = False
        for key in list(merged_commands.keys()):
            if command.startswith(key) and key.split()[0] == command.split()[0]:  # Nếu command là phần của key
                merged_commands[key]['total_prob'] = merged_commands[key]['total_prob'] + prob
                merged_commands[key]['count'] += 1
                merged = True
            if key.startswith(command) and key.split()[0] == command.split()[0]:  # Nếu key là phần của command
                if command not in merged_commands:
                    merged_commands[command] = {'total_prob': prob, 'count': 1}
                merged_commands[command]['total_prob'] += merged_commands[key]['total_prob']
                merged_commands[command]['count'] += merged_commands[key]['count']
                del merged_commands[key]
                merged = True
        
        if not merged:
            # Nếu không có key nào tương đồng, thêm mới vào merged_commands
            merged_commands[command] = {'total_prob': prob, 'count': 1}
    result = []
    # for key, value in merged_commands.items():
    #     average_prob = value['total_prob'] / value['count']  # Tính trung bình xác suất
    #     result.append((key, average_prob))
    
    # result = sorted(result, key=lambda x: x[1], reverse=True)
    result = [key for key, value in merged_commands.items()]
    result.sort(key=lambda key: merged_commands[key]['total_prob'] / merged_commands[key]['count'], reverse=True)
    
    return result


def get_command_prefix(command): 
    parts = command.split()
    prefix_parts = []
    for part in parts:
        if part.startswith('-') or (part in special_patterns) or not re.search(r'[a-zA-Z]', part) or not part.isalpha():
            break
        prefix_parts.append(part)

    return ' '.join(prefix_parts)

def calculate_next_command_probabilities(history, model, lambdas, n = 5):
    prev_command_1 = history[-1] if len(history) > 0 else "null"
    prev_command_2 = history[-2] if len(history) > 1 else "null"
    prev_command_3 = history[-3] if len(history) > 2 else "null"
    prev_command_4 = history[-4] if len(history) > 3 else "null"
    prev_command_5 = history[-5] if len(history) > 4 else "null"
    prev_command_6 = history[-6] if len(history) > 5 else "null"
    prev_command_7 = history[-7] if len(history) > 6 else "null"
    prev_command_8 = history[-8] if len(history) > 7 else "null"
    prev_command_9 = history[-9] if len(history) > 8 else "null"
    
    prev_command_1_cleaned = get_command_prefix(prev_command_1)
    prev_command_2_cleaned = get_command_prefix(prev_command_2)
    prev_command_3_cleaned = get_command_prefix(prev_command_3)
    prev_command_4_cleaned = get_command_prefix(prev_command_4)
    prev_command_5_cleaned = get_command_prefix(prev_command_5)
    prev_command_6_cleaned = get_command_prefix(prev_command_6)
    prev_command_7_cleaned = get_command_prefix(prev_command_7)
    prev_command_8_cleaned = get_command_prefix(prev_command_8)
    prev_command_9_cleaned = get_command_prefix(prev_command_9)

    context_prev_cmd1 = " ".join(prev_command_1_cleaned.split()[:1])
    context_prev_cmd2 = " ".join(prev_command_2_cleaned.split()[:1])
    context_prev_cmd3 = " ".join(prev_command_3_cleaned.split()[:1])
    context_prev_cmd4 = " ".join(prev_command_4_cleaned.split()[:1])
    context_prev_cmd5 = " ".join(prev_command_5_cleaned.split()[:1])
    context_prev_cmd6 = " ".join(prev_command_6_cleaned.split()[:1])
    context_prev_cmd7 = " ".join(prev_command_7_cleaned.split()[:1])
    context_prev_cmd8 = " ".join(prev_command_8_cleaned.split()[:1])
    context_prev_cmd9 = " ".join(prev_command_9_cleaned.split()[:1])

    next_command_probs = {}
    
    for next_command in model['unigram_probs'].keys():
        next_command_cleaned = get_command_prefix(next_command[0])
        context_next_command = " ".join(next_command_cleaned.split()[:1])
        if model['context_bigram_prob'].get((context_prev_cmd1, context_next_command), 0) == 0:
            continue

        score = (
            lambdas[0] * model['bigram_probs'].get((prev_command_1, next_command[0]), 0) +
            lambdas[1] * model['trigram_probs'].get((prev_command_2, prev_command_1, next_command[0]), 0) +
            lambdas[2] * model['fourgram_probs'].get((prev_command_3, prev_command_2, prev_command_1, next_command[0]), 0) +
            lambdas[3] * model['fivegram_probs'].get((prev_command_4, prev_command_3, prev_command_2, prev_command_1, next_command[0]), 0) +
            lambdas[4] * model['sixgram_probs'].get((prev_command_5, prev_command_4, prev_command_3, prev_command_2, prev_command_1, next_command[0]), 0) +
            lambdas[5] * model['sevengram_probs'].get((prev_command_6, prev_command_5, prev_command_4, prev_command_3, prev_command_2, prev_command_1, next_command[0]), 0) +
            lambdas[6] * model['eightgram_probs'].get((prev_command_7, prev_command_6, prev_command_5, prev_command_4, prev_command_3, prev_command_2, prev_command_1, next_command[0]), 0) +
            lambdas[7] * model['ninegram_probs'].get((prev_command_8, prev_command_7, prev_command_6, prev_command_5, prev_command_4, prev_command_3, prev_command_2, prev_command_1, next_command[0]), 0) +
            lambdas[8] * model['tengram_probs'].get((prev_command_9, prev_command_8, prev_command_7, prev_command_6, prev_command_5, prev_command_4, prev_command_3, prev_command_2, prev_command_1, next_command[0]), 0) +
            lambdas[9] * model['context_bigram_prob'].get((context_prev_cmd1, context_next_command), 0) +
            lambdas[10] * model['context_trigram_prob'].get((context_prev_cmd2, context_prev_cmd1, context_next_command), 0) +
            lambdas[11] * model['context_4gram_prob'].get((context_prev_cmd3, context_prev_cmd2, context_prev_cmd1, context_next_command), 0) +
            lambdas[12] * model['context_5gram_prob'].get((context_prev_cmd4, context_prev_cmd3, context_prev_cmd2, context_prev_cmd1, context_next_command), 0) +
            lambdas[13] * model['context_6gram_prob'].get((context_prev_cmd5, context_prev_cmd4, context_prev_cmd3, context_prev_cmd2, context_prev_cmd1, context_next_command), 0) +
            lambdas[14] * model['context_7gram_prob'].get((context_prev_cmd6, context_prev_cmd5, context_prev_cmd4, context_prev_cmd3, context_prev_cmd2, context_prev_cmd1, context_next_command), 0) +
            lambdas[15] * model['context_8gram_prob'].get((context_prev_cmd7, context_prev_cmd6, context_prev_cmd5, context_prev_cmd4, context_prev_cmd3, context_prev_cmd2, context_prev_cmd1, context_next_command), 0) +
            lambdas[16] * model['context_9gram_prob'].get((context_prev_cmd8, context_prev_cmd7, context_prev_cmd6, context_prev_cmd5, context_prev_cmd4, context_prev_cmd3, context_prev_cmd2, context_prev_cmd1, context_next_command), 0) +
            lambdas[17] * model['context_10gram_prob'].get((context_prev_cmd9, context_prev_cmd8, context_prev_cmd7, context_prev_cmd6, context_prev_cmd5, context_prev_cmd4, context_prev_cmd3, context_prev_cmd2, context_prev_cmd1, context_next_command), 0)
        )
        next_command_probs[next_command[0]] = score
    
    prefix_probs = defaultdict(float)
    prefix_counts = defaultdict(int)
    for command, prob in next_command_probs.items():
        prefix = get_command_prefix(command)
        # if prefix_probs[prefix] < prob:
        #     prefix_probs[prefix] = prob
        prefix_probs[prefix] += prob
        prefix_counts[prefix] += 1

    for prefix in prefix_counts:
        prefix_probs[prefix] = prefix_probs[prefix] / prefix_counts[prefix]

    sorted_prefix_probs = sorted(prefix_probs.items(), key=lambda x: x[1], reverse=True)
    result = merge_similar_keys(sorted_prefix_probs)

    return result[:n]

def is_file(value):
    pattern = r'.+\.[A-Za-z]+\.?$'
    return bool(re.match(pattern, value))

def is_message(value):
    return value.startswith('"') and value.endswith('"')

def determine_type(value):
    # Check for URL
    if re.match(r'^https?://', value):
        return '<URL>'
    # Check for directory (simple heuristic for directory-like paths)
    elif re.match(r'^(?:/[^<>:"|?*]+)+/?$', value) or re.match(r'.*\/[^.]+[^.]*$', value) or re.search(r'/$', value):
        return '<directory>'
    # Check for number
    elif value.isdigit():
        return '<number>'
    elif is_file(value):
        return '<file>'
    elif is_message(value):
        return '<string>'
        # return  f'"{value_type}"' 
    else:
        return '<string>'

def load_command_patterns(file_path):
    patterns = []
    with open(file_path, 'r') as file:
        for line in file:
            patterns.append(line.strip().split())  # Tách từng từ trong mỗi dòng
    return patterns

def generalize_words(command, patterns):
    words = re.findall(r'\".*?\"|\'.*?\'|\S+', command)
   
    generalized_command = []  # Kết quả tổng quát hóa
    possible_patterns = patterns[:]  # Bắt đầu với tất cả các mẫu

    # Duyệt từng từ trong dòng lệnh
    for i, word in enumerate(words):
        if word in special_words:
            generalized_command.append(word)
            new_command = ' '.join(words[i+1:])
            generalized_command.append(generalize_words(new_command, patterns)) 
            break
        else:
        # Giữ lại các pattern khớp với từ hiện tại
            matching_patterns = [pattern for pattern in possible_patterns if i < len(pattern) and pattern[i] == word]
            # Nếu có matching pattern, tiếp tục với từ tiếp theo
            if matching_patterns:
                generalized_command.append(word)
                possible_patterns = matching_patterns  # Giới hạn các pattern theo từ hiện tại
            else:
                # Nếu không có pattern nào khớp, detect loại của từ (ví dụ <string>, <number>)
                generalized_command.append(determine_type(word))
                break
    
    return ' '.join(generalized_command)

#Build UI
def calculate_predictions():
    # Lấy lịch sử lệnh từ ô nhập
    history_input = history_text.get("1.0", tk.END).strip()
    history = history_input.split("\n")  # Tách các dòng lệnh
    generalized_commands = [generalize_words(line, patterns) for line in history]

    if not history:
        messagebox.showerror("Error", "Hãy nhập ít nhất một lệnh trong lịch sử!")
        return

    # Kết quả gợi ý (bạn thay bằng logic thực tế của mình)
    top_commands = calculate_next_command_probabilities(generalized_commands, model, lambdas, n=5)

    # Cập nhật danh sách gợi ý
    suggestion_list.delete(*suggestion_list.get_children())
    for i, command in enumerate(top_commands, start=1):
        suggestion_list.insert("", "end", values=(i, command))


# mainnnn

model = load_model('model_suggest_by_line.pkl')
lambdas = [1/18] * 18  # Trọng số cho các n-gram  # Lịch sử câu lệnh
patterns = load_command_patterns('removed_duplicate_command.txt')

# Tạo giao diện chính
root = tk.Tk()
root.title("Gợi ý câu lệnh Bash")
root.geometry("700x550")
root.configure(bg="#f7f7f7")  # Màu nền sáng

# Tiêu đề
title_label = ttk.Label(root, text="Nhập lịch sử lệnh (mỗi dòng là một lệnh):", font=("Helvetica", 16, "bold"), background="#f7f7f7")
title_label.pack(pady=10)

# Khung nhập lịch sử lệnh
history_frame = ttk.Frame(root, padding=(10, 10))
history_frame.pack(fill="x", padx=20, pady=10)

# history_label = ttk.Label(history_frame, text="Nhập lịch sử lệnh (mỗi dòng là một lệnh):", font=("Helvetica", 12))
# history_label.pack(anchor="w")

history_text = tk.Text(history_frame, height=8, wrap="word", font=("Consolas", 12))
history_text.pack(fill="x", pady=5)

# Nút tính toán
button_frame = ttk.Frame(root, padding=(10, 10))
button_frame.pack(fill="x", padx=20, pady=5)

calculate_button = ttk.Button(button_frame, text="Dự Đoán", command=calculate_predictions)
calculate_button.pack(side="right", padx=5, pady=5)

# calculate_button = tk.Button(root, text="Gợi ý", font=("Helvetica", 14), command=on_suggest_click, bg="#4CAF50", fg="white")
# calculate_button.pack(pady=10)


# Khung kết quả
result_frame = ttk.Frame(root, padding=(10, 10))
result_frame.pack(fill="both", expand=True, padx=20, pady=10)

result_label = ttk.Label(result_frame, text="Kết quả gợi ý:", font=("Helvetica", 12))
result_label.pack(anchor="w")

columns = ("#1", "#2")
suggestion_list = ttk.Treeview(result_frame, columns=columns, show="headings", height=10)

# Định nghĩa tiêu đề các cột
suggestion_list.heading("#1", text="STT")
suggestion_list.heading("#2", text="Lệnh Gợi Ý")

# Định nghĩa kích thước các cột
suggestion_list.column("#1", width=50, anchor="center")
suggestion_list.column("#2", width=550, anchor="center")

suggestion_list.pack(fill="both", expand=True, pady=5)

# Thanh cuộn cho Treeview
scrollbar = ttk.Scrollbar(result_frame, orient="vertical", command=suggestion_list.yview)
suggestion_list.configure(yscrollcommand=scrollbar.set)
scrollbar.pack(side="right", fill="y")

# Chạy ứng dụng
root.mainloop()


# history = ['systemctl stop docker',
#  'systemctl stop kubelet',
#  'systemctl start docker',
#     'systemctl start kubelet',
#     'ls',
#     'ls',
#     'cd <directory>',
#     'ls',
#     'cd <directory> '
#  ]

# patterns = load_command_patterns('removed_duplicate_command.txt')

# generalized_commands = [generalize_words(line, patterns) for line in history]

# top_commands = calculate_next_command_probabilities(history, model, lambdas, n=5)
# print("Top câu lệnh tiếp theo:")
# for command, prob in top_commands:
#     print(f"Câu lệnh: {command}, Xác suất: {prob}")

