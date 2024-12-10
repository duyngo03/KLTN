import re
import time
from collections import Counter
import pickle

def load_model(filename):
    """Load the trained model from a file."""
    with open(filename, 'rb') as file:
        model = pickle.load(file)
    print(f"Model loaded from {filename}")
    return model

def load_data(filename):
    """Loads command data from a file and splits it into lines of words."""
    with open(filename, 'r') as file:
        lines = file.readlines()
    result = []
    for line in lines:
        # line = line.strip() + ' </s>'
        result.append(line.strip())
    return result

        

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

    top_suggestions = sorted(probabilities.items(), key=lambda item: item[1], reverse=True)[:top_n]

    return top_suggestions


def evaluate_model(model, test_data, lambdas, correct_log_file="correct_cases_suggest_word.txt", incorrect_log_file="failed_cases_suggest_word.txt", top_n=5):
    """
    Đánh giá mô hình với dữ liệu kiểm tra và tính toán các chỉ số độ chính xác như Top 1, Top 3, và Top 5.
    
    model: Một dictionary chứa các xác suất n-gram.
    test_data: Dữ liệu kiểm tra, một danh sách các câu văn bản.
    correct_log_file: Đường dẫn đến tệp lưu các trường hợp dự đoán đúng.
    incorrect_log_file: Đường dẫn đến tệp lưu các trường hợp dự đoán sai.
    top_n: Số lượng từ gợi ý muốn lấy (Top N).
    """
    correct_top_5 = 0
    correct_top_1 = 0
    correct_top_3 = 0 
    total_predictions = 0
    correct_cases = []
    missed_cases = []
    total_time = 0.0

    with open(correct_log_file, 'w', encoding='utf-8') as correct_file, \
            open(incorrect_log_file, 'w', encoding='utf-8') as incorrect_file:

        for line in test_data:
            words = line.split()
            is_correct_3 = False
            is_correct_1 = False
            if len(words) < 2:
                input_words = words
                input_words = ' '.join(input_words)
                true_next_word = "</s>"
                # continue  # Cần ít nhất 2 từ để dự đoán từ tiếp theo
            else:
                input_words = words[:-1]  # Tất cả các từ ngoại trừ từ cuối
                input_words = ' '.join(input_words)
                true_next_word = words[-1]  # Từ cuối cùng là từ cần dự đoán

            # Lấy top N từ gợi ý
            start_time = time.time()
            suggestions = suggest_next_word(model, lambdas, input_words, top_n=top_n)
            end_time = time.time()
            suggested_words = [word for word, prob in suggestions]
            total_time += (end_time - start_time)
            
            # Kiểm tra nếu từ đúng có trong top N gợi ý
            if suggested_words and suggested_words[0] == true_next_word:
                correct_top_1 += 1
                is_correct_1 = True

            if true_next_word in suggested_words[:3]:
                correct_top_3 += 1
                is_correct_3 = True 
                
            # Kiểm tra nếu từ đúng có trong top N gợi ý
            if true_next_word in suggested_words:
                correct_top_5 += 1
                correct_cases.append((input_words, true_next_word, suggested_words, is_correct_1, is_correct_3))
            else:
                missed_cases.append((input_words, true_next_word, suggested_words, is_correct_1, is_correct_3))
            
            total_predictions += 1
        
        # Lưu các trường hợp đúng vào file
        for prefix, actual_next_word, suggested_words, is_correct_1, is_correct_3 in correct_cases:
            correct_file.write(f"Câu lệnh: {prefix}\n")
            correct_file.write(f"Từ thực tế: {actual_next_word}\n")
            correct_file.write(f"Top {top_n} từ dự đoán: {suggested_words}\n")
            correct_file.write(f"Đúng top 1: {is_correct_1}\n")
            correct_file.write(f"Đúng top 3: {is_correct_3}\n")
            correct_file.write("\n")

        # Lưu các trường hợp sai vào file
        for prefix, actual_next_word, suggested_words, is_correct_1, is_correct_3 in missed_cases:
            incorrect_file.write(f"Câu lệnh: {prefix}\n")
            incorrect_file.write(f"Từ thực tế: {actual_next_word}\n")
            incorrect_file.write(f"Top {top_n} từ dự đoán: {suggested_words}\n")
            incorrect_file.write("\n")

    # Tính toán các độ chính xác
    average_time = total_time / total_predictions if total_predictions > 0 else 0
    accuracy_top_5 = correct_top_5 / total_predictions if total_predictions > 0 else 0
    accuracy_top_3 = correct_top_3 / total_predictions if total_predictions > 0 else 0
    accuracy_top_1 = correct_top_1 / total_predictions if total_predictions > 0 else 0
    
    return accuracy_top_5, accuracy_top_1, accuracy_top_3, average_time


# mainnnnnnn

model = load_model("model_suggest_in_line.pkl")

test_data = load_data("test.txt")

lambdas = [0.0145, 0.0551, 0.2664, 0.664]

accuracy_top_5, accuracy_top_1, accuracy_top_3, average_time = evaluate_model(model, test_data, lambdas, top_n=5)


print(f"Run Time {average_time:.4f}")
print(f"Modified Accuracy Top-1: {accuracy_top_1:.4f}")
print(f"Modified Accuracy Top-3: {accuracy_top_3:.4f}")
print(f"Modified Accuracy Top-5: {accuracy_top_5:.4f}")
