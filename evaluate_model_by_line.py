from collections import defaultdict
import re
import math
import time
import csv
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
import pickle


# Định nghĩa các ký tự đặc biệt để loại bỏ
special_patterns = ['<string>', '<file>', '<directory>', '<number>', '<URL>']
skip_prefixes = {"exit", "clear", "ssh"}

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
        # if command == 'git':
        #     print(prob)
        #     print(merged_commands)
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
    for key, value in merged_commands.items():
        average_prob = value['total_prob'] / value['count']  # Tính trung bình xác suất
        result.append((key, average_prob))
    
    result = sorted(result, key=lambda x: x[1], reverse=True)
    
    return result


def get_command_prefix(command): 
    parts = command.split()
    prefix_parts = []
    for part in parts:
        if part.startswith('-') or (part in special_patterns) or not re.search(r'[a-zA-Z]', part) or not part.isalpha():
            break
        prefix_parts.append(part)

    return ' '.join(prefix_parts)

def load_cleaned_commands(file_path):
    cleaned_commands = []
    with open(file_path, encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            # if line.startswith('#') or not line:  # Bỏ qua các dòng bắt đầu bằng '#'
            #     continue
            # new_line = get_command_prefix(line)
            cleaned_commands.append(line)
            if line.startswith('#'):
                cleaned_commands.append('null')
                cleaned_commands.append('null')
                cleaned_commands.append('null')
                cleaned_commands.append('null')
                cleaned_commands.append('null')
                cleaned_commands.append('null')
                cleaned_commands.append('null')
                cleaned_commands.append('null')
                cleaned_commands.append('null')

            # prefix = get_prefix(line)  # Lấy phần prefix trước ký tự đặc biệt
            # if prefix:  # Nếu vẫn còn nội dung, thêm vào danh sách
                # cleaned_commands.append(prefix)
    return cleaned_commands

    
# Cal xác suất 
def calculate_next_command_probabilities(history, unigrams_prob, bigrams_prob, trigrams_prob, fourgrams_prob, fivegram_prob, sixgram_prob, sevengram_prob, eightgram_prob, ninegram_prob, tengram_prob, context_bigram_prob, context_trigram_prob, context_4gram_prob, context_5gram_prob, context_6gram_prob, context_7gram_prob, context_8gram_prob, context_9gram_prob, context_10gram_prob, lambdas, n = 5):
    prev_command_1 = history[-1] if len(history) > 0 else ""
    prev_command_2 = history[-2] if len(history) > 1 else ""
    prev_command_3 = history[-3] if len(history) > 2 else ""
    prev_command_4 = history[-4] if len(history) > 3 else ""
    prev_command_5 = history[-5] if len(history) > 4 else ""
    prev_command_6 = history[-6] if len(history) > 5 else ""
    prev_command_7 = history[-7] if len(history) > 6 else ""
    prev_command_8 = history[-8] if len(history) > 7 else ""
    prev_command_9 = history[-9] if len(history) > 8 else ""
    
    
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
    
    for next_command in unigrams_prob.keys():
        # score = (
        #     lambdas[0] * unigrams_prob.get(next_command, 0) +
        #     lambdas[1] * bigrams_prob.get((prev_command_1, next_command[0]), 0) +
        #     lambdas[2] * trigrams_prob.get((prev_command_2, prev_command_1, next_command[0]), 0)
        # )
        next_command_cleaned = get_command_prefix(next_command[0])
        context_next_command = " ".join(next_command_cleaned.split()[:1])
        if context_bigram_prob.get((context_prev_cmd1, context_next_command), 0) == 0 and context_trigram_prob.get((context_prev_cmd2, context_prev_cmd1, context_next_command), 0) == 0:
            continue

        if next_command[0] == 'ls':
            print(bigram_probs.get((prev_command_1, next_command[0]), 0))
            print(trigrams_prob.get((prev_command_2, prev_command_1, next_command[0]), 0))
            print(fourgrams_prob.get((prev_command_3, prev_command_2, prev_command_1, next_command[0]), 0))
            print(fivegram_prob.get((prev_command_4, prev_command_3, prev_command_2, prev_command_1, next_command[0]), 0))
            print(sixgram_prob.get((prev_command_5, prev_command_4, prev_command_3, prev_command_2, prev_command_1, next_command[0]), 0))
            print(sevengram_prob.get((prev_command_6, prev_command_5, prev_command_4, prev_command_3, prev_command_2, prev_command_1, next_command[0]), 0))
            print(eightgram_prob.get((prev_command_7, prev_command_6, prev_command_5, prev_command_4, prev_command_3, prev_command_2, prev_command_1, next_command[0]), 0))
            print(ninegram_prob.get((prev_command_8, prev_command_7, prev_command_6, prev_command_5, prev_command_4, prev_command_3, prev_command_2, prev_command_1, next_command[0]), 0))
            print(tengram_prob.get((prev_command_9, prev_command_8, prev_command_7, prev_command_6, prev_command_5, prev_command_4, prev_command_3, prev_command_2, prev_command_1, next_command[0]), 0))
            print('-----------------')
        score = (
            lambdas[0] * bigram_probs.get((prev_command_1, next_command[0]), 0) +
            lambdas[1] * trigrams_prob.get((prev_command_2, prev_command_1, next_command[0]), 0) +
            lambdas[2] * fourgrams_prob.get((prev_command_3, prev_command_2, prev_command_1, next_command[0]), 0) +
            lambdas[3] * fivegram_prob.get((prev_command_4, prev_command_3, prev_command_2, prev_command_1, next_command[0]), 0) +
            lambdas[4] * sixgram_prob.get((prev_command_5, prev_command_4, prev_command_3, prev_command_2, prev_command_1, next_command[0]), 0) +
            lambdas[5] * sevengram_prob.get((prev_command_6, prev_command_5, prev_command_4, prev_command_3, prev_command_2, prev_command_1, next_command[0]), 0) +
            lambdas[6] * eightgram_prob.get((prev_command_7, prev_command_6, prev_command_5, prev_command_4, prev_command_3, prev_command_2, prev_command_1, next_command[0]), 0) +
            lambdas[7] * ninegram_prob.get((prev_command_8, prev_command_7, prev_command_6, prev_command_5, prev_command_4, prev_command_3, prev_command_2, prev_command_1, next_command[0]), 0) +
            lambdas[8] * tengram_prob.get((prev_command_9, prev_command_8, prev_command_7, prev_command_6, prev_command_5, prev_command_4, prev_command_3, prev_command_2, prev_command_1, next_command[0]), 0) +
            lambdas[9] * context_bigram_prob.get((context_prev_cmd1, context_next_command), 0) +
            lambdas[10] * context_trigram_prob.get((context_prev_cmd2, context_prev_cmd1, context_next_command), 0) +
            lambdas[11] * context_4gram_prob.get((context_prev_cmd3, context_prev_cmd2, context_prev_cmd1, context_next_command), 0) +
            lambdas[12] * context_5gram_prob.get((context_prev_cmd4, context_prev_cmd3, context_prev_cmd2, context_prev_cmd1, context_next_command), 0) +
            lambdas[13] * context_6gram_prob.get((context_prev_cmd5, context_prev_cmd4, context_prev_cmd3, context_prev_cmd2, context_prev_cmd1, context_next_command), 0) +
            lambdas[14] * context_7gram_prob.get((context_prev_cmd6, context_prev_cmd5, context_prev_cmd4, context_prev_cmd3, context_prev_cmd2, context_prev_cmd1, context_next_command), 0) +
            lambdas[15] * context_8gram_prob.get((context_prev_cmd7, context_prev_cmd6, context_prev_cmd5, context_prev_cmd4, context_prev_cmd3, context_prev_cmd2, context_prev_cmd1, context_next_command), 0) +
            lambdas[16] * context_9gram_prob.get((context_prev_cmd8, context_prev_cmd7, context_prev_cmd6, context_prev_cmd5, context_prev_cmd4, context_prev_cmd3, context_prev_cmd2, context_prev_cmd1, context_next_command), 0) +
            lambdas[17] * context_10gram_prob.get((context_prev_cmd9, context_prev_cmd8, context_prev_cmd7, context_prev_cmd6, context_prev_cmd5, context_prev_cmd4, context_prev_cmd3, context_prev_cmd2, context_prev_cmd1, context_next_command), 0)
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
        # print(prefix_counts[prefix])
        prefix_probs[prefix] = prefix_probs[prefix] / prefix_counts[prefix]

    sorted_prefix_probs = sorted(prefix_probs.items(), key=lambda x: x[1], reverse=True)
    result = merge_similar_keys(sorted_prefix_probs)

    return result[:n]
    # return sorted(next_command_probs.items(), key=lambda x: x[1], reverse=True)[:5]

def modified_accuracy_at_n(predictions, actual, n):
    """Calculate modified accuracy based on whether actual starts with any of the top n predictions."""
    
    # Kiểm tra xem liệu actual có bắt đầu với bất kỳ gợi ý nào trong top n không
    return 1 if any(
        pred and isinstance(pred, str) and actual.startswith(pred)  # Kiểm tra phần đầu của actual có trùng với từ đầu tiên của pred
        for pred, _ in predictions[:n]
    ) else 0



# Evaluation function with both accuracy metrics and logging in text files
def evaluate_model_with_text_logs_and_both_accuracies(test_commands, model, lambdas, success_log_path, fail_log_path):
    total_time = 0.0
    modified_accuracy_top_1_total = 0
    modified_accuracy_top_3_total = 0
    modified_accuracy_top_5_total = 0
    accuracy_total = 0
    N = len(test_commands) - 9
    print(N)
    test_cases_done = set()

    with open(success_log_path, 'w', encoding='utf-8') as success_file, \
         open(fail_log_path, 'w', encoding='utf-8') as fail_file:

        for i in range(N):
            context = test_commands[i:i + 9]
            actual_next_command = test_commands[i + 9]

            if (any(cmd.startswith("#") or not get_command_prefix(cmd).strip() or cmd.split()[0] in skip_prefixes for cmd in context) 
                or actual_next_command.startswith("#") or not get_command_prefix(actual_next_command).strip() or actual_next_command.split()[0] in skip_prefixes):
                continue
            test_case = (tuple(context), actual_next_command)
            if test_case in test_cases_done:
                continue

            test_cases_done.add(test_case)

            start_time = time.time()
            # Get the top predictions for the current command
            top_commands = calculate_next_command_probabilities(context, model, lambdas, n=5)
            end_time = time.time()

            total_time += (end_time - start_time)
            is_correct_top_1 = modified_accuracy_at_n(top_commands, actual_next_command, 1)
            is_correct_top_3 = modified_accuracy_at_n(top_commands, actual_next_command, 3)
            is_correct_top_5 = modified_accuracy_at_n(top_commands, actual_next_command, 5)
            modified_accuracy_top_1_total += is_correct_top_1
            modified_accuracy_top_3_total += is_correct_top_3
            modified_accuracy_top_5_total += is_correct_top_5

            # Format top predictions for logging
            top_commands_text = "; ".join([f"{cmd} (Prob: {prob:.4f})" for cmd, prob in top_commands])
            result_text = (
                f"Prev Command 9: {context[0]}\n"
                f"Prev Command 8: {context[1]}\n"
                f"Prev Command 7: {context[2]}\n"
                f"Prev Command 6: {context[3]}\n"
                f"Prev Command 5: {context[4]}\n"
                f"Prev Command 4: {context[5]}\n"
                f"Prev Command 3: {context[6]}\n"
                f"Prev Command 2: {context[7]}\n"
                f"Prev Command 1: {context[8]}\n"
                f"Actual Next Command: {actual_next_command}\n"
                f"Top 5 Predictions: {top_commands_text}\n"
                f"In top-1: {'Yes' if is_correct_top_1 else 'No'}\n"
                f"In top-3: {'Yes' if is_correct_top_3 else 'No'}\n"
                "-------------------------\n"
            )

            # Write to the appropriate file based on the original accuracy
            if is_correct_top_5 == 1:
                success_file.write(result_text)
            else:
                fail_file.write(result_text)

    average_time = total_time / len(test_cases_done)
    modified_accuracy_top_1_score = modified_accuracy_top_1_total / len(test_cases_done)
    modified_accuracy_top_3_score = modified_accuracy_top_3_total / len(test_cases_done)
    modified_accuracy_top_5_score = modified_accuracy_top_5_total / len(test_cases_done)
    
    return average_time, modified_accuracy_top_1_score,modified_accuracy_top_3_score, modified_accuracy_top_5_score

# mainnnnnnnnn
model = load_model('ngram_model9.pkl')

lambdas = [1/18, 1/18, 1/18, 1/18, 1/18, 1/18, 1/18, 1/18, 1/18, 1/18, 1/18, 1/18, 1/18, 1/18, 1/18, 1/18, 1/18, 1/18]
# lambdas = [0.195, 0.5306, 0.152, 0.1224]

success_log_path = 'model_success_cases9.txt'
fail_log_path = 'model_fail_cases9.txt'

commands = load_cleaned_commands('shuffled_block.txt')

train_commands, test_commands = train_test_split(commands, test_size=0.2, shuffle=False)

average_time, modified_accuracy_top_1_score, modified_accuracy_top_3_score, modified_accuracy_top_5_score = evaluate_model_with_text_logs_and_both_accuracies(
    test_commands, 
    model,
    success_log_path, 
    fail_log_path
)

print(f"Run Time {average_time:.4f}")
print(f"Modified Accuracy Top-1: {modified_accuracy_top_1_score:.4f}")
print(f"Modified Accuracy Top-3: {modified_accuracy_top_3_score:.4f}")
print(f"Modified Accuracy Top-5: {modified_accuracy_top_5_score:.4f}")
