from collections import defaultdict
import re
import math
import time
import csv
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
import pickle

special_patterns = ['<string>', '<file>', '<directory>', '<number>', '<URL>']
skip_prefixes = {"exit", "clear", "ssh"}

def save_model_to_file(model, filename):
    with open(filename, 'wb') as file:
        pickle.dump(model, file)

def load_cleaned_commands(file_path):
    cleaned_commands = []
    with open(file_path, encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
           
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

    return cleaned_commands

def get_command_prefix(command): 
    parts = command.split()
    prefix_parts = []
    for part in parts:
        if part.startswith('-') or (part in special_patterns) or not re.search(r'[a-zA-Z]', part) or not part.isalpha():
            break
        prefix_parts.append(part)

    return ' '.join(prefix_parts)

# Hàm xây dựng unigrams và bigrams, trigrams từ các dòng lệnh liên tiếp trong dữ liệu huấn luyện
def build_line_ngrams(data, n):
    ngram_counts = defaultdict(int)
    
    for i in range(len(data) - n + 1):
        ngram = tuple(data[i:i + n])
        if any(
            not get_command_prefix(cmd).strip() or
            cmd.startswith("#") or
            cmd.split()[0] in skip_prefixes 
            for cmd in ngram
            ):
            continue
        if n == 2 and ngram[0] == 'null' and ngram[1] == 'null':
            continue
        if n == 3 and ngram[0] == 'null' and ngram[1] == 'null' and ngram[2] == 'null':
            continue
        if n == 4 and ngram[0] == 'null' and ngram[1] == 'null' and ngram[2] == 'null' and ngram[3] == 'null':
            continue
        if n == 5 and ngram[0] == 'null' and ngram[1] == 'null' and ngram[2] == 'null' and ngram[3] == 'null' and ngram[4] == 'null':
            continue
        if n == 6 and ngram[0] == 'null' and ngram[1] == 'null' and ngram[2] == 'null' and ngram[3] == 'null' and ngram[4] == 'null' and ngram[5] == 'null':
            continue
        if n == 7 and ngram[0] == 'null' and ngram[1] == 'null' and ngram[2] == 'null' and ngram[3] == 'null' and ngram[4] == 'null' and ngram[5] == 'null' and ngram[6] == 'null':
            continue
        if n == 8 and ngram[0] == 'null' and ngram[1] == 'null' and ngram[2] == 'null' and ngram[3] == 'null' and ngram[4] == 'null' and ngram[5] == 'null' and ngram[6] == 'null' and ngram[7] == 'null':
            continue
        if n == 9 and ngram[0] == 'null' and ngram[1] == 'null' and ngram[2] == 'null' and ngram[3] == 'null' and ngram[4] == 'null' and ngram[5] == 'null' and ngram[6] == 'null' and ngram[7] == 'null' and ngram[8] == 'null':
            continue
        ngram_counts[ngram] += 1
    return ngram_counts

def build_context_ngram(data, n):
    ngram_counts = defaultdict(int)
    
    # Duyệt qua các chuỗi con có độ dài n
    for i in range(len(data) - n + 1):
        # Lấy các tiền tố của các phần tử trong context
        context = [get_command_prefix(data[i + j]) for j in range(n)]
        
        # Kiểm tra các tiền tố và bỏ qua nếu không hợp lệ
        if any(not cmd for cmd in context) or all(cmd == 'null' for cmd in context):
            continue
        
        # Chỉ lấy phần đầu của mỗi tiền tố để tạo context
        context_words = tuple(" ".join(cmd.split()[:1]) for cmd in context)
        
        # Tăng đếm cho context tương ứng
        ngram_counts[context_words] += 1

    return ngram_counts


# Chuyển n-gram sang xác suất
def convert_ngram_prob(ngram_counts, n=1, previous_counts=None):
    total_count = sum(ngram_counts.values())
    ngram_probs = {}
    for ngram, count in ngram_counts.items():
        if n == 1:
            # Xác suất của unigram là số lần xuất hiện của từ chia cho tổng số từ
             ngram_probs[ngram] = count / total_count
        else:
            # Xác suất của bigram/trigram là số lần xuất hiện của cặp từ chia cho số lần xuất hiện của phần tử trước
            prev_ngram = ngram[:-1]  # Lấy phần đầu của n-gram (ví dụ: cho bigram, lấy unigram trước đó)
            prev_count = previous_counts.get(prev_ngram, 0)  # Lấy số lần xuất hiện của phần trước ngram
            
            if prev_count > 0:
                ngram_probs[ngram] = count / prev_count
            else:
                ngram_probs[ngram] = 0
    return ngram_probs
    # return {key: count / total_count for key, count in ngram_counts.items()}


def convert_context_prob(context_counts, n):
    prefix_counts = defaultdict(int)
    for ngram, count in context_counts.items():
        prefix = ngram[:n-1]
        prefix_counts[prefix] += count
    context_ngram_probs = {}
    for ngram, count in context_counts.items():
        prefix = ngram[:n-1]
        context_ngram_probs[ngram] = count / prefix_counts[prefix]
    return context_ngram_probs
    

def modified_accuracy_at_n(predictions, actual, n):
    """Calculate modified accuracy based on whether actual starts with any of the top n predictions."""
    
    # Kiểm tra xem liệu actual có bắt đầu với bất kỳ gợi ý nào trong top n không
    return 1 if any(
        pred and isinstance(pred, str) and actual.startswith(pred)  # Kiểm tra phần đầu của actual có trùng với từ đầu tiên của pred
        for pred, _ in predictions[:n]
    ) else 0


commands = load_cleaned_commands('data_suggest_next_cmd.txt')

train_commands = commands
# train_commands, test_commands = train_test_split(commands, test_size=0.2, shuffle=False)

unigrams = build_line_ngrams(train_commands, 1)
bigrams = build_line_ngrams(train_commands, 2)
trigrams = build_line_ngrams(train_commands, 3)
fourgrams = build_line_ngrams(train_commands, 4)
fivegrams = build_line_ngrams(train_commands, 5)
sixgrams = build_line_ngrams(train_commands, 6)
sevengrams = build_line_ngrams(train_commands, 7)
eightgrams = build_line_ngrams(train_commands, 8)
ninegrams = build_line_ngrams(train_commands, 9)
tengrams = build_line_ngrams(train_commands, 10)

context_bigram = build_context_ngram(train_commands,2)
context_trigram = build_context_ngram(train_commands,3)
context_4gram = build_context_ngram(train_commands,4)
context_5gram = build_context_ngram(train_commands,5)
context_6gram = build_context_ngram(train_commands,6)
context_7gram = build_context_ngram(train_commands,7)
context_8gram = build_context_ngram(train_commands,8)
context_9gram = build_context_ngram(train_commands,9)
context_10gram = build_context_ngram(train_commands,10)

#Convert probabilities
unigram_probs = convert_ngram_prob(unigrams)
bigram_probs = convert_ngram_prob(bigrams, 2, unigrams)
trigram_probs = convert_ngram_prob(trigrams, 3, bigrams)
fourgram_probs = convert_ngram_prob(fourgrams, 4, trigrams) 
fivegram_probs = convert_ngram_prob(fivegrams, 5, fourgrams)
sixgram_probs = convert_ngram_prob(sixgrams, 6, fivegrams)
seven_probs = convert_ngram_prob(sevengrams, 7, sixgrams)
eight_probs = convert_ngram_prob(eightgrams, 8, sevengrams)
nine_probs = convert_ngram_prob(ninegrams, 9, eightgrams)
ten_probs = convert_ngram_prob(tengrams, 10, ninegrams)

context_bigram_prob = convert_context_prob(context_bigram, 2)
context_trigram_prob = convert_context_prob(context_trigram, 3)
context_4gram_prob = convert_context_prob(context_4gram, 4)
context_5gram_prob = convert_context_prob(context_5gram, 5)
context_6gram_prob = convert_context_prob(context_6gram, 6)
context_7gram_prob = convert_context_prob(context_7gram, 7)
context_8gram_prob = convert_context_prob(context_8gram, 8)
context_9gram_prob = convert_context_prob(context_9gram, 9)
context_10gram_prob = convert_context_prob(context_10gram, 10)


model = {
    'unigram_probs': unigram_probs,
    'bigram_probs': bigram_probs,
    'trigram_probs': trigram_probs,
    'fourgram_probs': fourgram_probs,
    'fivegram_probs': fivegram_probs,
    'sixgram_probs': sixgram_probs,
    'sevengram_probs': sevengrams,
    'eightgram_probs': eightgrams,
    'ninegram_probs': ninegrams,
    'tengram_probs': tengrams,
    'context_bigram_prob': context_bigram_prob,
    'context_trigram_prob': context_trigram_prob,
    'context_4gram_prob': context_4gram_prob,
    'context_5gram_prob': context_5gram_prob,
    'context_6gram_prob': context_6gram_prob,
    'context_7gram_prob': context_7gram_prob,
    'context_8gram_prob': context_8gram_prob,
    'context_9gram_prob': context_9gram_prob,
    'context_10gram_prob': context_10gram_prob
}

save_model_to_file(model, 'model_suggest_by_line.pkl')

# lambdas = [1/18, 1/18, 1/18, 1/18, 1/18, 1/18, 1/18, 1/18, 1/18, 1/18, 1/18, 1/18, 1/18, 1/18, 1/18, 1/18, 1/18, 1/18]


