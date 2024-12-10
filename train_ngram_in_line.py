import re
import time
from collections import Counter
import pickle

def load_data(filename):
    """Loads command data from a file and splits it into lines of words."""
    with open(filename, 'r') as file:
        lines = file.readlines()
    result = []
    for line in lines:
        # line = line.strip() + ' </s>'
        result.append(line.strip())
    return result

def generate_ngrams(data, n):
    """Generate n-grams from a list of data."""

    ngrams = []
    for line in data:
        line = line.strip() + ' </s>'
        words = line.split()
        words = [""] * (n-1) + words
        for i in range(len(words) - n + 1):
            ngram = tuple(words[i:i + n])  # Tạo n-gram
            ngrams.append(ngram)
    return ngrams
        
def calculate_unigram_probabilities(ngram_count_1, total_words):
    """Tính xác suất của 1-gram từ tần suất 1-gram và tổng số từ."""
    total_1gram_count = sum(ngram_count_1.values())  # Tổng số lần xuất hiện của tất cả các 1-gram
    probabilities = {gram: count / total_1gram_count for gram, count in ngram_count_1.items()}
    return probabilities

def calculate_ngram_probabilities(ngram_count, prev_ngram_count):
    """Tính xác suất của n-gram từ tần suất n-gram và n-1-gram."""
    probabilities = {}
    
    for ngram, count in ngram_count.items():
        # Lấy (n-1)-gram đầu tiên trong n-gram
        prev_ngram = ngram[:-1]
        # Xác suất của n-gram = tần suất của n-gram / tần suất của (n-1)-gram
        prev_ngram_count_value = prev_ngram_count.get(prev_ngram, 0)
        if prev_ngram_count_value > 0:
            prob = count / prev_ngram_count_value
        else:
            prob = 0  # Nếu (n-1)-gram không xuất hiện, xác suất bằng 0
            
        probabilities[ngram] = prob
    
    return probabilities


def save_model(model, filename):
    with open(filename, 'wb') as file:
        pickle.dump(model, file)
    print(f"Model saved to {filename}")


# mainnnnnnn
train_data = load_data("data_suggest_word.txt")


ngrams_1 = generate_ngrams(train_data, 1)
ngrams_2 = generate_ngrams(train_data, 2)
ngrams_3 = generate_ngrams(train_data, 3)
ngrams_4 = generate_ngrams(train_data, 4)

ngram_count_1 = Counter(ngrams_1)
ngram_count_2 = Counter(ngrams_2)
ngram_count_3 = Counter(ngrams_3)
ngram_count_4 = Counter(ngrams_4)

total_words = sum(ngram_count_1.values())
# print("Total words:", total_words)

unigram_prob = calculate_unigram_probabilities(ngram_count_1, total_words)
bigram_prob = calculate_ngram_probabilities(ngram_count_2, ngram_count_1)
trigram_prob = calculate_ngram_probabilities(ngram_count_3, ngram_count_2)
fourgram_prob = calculate_ngram_probabilities(ngram_count_4, ngram_count_3)

model = {
    'unigram': unigram_prob,
    'bigram': bigram_prob,
    'trigram': trigram_prob,
    'fourgram': fourgram_prob
}

save_model(model, "model_suggest_in_line.pkl")

lambdas = [0.0145, 0.0551, 0.2664, 0.664]

