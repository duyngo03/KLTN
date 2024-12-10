import numpy as np
import random
from collections import defaultdict, Counter

# Step 1: Data Preparation
def read_and_split_data(file_path, split_ratio=0.8):
    """Reads data from a file and splits it into training and held-out data."""
    with open(file_path, 'r') as file:
        lines = file.readlines()
    result = []
    # for line in lines:
    #     result.append(line.strip() + ' </s>')
    random.shuffle(lines)
    split_index = int(len(lines) * split_ratio)
    training_data = lines[:split_index]
    held_out_data = lines[split_index:]
    return training_data, held_out_data

# Step 2: NGramModel for Unigram, Bigram, and Trigram probabilities
class NGramModel:
    def __init__(self, training_data):
        self.unigram_counts = Counter()
        self.bigram_counts = defaultdict(Counter)
        self.trigram_counts = defaultdict(lambda: defaultdict(Counter))
        self.fourgram_counts = defaultdict(lambda: defaultdict(lambda: defaultdict(Counter)))
        self.fivegram_counts = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(Counter))))
        self.total_unigrams = 0

        self._train(training_data)

    def _train(self, data):
        """Processes the training data to populate n-gram counts."""
        for line in data:
            words = line.strip() + ' </s>'
            words = words.split()
            
            # words = ['<s>'] + line.strip().split() + ['</s>']
            self.total_unigrams += len(words)
            
            for i in range(len(words)):
                self.unigram_counts[words[i]] += 1
                prev4 = words[i-4] if i > 3 else ""
                prev3 = words[i-3] if i > 2 else ""
                prev2 = words[i-2] if i > 1 else ""
                prev1 = words[i-1] if i > 0 else ""
                self.bigram_counts[prev1][words[i]] += 1
                self.trigram_counts[prev2][prev1][words[i]] += 1
                self.fourgram_counts[prev3][prev2][prev1][words[i]] += 1
                self.fivegram_counts[prev4][prev3][prev2][prev1][words[i]] += 1

    def unigram_prob(self, w):
        return self.unigram_counts[w] / self.total_unigrams if self.total_unigrams > 0 else 0

    def bigram_prob(self, w1, w2):
        bigram_total = sum(self.bigram_counts[w1].values())
        return self.bigram_counts[w1][w2] / bigram_total if bigram_total > 0 else 0

    def trigram_prob(self, w1, w2, w3):
        trigram_total = sum(self.trigram_counts[w1][w2].values())
        return self.trigram_counts[w1][w2][w3] / trigram_total if trigram_total > 0 else 0

    def fourgram_prob(self, w1, w2, w3, w4):
        fourgram_total = sum(self.fourgram_counts[w1][w2][w3].values())
        return self.fourgram_counts[w1][w2][w3][w4] / fourgram_total if fourgram_total > 0 else 0

    def fivegram_prob(self, w1, w2, w3, w4, w5):
        fivegram_total = sum(self.fivegram_counts[w1][w2][w3][w4].values())
        return self.fivegram_counts[w1][w2][w3][w4][w5] / fivegram_total if fivegram_total > 0 else 0
# Step 3: Define the EMAlgorithm class to find optimal lambdas for each case
class EMAlgorithm:
    def __init__(self, model, held_out_data, max_iter=50, tol=1e-4):
        self.model = model
        self.held_out_data = held_out_data
        self.max_iter = max_iter
        self.tol = tol

    def e_step_bigram(self, prev2, prev1):
        """Compute responsibilities for each lambda in the bigram case."""
        unigram_p = self.model.unigram_prob(prev1)
        bigram_p = self.model.bigram_prob(prev2, prev1)

        probs = self.lambdas_bigram * np.array([unigram_p, bigram_p])
        total_prob = np.sum(probs)
        
        return probs / total_prob if total_prob > 0 else np.zeros(2)

    def e_step_trigram(self, prev3, prev2, prev1):
        """Compute responsibilities for each lambda in the trigram case."""
        unigram_p = self.model.unigram_prob(prev1)
        bigram_p = self.model.bigram_prob(prev2, prev1)
        trigram_p = self.model.trigram_prob(prev3, prev2, prev1)

        probs = self.lambdas_trigram * np.array([unigram_p, bigram_p, trigram_p])
        total_prob = np.sum(probs)
        
        return probs / total_prob if total_prob > 0 else np.zeros(3)

    def e_step_fourgram(self, prev4, prev3, prev2, prev1):
        """Compute responsibilities for each lambda in the fourgram case."""
        unigram_p = self.model.unigram_prob(prev1)
        bigram_p = self.model.bigram_prob(prev2, prev1)
        trigram_p = self.model.trigram_prob(prev3, prev2, prev1)
        fourgram_p = self.model.fourgram_prob(prev4, prev3, prev2, prev1)

        probs = self.lambdas_fourgram * np.array([unigram_p, bigram_p, trigram_p, fourgram_p])
        total_prob = np.sum(probs)
        
        return probs / total_prob if total_prob > 0 else np.zeros(4)

    def e_step_fivegram(self, prev5, prev4, prev3, prev2, prev1):
        """Compute responsibilities for each lambda in the fivegram case."""
        unigram_p = self.model.unigram_prob(prev1)
        bigram_p = self.model.bigram_prob(prev2, prev1)
        trigram_p = self.model.trigram_prob(prev3, prev2, prev1)
        fourgram_p = self.model.fourgram_prob(prev4, prev3, prev2, prev1)
        fivegram_p = self.model.fivegram_prob(prev5, prev4, prev3, prev2, prev1)

        probs = self.lambdas_fivegram * np.array([unigram_p, bigram_p, trigram_p, fourgram_p, fivegram_p])
        total_prob = np.sum(probs)
        
        return probs / total_prob if total_prob > 0 else np.zeros(5)

    def m_step(self, responsibilities, lambdas_count):
        """Update lambda values based on the responsibilities."""
        total_responsibilities = np.sum(responsibilities, axis=0)
        return total_responsibilities / np.sum(total_responsibilities) if np.sum(total_responsibilities) > 0 else np.ones(lambdas_count) / lambdas_count

    def train(self):
        # Initialize lambdas for each case
        self.lambdas_bigram = np.array([0.5, 0.5])  # for one-word input case
        self.lambdas_trigram = np.array([1/3, 1/3, 1/3])  # for two-word input case
        self.lambdas_fourgram = np.array([1/4, 1/4, 1/4, 1/4])  # for three-word input case
        self.lambdas_fivegram = np.array([1/5, 1/5, 1/5, 1/5, 1/5])  # for four-word input case

        # Train for bigram case (one-word input)
        for _ in range(self.max_iter):
            responsibilities = []
            for line in self.held_out_data:
                words = line.strip().split()
                # words = ['<s>'] + line.strip().split() + ['</s>']
                # if len(words) >= 2:
                #     w1, w2 = words[-2], words[-1]
                #     resp = self.e_step_bigram(w1, w2)
                #     responsibilities.append(resp)
                if len(words) == 1:
                    prev2 = words[-1]
                    prev1 = "</s>"
                else:
                    prev2 = words[-2] if len(words) > 1 else ""
                    prev1 = words[-1] if len(words) > 0 else ""
                resp = self.e_step_bigram(prev2, prev1)
                responsibilities.append(resp)
            new_lambdas = self.m_step(np.array(responsibilities), 2)
            if np.sum(np.abs(new_lambdas - self.lambdas_bigram)) < self.tol:
                break
            self.lambdas_bigram = new_lambdas

        # Train for trigram case (two or more words input)
        for _ in range(self.max_iter):
            responsibilities = []
            for line in self.held_out_data:
                words = line.strip().split()
                # words = ['<s>'] + line.strip().split() + ['</s>']
                # if len(words) >= 3:
                    # w1, w2, w3 = words[-3], words[-2], words[-1]
                    # resp = self.e_step_trigram(w1, w2, w3)
                    # responsibilities.append(resp)
                if len(words) == 1:  
                    prev3 = ""
                    prev2 = words[-1]
                    prev1 = "</s>"
                else: 
                    prev3 = words[-3] if len(words) > 2 else ""
                    prev2 = words[-2] if len(words) > 1 else ""
                    prev1 = words[-1] if len(words) > 0 else ""
                resp = self.e_step_trigram(prev3, prev2, prev1)
                responsibilities.append(resp)
            new_lambdas = self.m_step(np.array(responsibilities), 3)
            if np.sum(np.abs(new_lambdas - self.lambdas_trigram)) < self.tol:
                break
            self.lambdas_trigram = new_lambdas

        # Train for fourgram case (three or more words input)
        for _ in range(self.max_iter):
            responsibilities = []
            for line in self.held_out_data:
                words = line.strip().split()
                # words = ['<s>'] + line.strip().split() + ['</s>']
                # if len(words) >= 4:
                #     w1, w2, w3, w4 = words[-4], words[-3], words[-2], words[-1]
                #     resp = self.e_step_fourgram(w1, w2, w3, w4)
                #     responsibilities.append(resp)
                if len(words) == 1:
                    prev4 = ""
                    prev3 = ""
                    prev2 = words[-1]
                    prev1 = "</s>"
                else:
                    prev4 = words[-4] if len(words) > 3 else ""
                    prev3 = words[-3] if len(words) > 2 else ""
                    prev2 = words[-2] if len(words) > 1 else ""
                    prev1 = words[-1] if len(words) > 0 else ""
                resp = self.e_step_fourgram(prev4, prev3, prev2, prev1)
                responsibilities.append(resp)
            new_lambdas = self.m_step(np.array(responsibilities), 4)
            if np.sum(np.abs(new_lambdas - self.lambdas_fourgram)) < self.tol:
                break
            self.lambdas_fourgram = new_lambdas

        # Train for fivegram case (four or more words input)
        for _ in range(self.max_iter):
            responsibilities = []
            for line in self.held_out_data:
                words = line.strip().split()
                # words = ['<s>'] + line.strip().split() + ['</s>']
                # if len(words) >= 4:
                #     w1, w2, w3, w4 = words[-4], words[-3], words[-2], words[-1]
                #     resp = self.e_step_fourgram(w1, w2, w3, w4)
                #     responsibilities.append(resp)
                if len(words) == 1:
                    prev5 = ""
                    prev4 = ""
                    prev3 = ""
                    prev2 = words[-1]
                    prev1 = "</s>"
                else:
                    prev5 = words[-5] if len(words) > 4 else ""
                    prev4 = words[-4] if len(words) > 3 else ""
                    prev3 = words[-3] if len(words) > 2 else ""
                    prev2 = words[-2] if len(words) > 1 else ""
                    prev1 = words[-1] if len(words) > 0 else ""
                resp = self.e_step_fivegram(prev5, prev4, prev3, prev2, prev1)
                responsibilities.append(resp)
            new_lambdas = self.m_step(np.array(responsibilities), 4)
            if np.sum(np.abs(new_lambdas - self.lambdas_fivegram)) < self.tol:
                break
            self.lambdas_fivegram = new_lambdas

        # Round lambdas to 4 decimal places and print results
        print("Optimal lambdas for bigram suggestion (one-word input):", np.round(self.lambdas_bigram, 4))
        print("Optimal lambdas for trigram suggestion (two or more words input):", np.round(self.lambdas_trigram, 4))
        print("Optimal lambdas for fourgram suggestion (three or more words input):", np.round(self.lambdas_fourgram, 4))
        print("Optimal lambdas for fivegram suggestion (four or more words input):", np.round(self.lambdas_fivegram, 4))

# Step 4: Suggestion function based on input length
def suggest_next_word(model, lambdas_bigram, lambdas_trigram, input_prefix, top_n=5):
    input_words = input_prefix.split()
    if len(input_words) == 1:
        w1 = input_words[-1]
        probabilities = {
            w: lambdas_bigram[0] * model.unigram_prob(w) + lambdas_bigram[1] * model.bigram_prob(w1, w)
            for w in model.unigram_counts
        }
    else:
        w1, w2 = input_words[-2], input_words[-1]
        probabilities = {
            w: (lambdas_trigram[0] * model.unigram_prob(w) +
                lambdas_trigram[1] * model.bigram_prob(w2, w) +
                lambdas_trigram[2] * model.trigram_prob(w1, w2, w))
            for w in model.unigram_counts
        }

    # Return the word with the highest probability
    # return max(probabilities, key=probabilities.get)
    contains_compose = "compose" in probabilities
    if "compose" in probabilities:
        print(f"The word 'compose' is in the list of possible next words with probability: {probabilities['compose']}")
    else:
        print("The word 'compose' is NOT in the list of possible next words.")

    top_suggestions = sorted(probabilities.items(), key=lambda item: item[1], reverse=True)[:top_n]
    return top_suggestions

# Step 5: Execute the code
training_data, held_out_data = read_and_split_data('shuffled_combine_output.txt')
ngram_model = NGramModel(training_data)

# Run the EM algorithm for both cases
em = EMAlgorithm(ngram_model, held_out_data)
em.train()

# Test the suggestion function
# input_words = "sudo docker compose"
# top_suggestions = suggest_next_word(ngram_model, em.lambdas_bigram, em.lambdas_trigram, input_words)
# print("Top suggestions:", top_suggestions)
