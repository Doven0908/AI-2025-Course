from typing import Dict, Optional
import numpy as np
import math

def calculate_confidence_metrics(prob_list, text: Optional[str] = None) -> Dict[str, float]:
    num_tokens = len(prob_list)
    start_index = 1 if (text and len(text) > 0 and text[0] == "{") else 0
    end_index = num_tokens

    # Initialize accumulators for different metrics
    sampled_probs = []
    sampled_logprobs = []
    entropies = []
    top_k_prob_sums = [] # For pred_conf_5

    for i in range(start_index, end_index):
        # Ensure index is valid and corresponding logprobs entry is not empty
        if i < len(prob_list) and prob_list[i]:
            probs = list(prob_list[i].values())
            logprobs = [math.log(p) for p in probs if p > 0]
            # Extract probabilities from logprobs

            sampled_probs.append(probs)
            sampled_logprobs.append(logprobs)
            if probs:
                # Calculate entropy for the current token
                entropy = -sum(p * math.log(p) for p in probs if p > 0)
                entropies.append(entropy)

                # Calculate top-k probability sum (for pred_conf_5)
                k = min(5, len(probs))
                top_k_probs = sorted(probs, reverse=True)[:k]
                top_k_prob_sum = sum(top_k_probs)
                top_k_prob_sums.append(top_k_prob_sum)
            else:
                entropies.append(0.0)  # If no probabilities, entropy is 0
                top_k_prob_sums.append(0.0)  # If no probabilities, top-k sum is 0
                     
    results = {}
    # pred_conf_0: Arithmetic Mean of Sampled Token Probabilities
    # Represents the average "certainty" of the model at each step for the token it chose.
    pred_conf_0 = np.mean([probs[0] for probs in sampled_probs]) if sampled_probs else 0.0
    # pred_conf_1: Geometric Mean of Sampled Token Probabilities
    # More sensitive to low values; if any single token has a very low probability,
    # the geometric mean will be significantly pulled down.
    # Calculated as exp(average of logprobs).
    pred_conf_1 = np.exp(np.mean([logprobs[0] for logprobs in sampled_logprobs])) if sampled_logprobs else 0.0
    # pred_conf_2: Minimum Sampled Token Probability
    # Directly addresses the "weakest link" problem. A single very uncertain step
    # will result in a low confidence score.
    pred_conf_2 = np.min([probs[0] for probs in sampled_probs]) if sampled_probs else 0.0
    # pred_conf_3: Average Token Entropy (normalized to be a confidence score)
    # Entropy quantifies uncertainty. Lower entropy means higher certainty.
    # We normalize it to be between 0 and 1, where 1 is highest confidence.
    pred_conf_3 = 1 - (np.mean(entropies) / np.log(len(prob_list[0]))) if entropies else 0.0

    # pred_conf_4: Normalized Sequence Log-Likelihood (Raw Average Logprob)
    # This is the average log-probability of the tokens generated.
    # Higher (less negative) values indicate a more probable sequence.
    # This is a fundamental metric in language modeling.
    pred_conf_4 = np.mean([logprobs[0] for logprobs in sampled_logprobs]) if sampled_logprobs else 0.0
    # Note: This value will be negative. Higher (closer to 0) indicates higher confidence.

    # pred_conf_5: Average Top-K Probability Sum
    # Measures how much probability mass is concentrated within the top-K predicted tokens at each step.
    # If the sum of probabilities for the top-K tokens is high, it means the model is confident
    # about its top few choices.
    pred_conf_5 = np.mean(top_k_prob_sums) if top_k_prob_sums else 0.0

    # pred_conf_6: Average Max Probability
    # This is the average of the maximum probabilities assigned to the top token at each step.
    pred_conf_6 = np.mean([max(probs) for probs in sampled_probs])

    # pred_conf_7: Average Max Logprob
    # This is the average of the maximum log probabilities assigned to the top token at each step
    pred_conf_7 = np.mean([max(logprobs) for logprobs in sampled_logprobs])

    # pred_conf_8: perplexity
    # Perplexity is a measure of how well a probability distribution predicts a sample.
    # Lower perplexity indicates better predictive performance.
    if sampled_logprobs:
        perplexity = np.exp(-np.mean([logprobs[0] for logprobs in sampled_logprobs]))
    else:
        perplexity = float('inf')
    pred_conf_8 = perplexity

    # pred_conf_9 : UQ
    # Uncertainty quantification (UQ) can be calculated as the standard deviation of the sampled probabilities.
    pred_conf_9 = np.std([probs[0] for probs in sampled_probs]) if sampled_probs else 0.0

    results = {
        "pred_conf_0": pred_conf_0,
        "pred_conf_1": pred_conf_1,
        "pred_conf_2": pred_conf_2,
        "pred_conf_3": pred_conf_3,
        "pred_conf_4": pred_conf_4,
        "pred_conf_5": pred_conf_5,
        "pred_conf_6": pred_conf_6,
        "pred_conf_7": pred_conf_7,
        "pred_conf_8": pred_conf_8,
        "pred_conf_9": pred_conf_9,
    }

    return results


def certainty_from_choice(top_probs):
    H_norm = []
    for top in top_probs: # top 是 {token: p, ....}
        token_probs = list(top.values())
        kept = [p for p in token_probs if p >= 1e-5]
        s = sum(kept)
        r = max(0.0, 1.0 - s)
        k = len(kept)
        H = -sum(p * math.log(p) for p in kept)
        # if vocab_size and vocab_size > k and r > 0:
        #     p_tail = r / (vocab_size - k)
        #     H += (vocab_size - k) * (-p_tail * math.log(p_tail))
        H_norm.append(H / math.log(max(2,k)))
    C = 1.0 - (sum(H_norm) / len(H_norm))
    return C


def cgrs_certrainty_score(top_probs) -> float:
    for i,top in enumerate(top_probs):
        sorted_top = dict(sorted(top.items(), key=lambda x: x[1], reverse=True)[:5])
        top_probs[i] = sorted_top

    num_tokens = len(top_probs)
    start_index = 1 # pass {
    end_index = num_tokens
    if num_tokens < 1:
        return 0.0
    entropies = []
    for i in range(start_index, end_index):
        if top_probs[i]:
            probs = list(top_probs[i].values())
            entropy = -sum(p * math.log(p) for p in probs if p > 0)
            entropies.append(entropy)

    certainty_score = 1 - (np.mean(entropies) / np.log(len(top_probs[0]))) if entropies else 0.0
    return certainty_score
