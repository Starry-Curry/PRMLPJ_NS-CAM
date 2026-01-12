import re
import collections
import json
try:
    from rouge_score import rouge_scorer
except ImportError:
    rouge_scorer = None

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    s = str(s).lower()
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    s = re.sub(r'[^\w\s]', '', s) # remove punctuation
    return ' '.join(s.split())

def calculate_f1(truth, prediction):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(truth).split()
    if not prediction_tokens or not ground_truth_tokens:
        return 0.0
        
    common = collections.Counter(prediction_tokens) & collections.Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def calculate_rouge(truth, prediction):
    if not rouge_scorer:
        return {}
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(truth, prediction)
    return {k: v.fmeasure for k, v in scores.items()}

class Evaluator:
    def __init__(self, llm_interface=None):
        self.llm = llm_interface

    def evaluate_llm(self, question, truth, prediction):
        if not self.llm:
            return None, None
            
        prompt = f"""
        You are an evaluator. Compare the Ground Truth and the Prediction for the Question.
        
        Question: {question}
        Ground Truth: {truth}
        Prediction: {prediction}
        
        Assess if the Prediction conveys the same meaning as the Ground Truth.
        Provide a score from 0 to 10 (10 being perfect match in meaning) and a brief reason.
        
        Output JSON only: {{"score": <number>, "reason": "<string>"}}
        """
        try:
            resp = self.llm.generate(prompt, max_tokens=150)
            # Basic cleanup if markdown is returned
            clean_resp = resp.replace('```json', '').replace('```', '').strip()
            data = json.loads(clean_resp)
            return data.get('score'), data.get('reason')
        except Exception as e:
            return None, str(e)

    def evaluate_all(self, question, truth, prediction):
        f1 = calculate_f1(truth, prediction)
        rouge = calculate_rouge(truth, prediction)
        llm_score, llm_reason = self.evaluate_llm(question, truth, prediction)
        
        return {
            "f1": f1,
            "rouge": rouge,
            "llm_eval": {
                "score": llm_score,
                "reason": llm_reason
            }
        }
