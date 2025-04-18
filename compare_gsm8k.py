import torch
import re
import os
import argparse
import random
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    StoppingCriteriaList
)
from utils import (
    SpecificStringStoppingCriteria,
    extract_predicted_answer,
    extract_ground_truth
)
from datasets import load_dataset
from collections import Counter
import json

def extract_final_answer(outputs, tokenizer):
    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    output_text = output_text.split("A:")[-1].strip() 
    model_answer = extract_predicted_answer(output_text)
    return {'text': output_text, 'numeric': model_answer}

def parse_answer(example, ground_truth_answer, model_answers):
    # Process majority voting answers
    numeric_answers = [ma['numeric'] for ma in model_answers]
    filtered_answers = [num for num in numeric_answers if num is not None]
    majority_answer = Counter(filtered_answers).most_common(1)[0][0] if filtered_answers else None

    # Check correctness
    correct = (majority_answer == ground_truth_answer) if majority_answer is not None else False
    return {
        'question': example['question'],
        'gold_answer_text': example['answer'],
        'model_answers_text': [ma['text'] for ma in model_answers],
        'extracted_model_answers': numeric_answers,
        'extracted_gold_answer': ground_truth_answer,
        'majority_answer': majority_answer,
        'correct': correct
    }
    
def dump_single_results(model_type, args, results):
    """
    model_type str: Can be either 'base' or 'finetuned'
    """
    cnt = 0
    for result in results:
        if result['correct']:
            cnt += 1
    total = len(results)
    print(f"Accuracy: {cnt} / {total} = {cnt / total :.4f}")
    
    results.append({'accuracy': cnt / total})

    os.makedirs('eval_results/zero_shot', exist_ok=True)
    
    if model_type == 'base':
        model_name = args.base_model.split('/')[-1]
    elif model_type == 'finetuned':
        model_name = args.finetuned_model.split('/')[-1]
    else:
        raise ValueError("model_type should be either 'base' or 'finetuned'")
    result_file = f"eval_results/zero_shot/{model_name}_{model_type}"
    if args.use_cot_prompt:
        result_file += "_cot"
    if args.use_majority_vote:
        result_file += f"_maj1@{args.n_votes}_temp{args.temp}"
    result_file += "_results.json"

    with open(result_file, 'w') as f:
        json.dump(results, f, indent=4)

    print(f"Results saved to {result_file}")
    
def extract_comparison_result(b_result, f_result):
    return {
                "question": b_result['question'],
                "gold_answer_text": b_result['gold_answer_text'],
                "base_model_answer_text": b_result['model_answers_text'],
                "finetuned_model_answer_text": f_result['model_answers_text'],
                "extracted_base_model_answers": b_result['extracted_model_answers'],
                "extracted_finetuned_model_answers": f_result['extracted_model_answers'],
                "extracted_gold_answer": b_result['extracted_gold_answer'],
                "base_model_majority_answer": b_result['majority_answer'],
                "finetuned_model_majority_answer": f_result['majority_answer'],
                "base_model_correct": b_result['correct'],
                "finetuned_model_correct": f_result['correct'],
            }
  

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_model', type=str, default='mistralai/Mistral-7B-v0.1')
    parser.add_argument('--finetuned_model', type=str)
    parser.add_argument('--use_majority_vote', action='store_true')
    parser.add_argument("--temp", type=float, default=0)
    parser.add_argument('--n_votes', type=int, default=1)
    parser.add_argument("--use_cot_prompt", action="store_true")
    parser.add_argument("--base_device_map", type=str, default="auto")
    parser.add_argument("--finetuned_device_map", type=str, default="auto")
    args = parser.parse_args()


    random_seed = 42
    torch.manual_seed(random_seed)
    random.seed(random_seed)

    print('Loading base model and tokenizer...')    # Denote base model and things with b_
    b_tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    b_tokenizer.pad_token = b_tokenizer.eos_token
    b_model = AutoModelForCausalLM.from_pretrained(args.base_model, device_map=args.base_device_map, torch_dtype=torch.float16) 
    
    assert args.finetuned_model is not None, "Please provide a finetuned model path."   # Denote finetuned model and things with f_
    print('Loading finetuned model and tokenizer...')
    f_tokenizer = AutoTokenizer.from_pretrained(args.finetuned_model)
    f_tokenizer.pad_token = f_tokenizer.eos_token
    f_model = AutoModelForCausalLM.from_pretrained(args.finetuned_model, device_map=args.finetuned_device_map, torch_dtype=torch.float16) 
    
    print('\nLoading dataset...')
    dataset = load_dataset('gsm8k', "main", split='test')
    datasize = len(dataset)
    print('gsm8k test size:', datasize) 

    # Define a stopping condition for generation
    generation_util = [
        "Q:",
        "</s>",
        "<|im_end|>",
        #The following tokens were taken from https://github.com/QwenLM/Qwen/blob/main/eval/evaluate_gsm8k.py
        "<|endoftext|>",
        "\n\n\n",
        "\n\n",
        "Question:",
    ]

    b_results = []
    f_results = []
    for i in tqdm(range(datasize), desc='Evaluating'):
        example = dataset[i]
        if args.use_cot_prompt:
            input_text = "Q: {question}\nA: Let's think step by step.".format(question=example['question'])
        else:
            input_text = 'Q: ' + example['question'] + '\nA:'
        b_inputs = b_tokenizer(input_text, return_tensors='pt').to(b_model.device)
        f_inputs = f_tokenizer(input_text, return_tensors='pt').to(f_model.device)
        ground_truth_answer = extract_ground_truth(example['answer'])

        # Define a stopping condition for generation
        b_stop_criteria = SpecificStringStoppingCriteria(b_tokenizer, generation_util, len(input_text))
        f_stop_criteria = SpecificStringStoppingCriteria(f_tokenizer, generation_util, len(input_text))
        b_stopping_criteria_list = StoppingCriteriaList([b_stop_criteria])
        f_stopping_criteria_list = StoppingCriteriaList([f_stop_criteria])

        b_model_answers = []
        f_model_answers = []
        if args.use_majority_vote:
            for _ in range(args.n_votes):
                with torch.no_grad():
                    b_outputs = b_model.generate(**b_inputs, temperature=args.temp, max_new_tokens=512, do_sample=True, pad_token_id=b_tokenizer.eos_token_id, stopping_criteria=b_stopping_criteria_list)
                    f_outputs = f_model.generate(**f_inputs, temperature=args.temp, max_new_tokens=512, do_sample=True, pad_token_id=f_tokenizer.eos_token_id, stopping_criteria=f_stopping_criteria_list)
                b_model_answers.append(extract_final_answer(b_outputs, b_tokenizer))
                f_model_answers.append(extract_final_answer(f_outputs, f_tokenizer))
        else:
            with torch.no_grad():
                b_outputs = b_model.generate(**b_inputs, max_new_tokens=512, pad_token_id=b_tokenizer.eos_token_id, stopping_criteria=b_stopping_criteria_list)
                f_outputs = f_model.generate(**f_inputs, max_new_tokens=512, pad_token_id=f_tokenizer.eos_token_id, stopping_criteria=f_stopping_criteria_list)
            b_model_answers.append(extract_final_answer(b_outputs, b_tokenizer))
            f_model_answers.append(extract_final_answer(f_outputs, f_tokenizer))
            
        b_results.append(parse_answer(example, ground_truth_answer, b_model_answers))
        f_results.append(parse_answer(example, ground_truth_answer, f_model_answers))
        
    both_correct_results = []
    both_incorrect_results = []
    b_correct_results = []
    f_correct_results = []
    # TODO aquí en vez de devolver el que está bien, devolver un dict mezcla entre los dos que tenga toda la info. hacer un append un dict nuevo
    for b_result, f_result in zip(b_results, f_results):
        if b_result['correct'] and f_result['correct']:
            both_correct_results.append(extract_comparison_result(b_result, f_result))
        elif not b_result['correct'] and not f_result['correct']:
            both_incorrect_results.append(extract_comparison_result(b_result, f_result))
        elif b_result['correct'] and not f_result['correct']:
            b_correct_results.append(extract_comparison_result(b_result, f_result))
        elif not b_result['correct'] and f_result['correct']:
            f_correct_results.append(extract_comparison_result(b_result, f_result))
            
    os.makedirs('eval_results/zero_shot', exist_ok=True)
    b_model_name = args.base_model.split('/')[-1]
    f_model_name = args.finetuned_model.split('/')[-1]
    
    # Save both correct results
    both_correct_path = f"eval_results/zero_shot/c_{b_model_name}_c_{f_model_name}_results.json"
    with open(both_correct_path, 'w') as f:
        json.dump(both_correct_results, f, indent=4)
    print(f"Results saved to {both_correct_path}")
        
    # Save both incorrect results
    both_incorrect_path = f"eval_results/zero_shot/ic_{b_model_name}_ic_{f_model_name}_results.json"
    with open(both_incorrect_path, 'w') as f:
        json.dump(both_incorrect_results, f, indent=4)
    print(f"Results saved to {both_incorrect_path}")
    
    # Save base model correct only results
    b_correct_path = f"eval_results/zero_shot/c_{b_model_name}_ic_{f_model_name}_results.json"
    with open(b_correct_path, 'w') as f:
        json.dump(b_correct_results, f, indent=4)
    print(f"Results saved to {b_correct_path}")
    
    # Save finetuned model correct only results
    f_correct_path = f"eval_results/zero_shot/ic_{b_model_name}_c_{f_model_name}_results.json"
    with open(f_correct_path, 'w') as f:
        json.dump(f_correct_results, f, indent=4)
    print(f"Results saved to {f_correct_path}")
    
    dump_single_results("base", args, b_results)
    dump_single_results("finetuned", args, f_results)
    
    
                

if __name__ == '__main__':
    main()

