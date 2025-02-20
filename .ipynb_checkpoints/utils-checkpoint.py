import re

def correctness_reward_func(completions, **kwargs) -> list[float]:
    extracted_responses = [extract_xml_answer(r) for r in completions]
    # print('-'*20, f"Question:\n{q}", f"\nAnswer:\n{answer[0]}", f"\nResponse:\n{responses[0]}", f"\nExtracted:\n{extracted_responses[0]}")
    answer = kwargs['answers']
    return [2.0 if r == a else 0.0 for r, a in zip(extracted_responses, answer)]

def int_reward_func(completions, **kwargs) -> list[float]:
    extracted_responses = [extract_xml_answer(r) for r in completions]
    return [0.5 if r.isdigit() else 0.0 for r in extracted_responses]

def strict_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""

    pattern = r"^<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>\n$"
    matches = [re.match(pattern, r) for r in completions] 
    return [0.5 if match else 0.0 for match in matches]

def soft_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
    matches = [re.match(pattern, r) for r in completions] 
    return [0.5 if match else 0.0 for match in matches]

def count_xml(text) -> float:
    count = 0.0
    if text.count("<reasoning>\n") == 1:
        count += 0.125
    if text.count("\n</reasoning>\n") == 1:
        count += 0.125
    if text.count("\n<answer>\n") == 1:
        count += 0.125
        count -= len(text.split("<answer>")[0]) * 0.001
    if text.count("\n</answer>") == 1:
        count += 0.125
        count -= (len(text.split("\n</answer>")[-1]) - 1)*0.001
    return count

def xmlcount_reward_func(completions, **kwargs) -> list[float]:

    return [count_xml(c) for c in completions]

def extract_xml_answer(text: str) -> str:
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()

def total_reward(completions, **kwargs) -> list[float]:
    rewards = [
        correctness_reward_func(completions, **kwargs),
        int_reward_func(completions, **kwargs),
        strict_format_reward_func(completions, **kwargs),
        soft_format_reward_func(completions, **kwargs),
        xmlcount_reward_func(completions, **kwargs)
    ]
    return [sum(r) for r in zip(*rewards)], rewards

def extract_hash_answer(text: str | None) -> str | None:
    if "####" not in text:
        return None
    return text.split("####")[1].strip().replace(",", "").replace("$", "")