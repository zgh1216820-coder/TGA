"""HSSP 选择题解析：从预测文本中提取选项字母。"""
import re
import string
ALPHABET = list(string.ascii_uppercase)

def infer_hssp_choice_options(text, choices):
    text = re.split(r'\n\s*Explanation\s*:', str(text).strip(), maxsplit=1, flags=re.IGNORECASE)[0].strip()
    if not choices:
        return []
    matches = []
    for token in re.findall(r'\b[A-Z]\b|\b\d+\b', text.upper()):
        if token in choices and token not in matches:
            matches.append(token)
    if matches:
        return matches

    pred_option = can_infer(text, choices)
    if isinstance(pred_option, str):
        return [pred_option]
    return []

def parse_hssp_choice_list(input_string):
    # HSSP 选择题格式解析。
    choice_list_patterns = [
        r'Choice list:\[(.*?)\]',
        r'Choice lists: \[(.*?)\]',
        r'Choice list: \[(.*?)\]',
        r'Choice List: \[(.*?)\]',
    ]
    for pattern in choice_list_patterns:
        match = re.search(pattern, input_string, re.DOTALL)
        if not match:
            continue
        raw = match.group(1).strip()
        parts = [part.strip().strip("'\"") for part in re.split(r'\s*\|\s*|\s*,\s*', raw) if part.strip()]
        letters = []
        for idx, part in enumerate(parts):
            keyed = re.match(r'^([A-Z]|\d+)\s*[\.:)]\s*(.*)$', part, re.DOTALL)
            letters.append(keyed.group(1).strip() if keyed else ALPHABET[idx])
        if letters:
            return letters

    match = re.search(r'Options:\s*(.*?)(?:\n\s*Please answer|\Z)', input_string, re.IGNORECASE | re.DOTALL)
    if match:
        raw = match.group(1).strip()
        keyed = re.findall(
            r'(?s)([A-Z]|\d+)\s*[:\.)]\s*(.*?)(?=(?:\n\s*)?(?:[A-Z]|\d+)\s*[:\.)]|\Z)',
            raw,
        )
        if keyed:
            return [key.strip() for key, _ in keyed]

    match = re.findall(r'(?m)^\s*([A-D])\.\s+.*$', input_string)
    if match:
        return match

    return []

def parse_choice_list(input_string, hssp_format=False):
    return parse_hssp_choice_list(input_string)

def can_infer(answer, choices):
    if len(answer) == 0:
        return False
    answer = str(answer).lower()

    # Special case for ['Positive', 'Negative']
    if set(choices) == {'Positive', 'Negative'}:
        if 'yes' in answer or 'Yes' in answer:
            return 'Positive'
        elif 'no' in answer or 'No' in answer:
            return 'Negative'

    # First, look for exact matches if choices are not simple letters
    if not all(len(choice) == 1 and choice in string.ascii_uppercase for choice in choices):
        for choice in choices:
            if choice.lower() in answer or choice in answer:  # Allow for case-insensitive exact match
                return choice

    # Then, look for simple letter matches (A, B, C, ...)
    letter_matches = re.findall(r'\b[A-Z]\b', answer.upper())
    for letter in letter_matches:
        index = string.ascii_uppercase.index(letter)
        if index < len(choices):
            return choices[index]

    # If choices are simple letters, look for those
    if all(len(choice) == 1 and choice in string.ascii_uppercase for choice in choices):
        for choice in choices:
            if choice in answer.upper():
                return choice

    # remove underscore and try
    answer =  answer.strip().replace('_', ' ').lower()
    normalized_choices = [choice.replace('_', ' ').lower() for choice in choices]
    if answer in normalized_choices:
        return choices[normalized_choices.index(answer)]
    # Check for partial matches
    for i, choice in enumerate(normalized_choices):
        if answer in choice or choice in answer:
            return choices[i]


    # If no match found, return False
    return False
