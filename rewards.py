import re
import json

def reward_eyeballing(completions, solution, **kwargs):
    """
    Reward function for eyeballing task.
    Correct Answer is a single letter (A-E).
    Rules:
    - Text inside <answer> must be exactly one letter.
    - If not single letter (e.g. "A." or "Option A"): -1.0 (Format Error).
    - If single letter but wrong (e.g. "B" when ans is "A", or "Z"): 0.0.
    - If correct single letter: 1.0.
    """
    rewards = []
    for completion, sol in zip(completions, solution):
        # 1. Extract content from <answer>...</answer>
        start_tag = "<answer>"
        end_tag = "</answer>"
        text = completion
        if start_tag in completion and end_tag in completion:
            try:
                start_idx = completion.rfind(start_tag) + len(start_tag)
                end_idx = completion.find(end_tag, start_idx)
                if end_idx != -1:
                    text = completion[start_idx:end_idx]
            except:
                 pass
        
        # Normalize
        text = text.strip()
        sol = sol.strip()
        
        # Strict Format Check
        if len(text) != 1 or not text.isalpha():
            rewards.append(-1.0)
            continue
            
        # Is a single letter
        # Case Insensitive comparison just in case, though usually upper
        if text.upper() == sol.upper():
            rewards.append(1.0)
        else:
            rewards.append(0.0)
            
    return rewards

def reward_maze(completions, solution, **kwargs):
    """
    Reward function for maze task.
    Solution is a JSON string of a list of integers, e.g. "[1, 2, 3]".
    Rules:
    - Must be a valid list format. If not: -1.0.
    - If Exact Match: 1.0.
    - If Partial: (LongestPrefixMatch + LongestSuffixMatch) / PathLen.
    """
    rewards = []
    for completion, sol_str in zip(completions, solution):
        try:
            # Parse solution
            sol_path = json.loads(sol_str)
            total_len = len(sol_path)
            if total_len == 0:
                rewards.append(0.0) # Should not happen
                continue

            # Extract content from <answer>...</answer>
            extract_content = completion
            start_tag = "<answer>"
            end_tag = "</answer>"
            if start_tag in completion and end_tag in completion:
                start_idx = completion.rfind(start_tag) + len(start_tag)
                end_idx = completion.find(end_tag, start_idx)
                if end_idx != -1:
                    extract_content = completion[start_idx:end_idx]

            # Parse completion 
            # Look for a list pattern "[...]"
            match = re.search(r'\[(.*?)\]', extract_content, re.DOTALL)
            if match:
                content = match.group(1)
                # Split by comma
                try:
                    pred_path = [int(x.strip()) for x in content.split(',') if x.strip() and (x.strip().isdigit() or x.strip().lstrip('-').isdigit())]
                    
                    # Logic
                    if pred_path == sol_path:
                        rewards.append(1.0)
                    else:
                        # Compute Prefix Match
                        prefix_len = 0
                        for p, s in zip(pred_path, sol_path):
                            if p == s:
                                prefix_len += 1
                            else:
                                break
                        
                        # Compute Suffix Match
                        suffix_len = 0
                        for p, s in zip(reversed(pred_path), reversed(sol_path)):
                            if p == s:
                                suffix_len += 1
                            else:
                                break
                                
                        # Formula: (Pre + Suf) / Total
                        # Note: If pred_path is exactly sol_path, this logic would define Pre=Len, Suf=Len => 2.0
                        # But we handled exact match above with return 1.0.
                        
                        # Edge case: If paths overlap heavily but distinct? 
                        # E.g. A=[1,2], B=[1]. Pre=1, Suf=0. Score=0.5.
                        # E.g. A=[1,2,3], B=[1,9,3]. Pre=1, Suf=1. Total=3. Score=2/3.
                        
                        score = (prefix_len + suffix_len) / total_len
                        rewards.append(score)
                        
                except Exception:
                     # Parsed list but content error?
                    rewards.append(-1.0)
            else:
                rewards.append(-1.0) # Format Error: No list found
                
        except Exception:
            rewards.append(-1.0) # General parse error
            
    return rewards

def reward_format(completions, **kwargs):
    """
    Reward for following the format.
    Eyeballing: Simply outputting a valid option.
    Maze: Outputting a list.
    """
    rewards = []
    # We might need to know the task type to apply specific format check.
    # But here we only receive completions. 
    # If we mix data, we need 'solution' or input to know task type.
    # GRPO trainer passes 'solution' (ref: grpo_trainer.py logic if configured).
    
    # We will just return 0.0 here as placeholder or generic checks (e.g. not empty)
    for c in completions:
        if c.strip():
            rewards.append(0.1) # Small reward for generating something
        else:
            rewards.append(0.0)
    return rewards

# Combined function caller if needed, but ms-swift allows list of functions
