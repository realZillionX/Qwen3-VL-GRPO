import re
import json

def reward_eyeballing(completions, solution, **kwargs):
    """
    Reward function for eyeballing task.
    Correct Answer is a single letter (A-E).
    Reward: 1.0 if correct, 0.0 otherwise.
    """
    rewards = []
    for completion, sol in zip(completions, solution):
        # Extract the last single letter from the completion as the answer
        # Heuristic: Find simple letter pattern or take the last non-punctuated token
        # Using a regex to find "A", "B", "C", "D", "E"
        
        # Normalize
        text = completion.strip()
        sol = sol.strip()
        
        # Pattern: look for explicit "Option X" or just "X" at the end
        # Simple approach: Check if the solution letter is present and no other letters are prioritized?
        # Or blindly check strict match if the model is good?
        # Let's try to extract the last capital letter in the range A-E.
        
        matches = re.findall(r'[A-E]', text)
        if matches:
            prediction = matches[-1] # Take the last one
            if prediction == sol:
                rewards.append(1.0)
            else:
                rewards.append(0.0)
        else:
            rewards.append(0.0)
            
    return rewards

def reward_maze(completions, solution, **kwargs):
    """
    Reward function for maze task.
    Solution is a JSON string of a list of integers, e.g. "[1, 2, 3]".
    Completion should be a list of integers.
    Reward: 1.0 if exact match of path, partial reward for valid format? 
    For GRPO, binary reward 1.0/0.0 is standard, but soft reward for partial path correctness could be added.
    Here we stick to strict correctness for simplicity, or maybe checking endpoints.
    """
    rewards = []
    for completion, sol_str in zip(completions, solution):
        try:
            # Parse solution
            sol_path = json.loads(sol_str)
            
            # Parse completion 
            # Look for a list pattern "[...]"
            match = re.search(r'\[(.*?)\]', completion, re.DOTALL)
            if match:
                content = match.group(1)
                # Split by comma
                try:
                    pred_path = [int(x.strip()) for x in content.split(',') if x.strip().isdigit()]
                    
                    if pred_path == sol_path:
                        rewards.append(1.0)
                    else:
                        # Optional: Partial reward based on Longest Common Subsequence or validity
                        # For now, 0.0
                        rewards.append(0.0)
                except:
                    rewards.append(0.0)
            else:
                rewards.append(0.0)
                
        except Exception:
            rewards.append(0.0)
            
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
