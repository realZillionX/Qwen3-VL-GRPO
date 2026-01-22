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
        # 1. Extract content from <answer>...</answer> if present
        # If explicitly found, use that content. If not, fallback to full text (or 0 reward).
        start_tag = "<answer>"
        end_tag = "</answer>"
        if start_tag in completion and end_tag in completion:
            try:
                # Find the LAST answer block if multiple (or first, but typically last is result)
                # Let's take the content of the last matching block
                start_idx = completion.rfind(start_tag) + len(start_tag)
                end_idx = completion.find(end_tag, start_idx)
                if end_idx != -1:
                    text = completion[start_idx:end_idx]
                else:
                    text = completion
            except:
                 text = completion
        else:
            text = completion

        # Visualize extraction for debugging if needed (or just proceed)
        
        # 2. Heuristic extraction from the (extracted) text
        # Using a regex to find "A", "B", "C", "D", "E"
        
        # Normalize
        text = text.strip()

        
        # Normalize
        sol = sol.strip()
        
        # Pattern: look for explicit "Option X" or just "X" at the end
        # Simple approach: Check if the solution letter is present and no other letters are prioritized?
        # Or blindly check strict match if the model is good?
        # Let's try to extract the last capital letter in the range A-E.
        
        matches = re.findall(r'[A-E]', text)
        if matches:
            prediction = matches[-1] # Take the last one
            if prediction == sol:
                rewards.append(1.0) # Correct
            else:
                rewards.append(0.0) # Wrong answer, but valid format
        else:
            rewards.append(-1.0) # Format Error: No valid option found
            
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
            
            # Extract content from <answer>...</answer>
            start_tag = "<answer>"
            end_tag = "</answer>"
            if start_tag in completion and end_tag in completion:
                start_idx = completion.rfind(start_tag) + len(start_tag)
                end_idx = completion.find(end_tag, start_idx)
                if end_idx != -1:
                    completion = completion[start_idx:end_idx]

            # Parse completion 
            # Look for a list pattern "[...]"
            match = re.search(r'\[(.*?)\]', completion, re.DOTALL)
            if match:
                content = match.group(1)
                # Split by comma
                try:
                    pred_path = [int(x.strip()) for x in content.split(',') if x.strip().isdigit()]
                    
                    if pred_path == sol_path:
                        rewards.append(1.0) # Correct
                    else:
                        rewards.append(0.0) # Wrong path, but valid format
                except:
                    rewards.append(-1.0) # Format Error: List content invalid (not ints)
            else:
                rewards.append(-1.0) # Format Error: No list found
                
        except Exception:
            rewards.append(0.0) # Should not happen unless solution parsing fails
            
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
