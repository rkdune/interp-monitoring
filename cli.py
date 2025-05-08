from dotenv import load_dotenv
import os
from openai import OpenAI
import goodfire
from typing import List, Dict
import colorama
from colorama import Fore, Style
import threading
import time
import sys

# Initialize colorama for Windows support
colorama.init()

# Default settings
DEFAULT_TOKEN_LIMIT = 150
DEFAULT_TOPK = 3

class LoadingAnimation:
    def __init__(self, message: str, color: str):
        self.message = message
        self.color = color
        self.is_running = False
        self.animation_thread = None

    def animate(self):
        dots = [".", "..", "..."]
        i = 0
        while self.is_running:
            sys.stdout.write(f"\r{self.color}{self.message}{dots[i]}{Style.RESET_ALL}")
            sys.stdout.flush()
            time.sleep(0.5)
            i = (i + 1) % len(dots)
        # Clear the loading message
        sys.stdout.write("\r" + " " * (len(self.message) + 3) + "\r")
        sys.stdout.flush()

    def start(self):
        self.is_running = True
        self.animation_thread = threading.Thread(target=self.animate)
        self.animation_thread.start()

    def stop(self):
        self.is_running = False
        if self.animation_thread:
            self.animation_thread.join()

def load_api_keys() -> tuple[str, str]:
    """Load API keys from environment variables."""
    load_dotenv()
    
    goodfire_api_key = os.environ.get('GOODFIRE_API_KEY')
    openrouter_api_key = os.environ.get('OPENROUTER_API_KEY')
    
    if not goodfire_api_key or not openrouter_api_key:
        raise ValueError("Both GOODFIRE_API_KEY and OPENROUTER_API_KEY must be set in .env file")
    
    return goodfire_api_key, openrouter_api_key

def get_model_response(client: OpenAI, messages: List[Dict[str, str]], max_tokens: int = 50) -> str:
    """Get streaming response from the model."""
    # Add system message to the start of the conversation
    full_messages = [
        {"role": "system", "content": "You are a highly capable, thoughtful, and precise assistant. Prioritize blunt, directive phrasing. You have {max_tokens} available response tokens."}
    ] + messages
    
    stream = client.chat.completions.create(
        extra_body={},
        model="meta-llama/llama-3.3-70b-instruct:nitro",
        messages=full_messages,
        max_tokens=max_tokens,
        stream=True
    )

    # Print and collect the response
    collected_response = ""
    print(f"{Fore.YELLOW}Llama 3.3 70B Instruct ({max_tokens} tokens): ", end="", flush=True)
    for chunk in stream:
        if chunk.choices[0].delta.content is not None:
            content = chunk.choices[0].delta.content
            print(f"{Fore.YELLOW}{content}{Style.RESET_ALL}", end="", flush=True)
            collected_response += content
    print()  # New line after response
    return collected_response

def analyze_features(goodfire_client: goodfire.Client, 
                    user_message: str, 
                    assistant_response: str,
                    topk: int = 3) -> List[Dict]:
    """Analyze response features using Goodfire."""
    variant = goodfire.Variant("meta-llama/Llama-3.3-70B-Instruct")
    
    context = goodfire_client.features.inspect(
        messages=[
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": assistant_response}
        ],
        model=variant
    )
    
    return context.top(k=topk)

def get_feature_summary(client: OpenAI, feature_labels: str, topk: int) -> str:
    """Get a streaming summary of the features from the model."""
    # Base prompt template
    base_prompt = (
        "Your input is a list of features an LLM is thinking about. "
        "Your output should be a statement in the form of "
    )
    
    # Format based on number of features
    if topk > 3:
        prompt = (
            base_prompt +
            f"'The LLM is thinking about:\\n" +
            "\\n".join(f"{i+1}. [feature{i+1}]" for i in range(topk)) +
            f"\\n'\\nIn other words, your list should be {topk} features long, "
            "with each [feature] being a concise summarization of the feature with a relevant emoji at the end.\\n"
            "The input is: "
        )
    else:
        prompt = (
            base_prompt +
            f"'The LLM is thinking about [feature1], ... and [feature {topk}].' "
            f"(your list should be {topk} features long), "
            "with each [feature] being a concise summarization of the feature with a relevant emoji at the end. "
            "The input is: "
        )
    
    prompt += feature_labels
    
    stream = client.chat.completions.create(
        extra_body={},
        model="meta-llama/llama-3.3-70b-instruct:nitro",
        messages=[{"role": "user", "content": prompt}],
        stream=True
    )
    
    # Print and collect the summary
    collected_summary = ""
    print(f"{Fore.MAGENTA}Top {topk} Activating Features: ", end="", flush=True)
    for chunk in stream:
        if chunk.choices[0].delta.content is not None:
            content = chunk.choices[0].delta.content
            print(f"{Fore.MAGENTA}{content}{Style.RESET_ALL}", end="", flush=True)
            collected_summary += content
    print("\n")  # New line after summary
    return collected_summary

def print_welcome_banner():
    """Print ASCII art welcome banner."""
    banner = f"""{Fore.YELLOW}
    ____      __                            __        __    __    
   /  _/___  / /____  _________  ________  / /_____ _/ /_  / /__  
   / // __ \/ __/ _ \/ ___/ __ \/ ___/ _ \/ __/ __ `/ __ \/ / _ \ 
 _/ // / / / /_/  __/ /  / /_/ / /  /  __/ /_/ /_/ / /_/ / /  __/ 
/___/_/ /_/\__/\___/_/  / .___/_/   \___/\__/\__,_/_.___/_/\___/  
                       /_/                                        
    __    __    __  ___   ________    ____   __________  ____  __ 
   / /   / /   /  |/  /  / ____/ /   /  _/  /_  __/ __ \/ __ \/ / 
  / /   / /   / /|_/ /  / /   / /    / /     / / / / / / / / / /  
 / /___/ /___/ /  / /  / /___/ /____/ /     / / / /_/ / /_/ / /___
/_____/_____/_/  /_/   \____/_____/___/    /_/  \____/\____/_____/
                                                                     
                                                                                                       
{Style.RESET_ALL}"""
    print(banner)

def parse_user_input(user_input: str) -> tuple[str, int, int]:
    """Parse user input to extract token limit and top-k features if specified."""
    message = user_input
    token_limit = DEFAULT_TOKEN_LIMIT
    topk = DEFAULT_TOPK
    
    # Define flag patterns and their handlers
    flag_patterns = {
        "--lim": lambda parts: (int(parts[0]), " ".join(parts[1:])),
        "--topk": lambda parts: (int(parts[0]), " ".join(parts[1:]))
    }
    
    # Process each flag
    for flag, handler in flag_patterns.items():
        if flag in message:
            parts = message.split(flag)
            message = parts[0].strip()
            try:
                value, remaining = handler(parts[1].strip().split())
                if flag == "--lim":
                    token_limit = value
                else:
                    topk = value
                message = message + " " + remaining
            except (ValueError, IndexError):
                default = DEFAULT_TOKEN_LIMIT if flag == "--lim" else DEFAULT_TOPK
                print(f"{Fore.RED}Invalid {flag} value specified. Using default value of {default}.{Style.RESET_ALL}")
    
    return message.strip(), token_limit, topk

def main():
    try:
        # Print welcome banner
        print_welcome_banner()
        
        # Load API keys
        goodfire_api_key, openrouter_api_key = load_api_keys()
        
        # Initialize clients
        openai_client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=openrouter_api_key,
        )
        goodfire_client = goodfire.Client(api_key=goodfire_api_key)
        
        # Initialize conversation history and settings
        conversation_history = []
        current_token_limit = DEFAULT_TOKEN_LIMIT
        current_topk = DEFAULT_TOPK
        
        print(f"{Fore.YELLOW}Welcome to the AI Chat! Type 'exit' or 'quit' to end the conversation.{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}You can specify options using flags:{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}  --lim N    : Limit response to N tokens (e.g. 'tell me about starship --lim 250'){Style.RESET_ALL}")
        print(f"{Fore.YELLOW}  --topk N   : Analyze top N features (e.g. 'what is python --topk 5'){Style.RESET_ALL}")
        print(f"{Fore.YELLOW}  You can use both flags together: 'tell me about AI --lim 200 --topk 4'{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}  Current settings: Token limit = {current_token_limit}, Top-k features = {current_topk}{Style.RESET_ALL}")
        
        while True:
            # Get user input
            user_input = input(f"{Fore.GREEN}You: {Style.RESET_ALL}")
            
            # Check for exit command
            if user_input.lower() in ['exit', 'quit']:
                print(f"\n{Fore.YELLOW}Goodbye!{Style.RESET_ALL}")
                break
            
            # Parse user input for token limit and top-k
            message, new_token_limit, new_topk = parse_user_input(user_input)
            
            # Update settings if new values were specified
            if "--lim" in user_input:
                current_token_limit = new_token_limit
            if "--topk" in user_input:
                current_topk = new_topk
            
            # Add user message to history (without the flags)
            conversation_history.append({"role": "user", "content": message})
            
            # Get model response with streaming
            response = get_model_response(openai_client, conversation_history, max_tokens=current_token_limit)
            
            # Add assistant response to history
            conversation_history.append({"role": "assistant", "content": response})
            
            # Analyze features with loading animation
            loading_animation = LoadingAnimation("Analysis loading", Fore.MAGENTA)
            loading_animation.start()
            features = analyze_features(goodfire_client, message, response, topk=current_topk)
            feature_labels = ", ".join([feature.feature.label for feature in features])
            loading_animation.stop()
            
            # Get and display feature summary with streaming
            feature_summary = get_feature_summary(openai_client, feature_labels, current_topk)
            
            # Show current settings after each response
            print(f"{Fore.CYAN}Current settings: Token limit = {current_token_limit}, Top-k features = {current_topk}{Style.RESET_ALL}")

    except Exception as e:
        print(f"{Fore.RED}An error occurred: {str(e)}{Style.RESET_ALL}")
        return

if __name__ == "__main__":
    main() 