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
    """Get response from the model."""
    # Add system message to the start of the conversation
    full_messages = [
        {"role": "system", "content": "You are a helpful AI assistant. Keep all your responses to a single, concise sentence."}
    ] + messages
    
    completion = client.chat.completions.create(
        extra_body={},
        model="meta-llama/llama-3.3-70b-instruct:nitro",
        messages=full_messages,
        max_tokens=max_tokens
    )
    return completion.choices[0].message.content

def analyze_features(goodfire_client: goodfire.Client, 
                    user_message: str, 
                    assistant_response: str) -> List[Dict]:
    """Analyze response features using Goodfire."""
    variant = goodfire.Variant("meta-llama/Llama-3.3-70B-Instruct")
    
    context = goodfire_client.features.inspect(
        messages=[
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": assistant_response}
        ],
        model=variant
    )
    
    return context.top(k=3)

def get_feature_summary(client: OpenAI, feature_labels: str) -> str:
    """Get a summary of the features from the model."""
    prompt = (
        "Your input is a list of features an LLM is thinking about. "
        "Your output should be a statement in the form of 'The LLM is thinking about [feature1], [feature2], and [feature3].', "
        "with each [feature] being a concise description of the feature with a relevant emoji at the end. The input is: "
    ) + feature_labels
    
    completion = client.chat.completions.create(
        extra_body={},
        model="meta-llama/llama-3.3-70b-instruct:nitro",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=50
    )
    return completion.choices[0].message.content

def main():
    try:
        # Load API keys
        goodfire_api_key, openrouter_api_key = load_api_keys()
        
        # Initialize clients
        openai_client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=openrouter_api_key,
        )
        goodfire_client = goodfire.Client(api_key=goodfire_api_key)
        
        # Initialize conversation history
        conversation_history = []
        
        print(f"{Fore.CYAN}Welcome to the AI Chat! Type 'exit' or 'quit' to end the conversation.{Style.RESET_ALL}")
        print(f"{Fore.CYAN}Each response is limited to 50 tokens.{Style.RESET_ALL}\n")
        
        while True:
            # Get user input
            user_input = input(f"{Fore.GREEN}You: {Style.RESET_ALL}")
            
            # Check for exit command
            if user_input.lower() in ['exit', 'quit']:
                print(f"\n{Fore.CYAN}Goodbye!{Style.RESET_ALL}")
                break
            
            # Add user message to history
            conversation_history.append({"role": "user", "content": user_input})
            
            # Get model response with loading animation
            loading_animation = LoadingAnimation("Assistant is thinking", Fore.CYAN)
            loading_animation.start()
            response = get_model_response(openai_client, conversation_history)
            loading_animation.stop()
            print(f"{Fore.CYAN}Assistant: {response}{Style.RESET_ALL}")
            
            # Add assistant response to history
            conversation_history.append({"role": "assistant", "content": response})
            
            # Analyze features with loading animation
            loading_animation = LoadingAnimation("Analysis loading", Fore.MAGENTA)
            loading_animation.start()
            features = analyze_features(goodfire_client, user_input, response)
            feature_labels = ", ".join([feature.feature.label for feature in features])
            feature_summary = get_feature_summary(openai_client, feature_labels)
            loading_animation.stop()
            print(f"{Fore.MAGENTA}Analysis: {feature_summary}{Style.RESET_ALL}\n")

    except Exception as e:
        print(f"{Fore.RED}An error occurred: {str(e)}{Style.RESET_ALL}")
        return

if __name__ == "__main__":
    main() 