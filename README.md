# Interpretable LLM Command-Line Chat Interface

A command-line tool that allows you to chat with Llama 3.3 70B Instruct while simultaneously viewing the internal "concepts" it's thinking about to form its responses.

## Features

- Interactive chat with Llama 3.3 70B Instruct
- Streamed responses
- Real-time feature analysis of model responses
- Configurable response length and feature analysis depth

## Setup

1. Clone this repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Create a `.env` file in the project root with your API keys:
   ```
   GOODFIRE_API_KEY=your_goodfire_api_key
   OPENROUTER_API_KEY=your_openrouter_api_key
   ```

## Usage

Run the tool:
```bash
python cli.py
```

### Command Line Options

You can control the tool's behavior using these flags in your messages:

- `--lim N`: Set the maximum number of tokens in the response (default: 150)
  ```
  tell me about quantum computing --lim 200
  ```

- `--topk N`: Set the number of features to analyze (default: 3)
  ```
  explain neural networks --topk 5
  ```

You can use both flags together:
```
tell me about AI --lim 250 --topk 4
```

### Example Session

```
You: Tell me about the ethos of Lee Kuan Yew --lim 300 --topk 4

Llama 3.3 70B Instruct (300 tokens): Lee Kuan Yew's ethos centered around pragmatic governance, economic growth, and social stability. Key tenets:

1. **Meritocracy**: Leaders and citizens should be judged on ability, not wealth or birth.
2. **Pragmatism over ideology**: Policies should be driven by effectiveness, not dogma.
3. **Economic growth**: Prosperity is key to national survival and stability.
4. **Social cohesion**: Multiracial, multireligious harmony is crucial for national unity.
5. **Strong governance**: Efficient, corruption-free government is essential for progress.
6. **Individual responsibility**: Citizens should take personal responsibility for their lives and contribute to society.
7. **Long-term thinking**: Decisions should prioritize the future, not short-term gains.

These principles guided Singapore's transformation from a poor post-colonial state to a modern, prosperous nation.

Top 4 Activating Features: The LLM is thinking about:
1. Singapore tourism and travel planning content 🗺 ️
2. Emotional storytelling with narrative progression tokens 💬
3. Singapore's public infrastructure and government services 🚂
4. Socioeconomic development and national progress 🚀
```

## Notes

- Settings persist throughout the conversation until changed
- Type 'exit' or 'quit' to end the session
- Your current settings are displayed after each response