import goodfire
from typing import List, Dict, Any
import pandas as pd
from rich.console import Console
from rich.table import Table
from colorama import Fore, Style
import sys

def search_features(client: goodfire.Client, 
                   search_term: str, 
                   model_variant: Any,
                   top_k: int = 5) -> List:
    """
    Search for features matching the given search term.
    
    Args:
        client: Goodfire client
        search_term: The category name to search for
        model_variant: The model variant to use
        top_k: Number of top features to return
        
    Returns:
        List of found features
    """
    features = client.features.search(
        search_term,
        model=model_variant,
        top_k=top_k
    )
    
    return features

def track_feature_activations(goodfire_client: goodfire.Client,
                             user_message: str,
                             assistant_response: str,
                             features_to_track: List,
                             model_variant: Any,
                             sample_rate: int = 1) -> Dict[str, List[float]]:
    """
    Track feature activations for each token in the response.
    
    Args:
        goodfire_client: Goodfire client
        user_message: The user's message
        assistant_response: The assistant's response
        features_to_track: List of features to track
        model_variant: The model variant used
        sample_rate: Sample every Nth token (default: 1 = all tokens)
        
    Returns:
        Dictionary mapping feature labels to lists of activation values
    """
    # Initialize dictionary to store activations for each feature
    feature_activations = {}
    feature_names = {}
    
    # Extract the correct identifiers from the features
    for feature in features_to_track:
        # Use the feature label as the key since it's more reliable
        feature_activations[feature.label] = []
        feature_names[feature.label] = feature.label
    
    # Track feature activations for each token in the response
    tokens = assistant_response.split()
    total_tokens = len(tokens)
    
    # Sample indices based on sample_rate
    if sample_rate > 1 and total_tokens > sample_rate:
        # Always include the first and last token
        indices = [0] + [i for i in range(1, total_tokens-1, sample_rate)] + [total_tokens-1]
    else:
        indices = range(total_tokens)
    
    # Progress tracking variables
    total_steps = len(indices)
    progress_interval = max(1, total_steps // 10)  # Show progress every 10%
    
    # Store the actual tokens we're tracking
    tracked_tokens = []
    
    for step, i in enumerate(indices):
        # Show progress indicator
        if step % progress_interval == 0 or step == total_steps - 1:
            progress_percent = int((step + 1) / total_steps * 100)
            progress_bar = f"[{'=' * (progress_percent // 10)}>{' ' * (10 - progress_percent // 10)}]"
            sys.stdout.write(f"\r{Fore.CYAN}Progress: {progress_bar} {progress_percent}%{Style.RESET_ALL}")
            sys.stdout.flush()
            
        # Get partial response up to current token
        partial_response = " ".join(tokens[:i+1])
        
        # Store the current token
        tracked_tokens.append(tokens[i] if i < len(tokens) else "")
        
        # Inspect the features for this partial response
        context_inspector = goodfire_client.features.inspect(
            messages=[
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": partial_response}
            ],
            model=model_variant
        )
        
        # Get the top activated features from the context inspector
        # According to the docs, we need to use the top() method instead of accessing features directly
        top_features = context_inspector.top(k=100)  # Get more features to ensure we catch all tracked ones
        
        # Extract activations for tracked features
        for feature_label in feature_activations:
            # Find feature in the response if it exists
            activation = 0.0
            for feature_activation in top_features:
                if feature_activation.feature.label == feature_label:
                    activation = feature_activation.activation
                    break
            
            # Add activation value to list
            feature_activations[feature_label].append(activation)
    
    # Clear the progress line
    sys.stdout.write("\r" + " " * 60 + "\r")
    sys.stdout.flush()
    
    # Store the token positions that were sampled
    token_positions = list(indices)
    
    return feature_activations, feature_names, token_positions, tracked_tokens

def display_feature_table(feature_activations: Dict[str, List[float]], 
                         feature_names: Dict[str, str],
                         token_positions: List[int] = None,
                         tracked_tokens: List[str] = None) -> None:
    """
    Display a table of feature activations.
    
    Args:
        feature_activations: Dictionary mapping feature labels to lists of activation values
        feature_names: Dictionary mapping feature labels to feature names
        token_positions: List of token positions that were sampled (optional)
        tracked_tokens: List of tracked tokens (optional)
    """
    # Create DataFrame from feature activations
    data = {}
    for feature_label, activations in feature_activations.items():
        feature_name = feature_names[feature_label]
        
        # Skip if no activations
        if not activations:
            continue
            
        # Calculate statistics
        avg_activation = sum(activations) / len(activations)
        max_activation = max(activations) if activations else 0
        min_activation = min(activations) if activations else 0
        
        # Calculate at which token the max activation occurred
        max_activation_idx = activations.index(max_activation) if activations else 0
        
        # Convert to token position if token_positions is provided
        if token_positions and max_activation_idx < len(token_positions):
            max_token_idx = token_positions[max_activation_idx] + 1  # 1-based indexing for display
        else:
            max_token_idx = max_activation_idx + 1
        
        # Store computed values
        data[feature_name] = {
            'avg': avg_activation,
            'max': max_activation,
            'min': min_activation,
            'max_token': max_token_idx
        }
    
    # Create and display rich table with sorted data (highest avg activation first)
    console = Console()
    table = Table(title="Feature Activation Summary")
    
    table.add_column("Feature", style="cyan")
    table.add_column("Avg Activation", style="magenta")
    table.add_column("Max Activation", style="green")
    table.add_column("Min Activation", style="red")
    table.add_column("Max at Token #", style="yellow")
    
    # Sort features by average activation (descending)
    sorted_features = sorted(data.items(), key=lambda x: x[1]['avg'], reverse=True)
    
    # Add rows for each feature
    for feature_name, stats in sorted_features:
        table.add_row(
            feature_name,
            f"{stats['avg']:.4f}",
            f"{stats['max']:.4f}",
            f"{stats['min']:.4f}",
            f"{stats['max_token']}"
        )
    
    console.print(table)
    
    # Create a token-by-token activation table
    if token_positions and tracked_tokens:
        # Define the feature columns and create a legend
        feature_columns = list(feature_activations.keys())
        
        # Add a legend for feature numbers
        print(f"\n{Fore.YELLOW}Feature Legend:{Style.RESET_ALL}")
        for idx, feature_name in enumerate(feature_columns):
            print(f"{Fore.CYAN}F{idx+1}: {feature_name}{Style.RESET_ALL}")
        
        # Check if we have too many tokens to display in a single table
        max_tokens_per_table = 20
        total_tokens = len(token_positions)
        
        # If we have too many tokens, split into multiple tables
        num_tables = (total_tokens + max_tokens_per_table - 1) // max_tokens_per_table
        
        for table_idx in range(num_tables):
            start_idx = table_idx * max_tokens_per_table
            end_idx = min(start_idx + max_tokens_per_table, total_tokens)
            
            # Create a new table for this group of tokens
            table_title = f"Token-by-Token Feature Activations"
            if num_tables > 1:
                table_title += f" (Group {table_idx + 1}/{num_tables}, Tokens {token_positions[start_idx] + 1}-{token_positions[end_idx - 1] + 1})"
            
            token_table = Table(title=table_title)
            
            # Add column for token number and token value
            token_table.add_column("Token #", style="yellow")
            token_table.add_column("Token", style="white", max_width=20)
            
            # Add column for each feature
            for idx, feature_name in enumerate(feature_columns):
                token_table.add_column(f"F{idx+1}", style="cyan")
            
            # Add rows for each token in this group
            for i in range(start_idx, end_idx):
                token_pos = token_positions[i]
                token_text = tracked_tokens[i]
                
                # Truncate very long tokens for display
                if len(token_text) > 20:
                    token_text = token_text[:17] + "..."
                
                row_values = [str(token_pos + 1), token_text]
                
                # Add activation value for each feature
                for feature_name in feature_columns:
                    activations = feature_activations[feature_name]
                    if i < len(activations):
                        # Color code the activation values
                        value = activations[i]
                        if value >= 2.0:
                            row_values.append(f"[bold green]{value:.4f}[/bold green]")
                        elif value >= 1.0:
                            row_values.append(f"[green]{value:.4f}[/green]")
                        elif value > 0:
                            row_values.append(f"[blue]{value:.4f}[/blue]")
                        else:
                            row_values.append(f"{value:.4f}")
                    else:
                        row_values.append("N/A")
                
                token_table.add_row(*row_values)
            
            console.print(token_table) 