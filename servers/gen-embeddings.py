#!python3
import os
import mimetypes
import logging
import pandas as pd
import ollama
from fastmcp import FastMCP

rootdir = '/Users/jleong/Feedback-MCP'
logfile = f"{rootdir}/logs/embedding.log"
outfile = f"{rootdir}/data/embedding"

# Configure logging
logging.basicConfig(
    filename= f"{logfile}",
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)
# Suppress noisy library logs
logging.getLogger('httpx').setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

# Initialize FastMCP Server
mcp = FastMCP("Feedback-Embeddings-Appender")

@mcp.tool()
def generate_feedback_embeddings(input_file: str) -> str:
    """
    Generates text embeddings for feedback using Ollama and saves a new CSV.
    
    Args:
        input_file: The absolute path to the CSV file containing 'feedback'.
    Returns:
        A success message with the path to the vectorized file.
    """
    # 1. Validation: Check if file exists
    if not os.path.isfile(input_file):
        logger.error(f"Error: The file '{input_file}' was not found.")
        return f"Error: The file '{input_file}' was not found."

    # 2. Validation: Check if file is a text or CSV file
    mime_type, _ = mimetypes.guess_type(input_file)
    if not (input_file.lower().endswith('.csv') or (mime_type and mime_type.startswith('text/'))):
        logger.error(f"Error: '{input_file}' does not appear to be a CSV file.")
        return f"Error: '{input_file}' does not appear to be a CSV file."

    try:
        # Load the CSV
        df = pd.read_csv(input_file)

        # 3. Ensure required columns exist
        required = ['uuid', 'title', 'feedback', 'rating', 'published']
        if not all(col in df.columns for col in required):
            logger.error(f"Error: CSV missing one or more required columns: {required}")
            return f"Error: CSV missing one or more required columns: {required}"

        logging.info(f"Vectorizing {len(df)} entries using nomic-embed-text...")

        # 4. Define local counter for progress tracking
        total_rows = len(df)
        
        processed_count = 0       
         
        def embed_row(text):
            nonlocal processed_count
            
            # Calls Ollama's local embedding model
            result = ollama.embeddings(model='nomic-embed-text', prompt=str(text))['embedding']
            
            processed_count += 1
            # Log progress every 10% or every 50 rows (whichever is smaller)
            log_interval = max(1, total_rows // 10)
            if processed_count % min(log_interval, 50) == 0 or processed_count == total_rows:
                percent = (processed_count / total_rows) * 100
                logger.info(f"Progress: {processed_count}/{total_rows} ({percent:.1f}%) complete.")
            
            return result
            
        # Apply embedding
        df['embedding'] = df['feedback'].apply(embed_row)
        
        # 5. Construct output path
        original_name = os.path.basename(input_file)
        new_filename = f"vectorized_{original_name}"
        output_path  = f"{rootdir}/data/embedding/{original_name}"

        # 6. Save result
        column_order = ['uuid', 'title', 'feedback', 'rating', 'published', 'embedding']
        df.to_csv(output_path, columns=column_order, index=False)

        logger.info(f"Success! Vectorized file saved at: {output_path}")
        return f"Success! Vectorized file created at: {output_path}"

    except Exception as e:
        logger.error(f"An error occurred during processing: {str(e)}")
        return f"An error occurred during processing: {str(e)}"

if __name__ == "__main__":
    mcp.run()
