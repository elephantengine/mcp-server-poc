#!python3
from fastmcp import FastMCP
import pandas as pd
import uuid
import os
import mimetypes
import logging

rootdir = '/Users/jleong/Feedback-MCP'
logfile = f"{rootdir}/logs/add-uuid.log"
outfile = f"{rootdir}/data/uuid"

# Configure logging
logging.basicConfig(
    filename= f"{logfile}",
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastMCP Server
mcp = FastMCP("Feedback-UUID-Prepender")

@mcp.tool()
def process_csv_to_file(file_path: str) -> str:
    """
    Adds UUIDs to a CSV and saves the result as a new file.
    
    Args:
        file_path: The absolute path to the CSV file to process.
    Returns:
        The path to the newly created file.
    """

    logger.info(f"Received request to process: {file_path}.")

    # 1. Validation: Check if file exists
    if not os.path.isfile(file_path):
        logger.error(f"Error: The file '{file_path}' was not found.")
        return f"Error: The file '{file_path}' was not found."

    # 2. Validation: Basic MIME check
    mime_type, _ = mimetypes.guess_type(file_path)
    if not (file_path.lower().endswith('.csv') or (mime_type and mime_type.startswith('text/'))):
        logger.error(f"Error: '{file_path}' is not a valid CSV or text file.")
        return f"Error: '{file_path}' is not a valid CSV or text file."

    try:
        # Load the CSV
        df = pd.read_csv(file_path)
        
        # Generate UUIDs
        df['uuid'] = [str(uuid.uuid4()) for _ in range(len(df))]

        # Reorder columns
        all_cols = ['uuid', 'title', 'feedback', 'rating', 'published']
        existing_cols = [col for col in all_cols if col in df.columns]
        
        # Construct the output file path
        original_name = os.path.basename(file_path)
        output_path  = f"{rootdir}/data/uuid/{original_name}"

        # Save to the new file
        df.to_csv(output_path, columns=existing_cols, index=False)

        # Return the path to the new file
        logger.info(f"Success! Processed file saved at: {output_path}")
        return f"Success! Processed file saved at: {output_path}"

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        return f"An error occurred: {str(e)}"

if __name__ == "__main__":
    mcp.run()
