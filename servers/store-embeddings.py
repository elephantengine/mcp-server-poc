#!python3
import os
import logging
import sys
import pandas as pd
import psycopg2
from psycopg2 import extras
from fastmcp import FastMCP

rootdir = '/absolute/path/to/project/dir'
logfile = f"{rootdir}/logs/store-embeddings.log"

# Initialize FastMCP Server
mcp = FastMCP("Feedback-Store-Embeddings")

# Configure logging to stderr to avoid breaking the MCP protocol
logging.basicConfig(
    filename= f"{logfile}",
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@mcp.tool()
def store_feedback_embeddings(input_file: str) -> str:
    """
    Reads a vectorized CSV and stores the records in the PostgreSQL 'feedback' table.
    
    Args:
        input_file: The absolute path to the CSV file containing embeddings.
    """
    # 1. Validation: Check if file exists
    if not os.path.isfile(input_file):
        logger.error(f"Error: The file '{input_file}' was not found.")
        return f"Error: The file '{input_file}' was not found."

    # 2. Database Connection (Uses standard PG environment variables or defaults)
    # Ensure these are set in your claude_desktop_config.json 'env' block
    try:
        conn = psycopg2.connect(
            host=os.getenv("PGHOST", "localhost"),
            database=os.getenv("PGDATABASE", "postgres"),
            user=os.getenv("PGUSER"),
            password=os.getenv("PGPASSWORD"),
            port=os.getenv("PGPORT", "5432")
        )
        cur = conn.cursor()
    except Exception as e:
        logger.error(f"Database connection failed: {str(e)}")        
        return f"Database connection failed: {str(e)}"

    try:
        # 3. Load the vectorized CSV
        df = pd.read_csv(input_file)
        
        # Verify columns
        required = ['uuid', 'title', 'feedback', 'rating', 'published', 'embedding']
        if not all(col in df.columns for col in required):
            logger.error(f"Error: CSV missing required columns: {required}")
            return f"Error: CSV missing required columns: {required}"

        total_rows = len(df)
        logger.info(f"Preparing to ingest {total_rows} records into the database...")

        # 4. Ingest data
        # We use execute_values for high-performance batch insertion
        insert_query = """
            INSERT INTO feedback_vectors (uuid, title, feedback, rating, published, embedding)
            VALUES %s
            ON CONFLICT (uuid) DO NOTHING;
        """

        # Convert dataframe to list of tuples
        data_tuples = [tuple(x) for x in df[required].values]

        extras.execute_values(cur, insert_query, data_tuples)
        conn.commit()

        logger.info(f"Successfully ingested {total_rows} records.")
        return f"Success! {total_rows} records processed and stored in the 'feedback' table."

    except Exception as e:
        conn.rollback()
        logger.error(f"Ingestion failed: {str(e)}")
        return f"An error occurred during storage: {str(e)}"
    
    finally:
        cur.close()
        conn.close()

if __name__ == "__main__":
    mcp.run()%                                                 
