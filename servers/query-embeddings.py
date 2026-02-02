#!/usr/bin/env python3
"""
MCP Server for querying stored feedback embeddings using semantic search.
Supports similarity search and various filtering options.
"""

import asyncio
import json
import os
import sys
from typing import Any, Optional

import asyncpg
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

# Database configuration from environment
DB_HOST = os.getenv("POSTGRES_HOST", "localhost")
DB_PORT = os.getenv("POSTGRES_PORT", "5432")
DB_NAME = os.getenv("POSTGRES_DB", "feedback_vectors")
DB_USER = os.getenv("POSTGRES_USER", "readonly")
DB_PASSWORD = os.getenv("POSTGRES_PASSWORD", "readonly")

# Ollama configuration
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")


async def get_db_connection():
    """Create and return a database connection."""
    return await asyncpg.connect(
        host=DB_HOST,
        port=DB_PORT,
        database=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD
    )


async def generate_query_embedding(query_text: str) -> list[float]:
    """Generate embedding for a query using Ollama."""
    import aiohttp
    
    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{OLLAMA_HOST}/api/embeddings",
            json={
                "model": EMBEDDING_MODEL,
                "prompt": query_text
            }
        ) as response:
            if response.status != 200:
                raise Exception(f"Ollama API error: {response.status}")
            
            result = await response.json()
            return result["embedding"]

async def semantic_search(
    query: str,
    limit: int = 10,
    min_rating: Optional[int] = None,
    max_rating: Optional[int] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> list[dict[str, Any]]:
    """
    Perform semantic search on feedback embeddings.
    
    Args:
        query: The search query text
        limit: Maximum number of results to return
        min_rating: Minimum rating filter (1-10)
        max_rating: Maximum rating filter (1-10)
        start_date: Start date filter (ISO format)
        end_date: End date filter (ISO format)
    
    Returns:
        List of matching feedback items with similarity scores
    """
    # Generate embedding for the query
    query_embedding = await generate_query_embedding(query)
    
    # Build the SQL query with filters
    sql_parts = [
        "SELECT uuid, title, feedback, rating, published,",
        "  1 - (embedding <=> $1::vector) as similarity",
        "FROM feedback_vectors",
        "WHERE 1=1"
    ]
    
    params = [query_embedding]
    param_idx = 2
    
    if min_rating is not None:
        sql_parts.append(f"  AND rating >= ${param_idx}")
        params.append(min_rating)
        param_idx += 1
    
    if max_rating is not None:
        sql_parts.append(f"  AND rating <= ${param_idx}")
        params.append(max_rating)
        param_idx += 1
    
    if start_date is not None:
        sql_parts.append(f"  AND published >= ${param_idx}::timestamp")
        params.append(start_date)
        param_idx += 1
    
    if end_date is not None:
        sql_parts.append(f"  AND published <= ${param_idx}::timestamp")
        params.append(end_date)
        param_idx += 1
    
    sql_parts.append("ORDER BY similarity DESC")
    sql_parts.append(f"LIMIT ${param_idx}")
    params.append(limit)
    
    sql = "\n".join(sql_parts)
    
    # Execute query
    conn = await get_db_connection()
    try:
        rows = await conn.fetch(sql, *params)
        results = []
        for row in rows:
            results.append({
                "uuid": str(row["uuid"]),
                "title": row["title"],
                "feedback": row["feedback"],
                "rating": row["rating"],
                "published": row["published"].isoformat() if row["published"] else None,
                "similarity": float(row["similarity"])
            })
        return results
    finally:
        await conn.close()


async def get_feedback_stats(
    min_rating: Optional[int] = None,
    max_rating: Optional[int] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> dict[str, Any]:
    """
    Get statistics about the feedback dataset.
    
    Args:
        min_rating: Minimum rating filter
        max_rating: Maximum rating filter
        start_date: Start date filter (ISO format)
        end_date: End date filter (ISO format)
    
    Returns:
        Dictionary with statistics
    """
    sql_parts = [
        "SELECT",
        "  COUNT(*) as total_count,",
        "  AVG(rating) as avg_rating,",
        "  MIN(rating) as min_rating,",
        "  MAX(rating) as max_rating,",
        "  MIN(published) as earliest_date,",
        "  MAX(published) as latest_date",
        "FROM feedback_vectors",
        "WHERE 1=1"
    ]
    
    params = []
    param_idx = 1
    
    if min_rating is not None:
        sql_parts.append(f"  AND rating >= ${param_idx}")
        params.append(min_rating)
        param_idx += 1
    
    if max_rating is not None:
        sql_parts.append(f"  AND rating <= ${param_idx}")
        params.append(max_rating)
        param_idx += 1
    
    if start_date is not None:
        sql_parts.append(f"  AND published >= ${param_idx}::timestamp")
        params.append(start_date)
        param_idx += 1
    
    if end_date is not None:
        sql_parts.append(f"  AND published <= ${param_idx}::timestamp")
        params.append(end_date)
        param_idx += 1
    
    sql = "\n".join(sql_parts)
    
    conn = await get_db_connection()
    try:
        row = await conn.fetchrow(sql, *params)
        return {
            "total_count": row["total_count"],
            "avg_rating": float(row["avg_rating"]) if row["avg_rating"] else None,
            "min_rating": row["min_rating"],
            "max_rating": row["max_rating"],
            "earliest_date": row["earliest_date"].isoformat() if row["earliest_date"] else None,
            "latest_date": row["latest_date"].isoformat() if row["latest_date"] else None
        }
    finally:
        await conn.close()


async def get_rating_distribution(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> list[dict[str, Any]]:
    """
    Get the distribution of ratings.
    
    Args:
        start_date: Start date filter (ISO format)
        end_date: End date filter (ISO format)
    
    Returns:
        List of rating counts
    """
    sql_parts = [
        "SELECT rating, COUNT(*) as count",
        "FROM feedback_vectors",
        "WHERE 1=1"
    ]
    
    params = []
    param_idx = 1
    
    if start_date is not None:
        sql_parts.append(f"  AND published >= ${param_idx}::timestamp")
        params.append(start_date)
        param_idx += 1
    
    if end_date is not None:
        sql_parts.append(f"  AND published <= ${param_idx}::timestamp")
        params.append(end_date)
        param_idx += 1
    
    sql_parts.append("GROUP BY rating")
    sql_parts.append("ORDER BY rating DESC")
    
    sql = "\n".join(sql_parts)

    conn = await get_db_connection()
    try:
        rows = await conn.fetch(sql, *params)
        return [{"rating": row["rating"], "count": row["count"]} for row in rows]
    finally:
        await conn.close()


async def main():
    """Main entry point for the MCP server."""
    server = Server("feedback-query")
    
    @server.list_tools()
    async def list_tools() -> list[Tool]:
        """List available tools."""
        return [
            Tool(
                name="search_feedback",
                description=(
                    "Search feedback using semantic similarity. Finds feedback entries most similar to the query. "
                    "Returns feedback with title, text, rating, date, and similarity score. "
                    "Useful for finding specific topics, themes, or issues mentioned in customer feedback."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Natural language search query (e.g., 'complaints about legroom', 'positive comments about service')"
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Maximum number of results to return (default: 10, max: 100)",
                            "default": 10,
                            "minimum": 1,
                            "maximum": 100
                        },
                        "min_rating": {
                            "type": "integer",
                            "description": "Filter by minimum rating (1-10)",
                            "minimum": 1,
                            "maximum": 10
                        },
                        "max_rating": {
                            "type": "integer",
                            "description": "Filter by maximum rating (1-10)",
                            "minimum": 1,
                            "maximum": 10
                        },
                        "start_date": {
                            "type": "string",
                            "description": "Filter by start date (ISO format: YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS)"
                        },
                        "end_date": {
                            "type": "string",
                            "description": "Filter by end date (ISO format: YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS)"
                        }
                    },
                    "required": ["query"]
                }
            ),
            Tool(
                name="get_feedback_statistics",
                description=(
                    "Get statistical overview of the feedback dataset including count, average rating, "
                    "rating range, and date range. Optionally filter by rating or date range."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "min_rating": {
                            "type": "integer",
                            "description": "Filter by minimum rating (1-10)",
                            "minimum": 1,
                            "maximum": 10
                        },
                        "max_rating": {
                            "type": "integer",
                            "description": "Filter by maximum rating (1-10)",
                            "minimum": 1,
                            "maximum": 10
                        },
                        "start_date": {
                            "type": "string",
                            "description": "Filter by start date (ISO format: YYYY-MM-DD)"
                        },
                        "end_date": {
                            "type": "string",
                            "description": "Filter by end date (ISO format: YYYY-MM-DD)"
                        }
                    }
                }
            ),
            Tool(
                name="get_rating_distribution",
                description=(
                    "Get the distribution of ratings across the feedback dataset. "
                    "Shows how many feedback entries exist for each rating value. "
                    "Optionally filter by date range."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "start_date": {
                            "type": "string",
                            "description": "Filter by start date (ISO format: YYYY-MM-DD)"
                        },
                        "end_date": {
                            "type": "string",
                            "description": "Filter by end date (ISO format: YYYY-MM-DD)"
                        }
                    }
                }
            )
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: Any) -> list[TextContent]:
        """Handle tool calls."""
        try:
            if name == "search_feedback":
                results = await semantic_search(
                    query=arguments["query"],
                    limit=arguments.get("limit", 10),
                    min_rating=arguments.get("min_rating"),
                    max_rating=arguments.get("max_rating"),
                    start_date=arguments.get("start_date"),
                    end_date=arguments.get("end_date")
                )
                return [TextContent(
                    type="text",
                    text=json.dumps(results, indent=2)
                )]
            
            elif name == "get_feedback_statistics":
                stats = await get_feedback_stats(
                    min_rating=arguments.get("min_rating"),
                    max_rating=arguments.get("max_rating"),
                    start_date=arguments.get("start_date"),
                    end_date=arguments.get("end_date")
                )
                return [TextContent(
                    type="text",
                    text=json.dumps(stats, indent=2)
                )]
            
            elif name == "get_rating_distribution":
                distribution = await get_rating_distribution(
                    start_date=arguments.get("start_date"),
                    end_date=arguments.get("end_date")
                )
                return [TextContent(
                    type="text",
                    text=json.dumps(distribution, indent=2)
                )]
            
            else:
                raise ValueError(f"Unknown tool: {name}")
        
        except Exception as e:
            return [TextContent(
                type="text",
                text=f"Error: {str(e)}"
            )]
    
    # Run the server
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options()
        )


if __name__ == "__main__":
    asyncio.run(main())

