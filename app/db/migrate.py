from app.models.ingestion import create_ingested_files_table
from app.logger import logger

async def run_migrations():
    """Run all declarative table creations/migrations."""
    logger.info("Running database migrations...")
    
    # Run the ingestion table migration
    await create_ingested_files_table()
    
    logger.info("Database migrations complete.")
