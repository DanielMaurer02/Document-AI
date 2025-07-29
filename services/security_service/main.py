from fastapi import HTTPException, Security, status, Depends
from fastapi.security import APIKeyQuery, APIKeyHeader
from sqlmodel import Session, select
from datetime import datetime
from typing import Optional
import logging

from models import ApiKey
from services.db_service.main import Database

# Initialize database
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
db = Database()

api_key_query = APIKeyQuery(name="api-key", auto_error=False)
api_key_header = APIKeyHeader(name="x-api-key", auto_error=False)


def get_db_session():
    """Get database session."""
    with Session(db._engine) as session:
        yield session


def validate_api_key(key: str, session: Session) -> Optional[ApiKey]:
    """Validate API key against database."""
    if not key:
        return None
    
    statement = select(ApiKey).where(ApiKey.key == key, ApiKey.is_active == True)
    api_key = session.exec(statement).first()
    
    if api_key:
        # Update last used timestamp
        api_key.last_used_at = datetime.utcnow()
        session.add(api_key)
        session.commit()
        session.refresh(api_key)
    
    return api_key


def get_api_key(
    api_key_query: str = Security(api_key_query),
    api_key_header: str = Security(api_key_header),
    session: Session = Depends(get_db_session),
) -> str:
    """Retrieve and validate an API key from the query parameters or HTTP header.

    Args:
        api_key_query: The API key passed as a query parameter.
        api_key_header: The API key passed in the HTTP header.
        session: Database session.

    Returns:
        The validated API key.

    Raises:
        HTTPException: If the API key is invalid or missing.
    """
    # Try query parameter first
    if api_key_query:
        api_key = validate_api_key(api_key_query, session)
        if api_key:
            return api_key_query
    
    # Try header
    if api_key_header:
        api_key = validate_api_key(api_key_header, session)
        if api_key:
            return api_key_header
    
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or missing API Key",
    )


def create_api_key(name: str, key: str, session: Session, is_initial: bool = False) -> ApiKey:
    """Create a new API key."""
    api_key = ApiKey(name=name, key=key, is_initial=is_initial)
    session.add(api_key)
    session.commit()
    session.refresh(api_key)
    logger.info(f"Created API key for {name}")
    return api_key


def delete_key(session: Session, api_key: str = "", is_initial: bool = False) -> int:
    """Delete all initial API keys and return count of deleted keys."""
    if is_initial:
        statement = select(ApiKey).where(ApiKey.is_initial == True)
    else:
        statement = select(ApiKey).where(ApiKey.key == api_key)
    initial_keys = session.exec(statement).all()
    logger.info(f"Deleting {len(initial_keys)} initial API key(s) with key: {api_key}")
    
    count = len(initial_keys)
    for key in initial_keys:
        session.delete(key)
    
    session.commit()
    return count

