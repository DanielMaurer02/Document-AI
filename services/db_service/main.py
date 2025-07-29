from sqlalchemy import create_engine
from sqlmodel import SQLModel, Session
import os

class Database:
    def __init__(self, sqlite_file_name: str = "database.db"):
        # Use data directory for persistence in Docker
        data_dir = "/app/data" if os.path.exists("/app/data") else "."
        db_path = os.path.join(data_dir, sqlite_file_name)
        self._sqlite_url = f"sqlite:///{db_path}"

        connect_args = {"check_same_thread": False}
        self._engine = create_engine(self._sqlite_url, connect_args=connect_args)
        self._create_db_and_tables()


    def _create_db_and_tables(self):
        SQLModel.metadata.create_all(self._engine)

    def get_session(self):
        with Session(self._engine) as session:
            yield session
