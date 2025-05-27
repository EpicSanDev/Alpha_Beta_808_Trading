from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, UniqueConstraint
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import datetime

DATABASE_URL = "sqlite:///./trading_web.db"  # Conformément à l'énoncé

Base = declarative_base()

class OpenInterest(Base):
    __tablename__ = "open_interest"

    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String, index=True, nullable=False)
    timestamp = Column(DateTime, index=True, nullable=False)
    value = Column(Float, nullable=False)

    __table_args__ = (UniqueConstraint('symbol', 'timestamp', name='_symbol_timestamp_uc_open_interest'),)

    def __repr__(self):
        return f"<OpenInterest(symbol='{self.symbol}', timestamp='{self.timestamp}', value={self.value})>"

class FundingRate(Base):
    __tablename__ = "funding_rates"

    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String, index=True, nullable=False)
    timestamp = Column(DateTime, index=True, nullable=False)
    value = Column(Float, nullable=False)

    __table_args__ = (UniqueConstraint('symbol', 'timestamp', name='_symbol_timestamp_uc_funding_rates'),)

    def __repr__(self):
        return f"<FundingRate(symbol='{self.symbol}', timestamp='{self.timestamp}', value={self.value})>"

class MarkPrice(Base):
    __tablename__ = "mark_prices"

    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String, index=True, nullable=False)
    timestamp = Column(DateTime, index=True, nullable=False)
    value = Column(Float, nullable=False)

    __table_args__ = (UniqueConstraint('symbol', 'timestamp', name='_symbol_timestamp_uc_mark_prices'),)

    def __repr__(self):
        return f"<MarkPrice(symbol='{self.symbol}', timestamp='{self.timestamp}', value={self.value})>"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False}) # check_same_thread pour SQLite

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def create_db_and_tables():
    Base.metadata.create_all(bind=engine)

if __name__ == "__main__":
    create_db_and_tables()
    print("Base de données et tables créées (si elles n'existaient pas déjà).")
    # Exemple d'utilisation
    # db = SessionLocal()
    # new_oi = OpenInterest(symbol="BTCUSDT", timestamp=datetime.datetime.now(), value=10000.5)
    # db.add(new_oi)
    # db.commit()
    # db.refresh(new_oi)
    # print(f"Nouvel OI ajouté : {new_oi}")
    # db.close()