from sqlalchemy import Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy_utils import database_exists, create_database
from sqlalchemy.orm import sessionmaker

BASE = declarative_base()


class WantedVehicles(BASE):

    __tablename__ = "wanted_vehicles"

    UNIQUEID = Column("unique_id", Integer, primary_key=True)
    LICENSEPLATE = Column("license_plate", String(15), nullable=False)

    def __init__(self, licensePlate):
        self.LICENSEPLATE = licensePlate

    def __repr__(self):
        return f"UniqueId: {self.UNIQUEID} | License plate: {self.LICENSEPLATE}"


class VehicleLogs(BASE):

    __tablename__ = "vehicle_logs"

    PASSERID = Column("passer_id", Integer, primary_key=True)
    TIMESTAMPT = Column("time_stamp", String(15))
    LICENSEPLATE = Column("license_plate", String(15))
    BOUNDINGBOX = Column("bounding_box", String(100))  # Increased from 50 to 100
    CONFIDENCE = Column("confidence", Integer)

    def __init__(self, timestamp, licenseplate, boundingbox, confidence):
        self.TIMESTAMPT = timestamp
        self.LICENSEPLATE = licenseplate
        self.BOUNDINGBOX = boundingbox
        self.CONFIDENCE = confidence

    def __repr__(self):
        return f"timestamp: {self.TIMESTAMPT} | licenseplate: {self.LICENSEPLATE} | Bounding box: {self.BOUNDINGBOX} | confidence: {self.CONFIDENCE}"
