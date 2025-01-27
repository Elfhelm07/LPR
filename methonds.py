from models import *
from sqlalchemy import create_engine


class DBLinker:

    def __init__(self, connectionString):
        self.ENGINE = create_engine(connectionString)

        if database_exists(self.ENGINE.url):
            print("Database Exists")
        else:
            create_database(self.ENGINE.url)
            print("created database")

        connection = self.ENGINE.connect()
        BASE.metadata.create_all(bind=self.ENGINE)
        sessionMaker = sessionmaker(bind=self.ENGINE)
        self.SESSION = sessionMaker()

    def newWantedVehicle(self, licensePlate):
        self.SESSION.add(WantedVehicles(licensePlate))
        self.SESSION.commit()

    def removeWantedVehicle(self, licensePlate):
        row_to_delete = (
            self.SESSION.query(WantedVehicles)
            .filter_by(LICENSEPLATE=licensePlate)
            .first()
        )
        self.SESSION.delete(row_to_delete)
        self.SESSION.commit()

    def searchWantedVehicle(self, LicensePlate):

        requestedVehicle = (
            self.SESSION.query(WantedVehicles)
            .filter_by(LICENSEPLATE=LicensePlate)
            .first()
        )

        if requestedVehicle:
            return requestedVehicle
        else:
            return "no match found"

    def recordVehicle(self, timestamp, licensePlate, boundingbox, confidence):
        try:
            self.SESSION.add(VehicleLogs(timestamp, licensePlate, boundingbox, confidence))
            self.SESSION.commit()
        except Exception as e:
            self.SESSION.rollback()
            raise e  # Re-raise the exception to be caught in the main app


# if __name__ == "__main__":
#     linker = DBLinker(
#     "mysql+pymysql://root:2003@localhost:3306/SecureScanAlpha"
# )

#     linker.recordVehicle(1234567890, "newLicensePlate", "1,2,3,4", 100)
#     linker.newWantedVehicle("newLicensePlate")

#     print(linker.searchWantedVehicle("newLicensePlate"))

#     linker.removeWantedVehicle("newLicensePlate")
