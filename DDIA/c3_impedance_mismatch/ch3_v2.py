from sqlalchemy import create_engine, Column, Integer, String, ForeignKey
from sqlalchemy.orm import declarative_base, relationship, sessionmaker
from sqlalchemy.orm import joinedload

engine = create_engine("sqlite:///demo.db", echo=True)

Base = declarative_base()

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True)
    name = Column(String)

    addresses = relationship("Address", back_populates="user")

class Address(Base):
    __tablename__ = "addresses"

    id = Column(Integer, primary_key=True)
    city = Column(String)

    user_id = Column(Integer, ForeignKey("users.id"))

    user = relationship("User", back_populates="addresses")

Base.metadata.create_all(engine)

Session = sessionmaker(bind=engine)
session = Session()

user = User(
    name="Suraj",
    addresses=[
        Address(city="Delhi"),
        Address(city="Mumbai")
    ]
)

session.add(user)
user = User(
    name="Tanuj",
    addresses=[
        Address(city="Delhi"),
        Address(city="Punjab")
    ]
)
session.add(user)
session.commit()

print("lazy loading")
users = session.query(User).all()

for user in users:
    print(user.name,"\n**************\n")
    for addr in user.addresses:
        print(addr.city)
        
        
print("\n\n Eager loading\n")
users = session.query(User)\
    .options(joinedload(User.addresses))\
    .all()
for user in users:
    print(user.name)
    for add in user.addresses:
        print(" ",add.city)