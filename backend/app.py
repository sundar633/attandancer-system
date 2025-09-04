import os, base64, pickle
import numpy as np, cv2, face_recognition
from flask import Flask, request
from flask_cors import CORS
from sqlalchemy import create_engine, Column, Integer, String, LargeBinary, DateTime, ForeignKey, func
from sqlalchemy.orm import sessionmaker, declarative_base, relationship

# Config
DATABASE_URL = os.getenv("DATABASE_URL")  # use Supabase connection string
THRESHOLD = float(os.getenv("THRESHOLD", 0.5))
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*")

# Flask app
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": ALLOWED_ORIGINS}})

# DB setup
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

class Student(Base):
    __tablename__ = "students"
    id = Column(Integer, primary_key=True)
    name = Column(String, unique=True)
    encoding = Column(LargeBinary)
    attendance = relationship("Attendance", back_populates="student")

class Attendance(Base):
    __tablename__ = "attendance"
    id = Column(Integer, primary_key=True)
    student_id = Column(Integer, ForeignKey("students.id"))
    marked_at = Column(DateTime, server_default=func.now())
    student = relationship("Student", back_populates="attendance")

def decode_image(image_b64):
    if "," in image_b64:
        image_b64 = image_b64.split(",")[1]
    img_bytes = base64.b64decode(image_b64)
    npimg = np.frombuffer(img_bytes, np.uint8)
    return cv2.imdecode(npimg, cv2.IMREAD_COLOR)

def get_face_encoding(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    boxes = face_recognition.face_locations(rgb)
    if len(boxes) != 1:
        return None
    return face_recognition.face_encodings(rgb, boxes)[0]

@app.route("/health")
def health():
    return {"status":"ok"}

@app.route("/register", methods=["POST"])
def register():
    data = request.get_json()
    name, image = data.get("name"), data.get("image")
    frame = decode_image(image)
    encoding = get_face_encoding(frame)
    if not encoding:
        return {"status":"failed","message":"No face detected"}, 400

    db = SessionLocal()
    if db.query(Student).filter(Student.name==name).first():
        return {"status":"failed","message":"Name exists"}, 400
    student = Student(name=name, encoding=pickle.dumps(encoding))
    db.add(student)
    db.commit()
    db.close()
    return {"status":"success","message":f"Registered {name}"}

@app.route("/mark-attendance", methods=["POST"])
def mark_attendance():
    data = request.get_json()
    frame = decode_image(data.get("image"))
    encoding = get_face_encoding(frame)
    if not encoding:
        return {"status":"failed","message":"No face detected"}, 400

    db = SessionLocal()
    students = db.query(Student).all()
    encodings = [pickle.loads(s.encoding) for s in students]
    distances = face_recognition.face_distance(encodings, encoding)
    idx = np.argmin(distances)

    if distances[idx] < THRESHOLD:
        student = students[idx]
        db.add(Attendance(student_id=student.id))
        db.commit()
        db.close()
        return {"status":"success","message":f"Attendance marked for {student.name}"}
    db.close()
    return {"status":"failed","message":"No match"}, 403

@app.route("/students")
def students():
    db = SessionLocal()
    res = [{"id":s.id,"name":s.name} for s in db.query(Student).all()]
    db.close()
    return res

@app.route("/attendance")
def attendance():
    db = SessionLocal()
    res = [{"id":a.id,"student":a.student.name,"time":a.marked_at.isoformat()} 
           for a in db.query(Attendance).order_by(Attendance.marked_at.desc()).all()]
    db.close()
    return res

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
