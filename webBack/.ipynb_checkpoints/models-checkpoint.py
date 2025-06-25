from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

db = SQLAlchemy()

class DetectionRecord(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    video_name = db.Column(db.String(200), nullable=False)
    similarity_score = db.Column(db.Float, nullable=True)
    prediction = db.Column(db.String(10), nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

    def to_dict(self):
        return {
            "id": self.id,
            "video_name": self.video_name,
            "similarity_score": self.similarity_score,
            "prediction": self.prediction,
            "timestamp": self.timestamp.strftime("%Y-%m-%d %H:%M:%S")
        }
