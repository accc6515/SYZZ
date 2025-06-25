# models.py
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

db = SQLAlchemy()

class DetectionRecord(db.Model):
    __tablename__ = 'detection_records' 
    id = db.Column(db.Integer, primary_key=True) 
    video_name = db.Column(db.String(200), nullable=False) 
    similarity_score = db.Column(db.Float, nullable=True) 
    prediction = db.Column(db.String(10), nullable=False) 
    timestamp = db.Column(db.DateTime, default=datetime.utcnow, nullable=False) 

    
    def to_dict(self):
        return {
            "id": self.id,
            "video_name": self.video_name,
            "similarity_score": self.similarity_score,
            "prediction": self.prediction,
            "timestamp": self.timestamp.strftime("%Y-%m-%d %H:%M:%S")
        }

    def __repr__(self): 
        return f'<DetectionRecord {self.id} - 文件名: {self.video_name} - 预测: {self.prediction}>'

class VideoInfo(db.Model):
    __tablename__ = 'video_infos' 
    id = db.Column(db.Integer, primary_key=True) 
    original_filename = db.Column(db.String(255), nullable=False, index=True) 
    processed_filename = db.Column(db.String(255), nullable=False, unique=True) 
    filesize_bytes = db.Column(db.Integer, nullable=True) 
    storage_path_identifier = db.Column(db.String(500), nullable=True) 
    upload_timestamp = db.Column(db.DateTime, nullable=False, default=datetime.utcnow) 

    def __repr__(self): 
        return f'<VideoInfo {self.id} - 原始文件名: {self.original_filename}>'