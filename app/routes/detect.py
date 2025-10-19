from fastapi import APIRouter, WebSocket, WebSocketDisconnect, BackgroundTasks
from pydantic import BaseModel
import base64
import numpy as np

from datetime import datetime
from pathlib import Path
from app.services.mediapipe.base_detector import BasePoseDetector
import warnings
from collections import deque

router = APIRouter()

@router.post("/pose_detection")
async def stream_pose_estimation():
    """WebSocket endpoint for streaming pose estimation results"""

    return {"message": "Pose estimation results will be streamed here."}