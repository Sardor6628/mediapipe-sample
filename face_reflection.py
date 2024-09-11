import mediapipe as mp
import  numpy
import cv2
cap=cv2.VideoCapture(1)


facmesh =mp.solutions.face_mesh
face =facmesh.FaceMesh()