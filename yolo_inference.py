from ultralytics import YOLO


model = YOLO("models/best.pt")  # Load a pretrained YOLOv8n model
print(f"MODEL:\n{model} ")
print("="*80)
results = model.predict("input_videos/full.mp4", save=True)
print(f"first frame:\n{results[0]}")
print("="*80)
for box in results[0].boxes:
    print(f"box: {box}")