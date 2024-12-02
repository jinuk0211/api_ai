from qwen import model, procesoor
from fastapi import FastAPI
from keyframe import frame_interval_file
from llama3B import pipe

@app.post("/send-sumamry")
async def send_item():
    frame_interval_file(url)
    txt_list = []
    frame_list=[]
    for i in os.listdir('keyframe'):
      frame_time = i.split('_')[1].split('.')[0]
      messages = [
          {
              "role": "user",
              "content": [
                  {
                      "type": "image",
                      "image": f"keyframe/{i}",
                  },
                  {"type": "text", "text": "Describe this image."},
              ],
          }
      ]

      #  inference
      text = processor.apply_chat_template(
          messages, tokenize=False, add_generation_prompt=True
      )
      image_inputs, video_inputs = process_vision_info(messages)
      inputs = processor(
          text=[text],
          images=im age_inputs,
          videos=video_inputs,
          padding=True,
          return_tensors="pt",
      )
      inputs = inputs.to("cuda")

      # Inference
      generated_ids = model.generate(**inputs, max_new_tokens=128)
      generated_ids_trimmed = [
          out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
      ]
      output_text = processor.batch_decode(
          generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
      )
      print(output_text)
      txt_list.append(output_text)
      frame_list.append(i)

    for txt, frame in zip(txt_list, frame_list):
      f'{txt} : {frame}'
      item_data = {
        "name": frame,  # Use frame name as item
        "description": txt[0], 
        "price": 10.0,
        "quantity": 1
      }
    log_list = [f'{frame_list[i]}: {txt_list[i]}' for i in range(len(frame_list))]

    messages = [
    {"role": "system", "content": "You are chatbot who should summarize frame time and corresponding caption text!"},
    {"role": "user", "content": f"you have to summarize {','.join(log_list)}"},]


    outputs = pipe(
    messages,
    max_new_tokens=256,)


    return {"message": item_data
            "sumamry": outputs}

import os
import cv2
import boto3
import torch
import numpy as np
from Detection.Utils import ResizePadding
from DetectorLoader import TinyYOLOv3_onecls
from PoseEstimateLoader import SPPE_FastPose
from Track.Tracker import Detection, Tracker
from ActionsEstLoader import TSSTG
from detection_utils import *

@app.post("/anomaly")
async def anomaly_txt()
    bucket_name = "sagemaker-ap-northeast-2-340752839813"  # S3 버킷 이름
    file_key = "action.mp4"  # S3 파일 키
    local_video_path = "./input_video.mp4"  # 로컬로 저장할 비디오 경로

    # 모델 초기화
    detect_model = TinyYOLOv3_onecls(384, device="cuda")
    pose_model = SPPE_FastPose("resnet50", 224, 160, device="cuda")
    tracker = Tracker(max_age=30, n_init=3)
    action_model = TSSTG()
    resize_fn = ResizePadding(384, 384)

    # S3 클라이언트 설정 및 비디오 다운로드
    s3 = boto3.client("s3")
    s3.download_file(bucket_name, file_key, local_video_path)

    # 비디오 로드
    cap = cv2.VideoCapture(local_video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    results = []
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        # Preprocess the frame
        frame = preprocess(frame, resize_fn)

        # Detect humans bbox in the frame
        detected = detect_model.detect(frame, need_resize=False, expand_bb=10)

        # Predict each track's bbox with Kalman filter
        tracker.predict()
        for track in tracker.tracks:
            det = torch.tensor([track.to_tlbr().tolist() + [0.5, 1.0, 0.0]], dtype=torch.float32)
            detected = torch.cat([detected, det], dim=0) if detected is not None else det

        detections = []
        if detected is not None:
            # Predict poses
            poses = pose_model.predict(frame, detected[:, 0:4], detected[:, 4])
            detections = [
                Detection(
                    np.array((ps['keypoints'].numpy()[:, 0].min() - 20,
                              ps['keypoints'].numpy()[:, 1].min() - 20,
                              ps['keypoints'].numpy()[:, 0].max() + 20,
                              ps['keypoints'].numpy()[:, 1].max() + 20)),
                    np.concatenate((ps['keypoints'].numpy(), ps['kp_score'].numpy()), axis=1),
                    ps['kp_score'].mean().numpy()
                )
                for ps in poses
            ]

        # Update tracker with detections
        tracker.update(detections)

        # Analyze actions
        for track in tracker.tracks:
            if not track.is_confirmed():
                continue

            if len(track.keypoints_list) == 30:  # Use 30 frames for action prediction
                pts = np.array(track.keypoints_list, dtype=np.float32)
                out = action_model.predict(pts, frame.shape[:2])
                action_name = action_model.class_names[out[0].argmax()]

                # Save frames with "Fall Down" or "Lying Down" actions
                if action_name in ["Fall Down", "Lying Down"]:
                    s3_key = f"frames/frame_{frame_count}_{action_name.replace(' ', '_')}.jpg"
                    s3_link = upload_image_to_s3(s3, bucket_name, s3_key, frame[:, :, ::-1])
                    if s3_link:
                        results.append({"frame_path": s3_link, "action": action_name})

    cap.release()

    # 결과 출력
    print("Results:")
    for result in results:
        print(result)

