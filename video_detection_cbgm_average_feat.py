import argparse
from glob import glob
import telnetlib
from tkinter import image_names

import matplotlib.pyplot as plt
import torch
import torch.utils.data
from PIL import Image
from torchvision.transforms import functional as F

from defaults import get_default_cfg
from models.coat import COAT
from utils.utils import resume_from_ckpt
import cv2
import numpy as np
from utils.km import run_kuhn_munkres

import os

# Create function update_video_frame to update the video frame
def update_video_frame(frame, detections, km_res, query_names):
    for km in km_res:
        x1, y1, x2, y2 = [ int(x) for x in detections[km[1]] ]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.putText(frame, f"{query_names[km[0]]}-{round(km[2],3)}", (x1 + 15, y1 + 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    return frame


def main(args):
    cfg = get_default_cfg()
    if args.cfg_file:
        cfg.merge_from_file(args.cfg_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    device = torch.device(cfg.DEVICE)

    print("Creating model")
    model = COAT(cfg)
    model.to(device)
    model.eval()

    resume_from_ckpt(args.ckpt, model)

    # For each image matching the pattern "query-*.jpg", add the image to the 
    # list query_img and pass it to F.tensor to convert it to a tensor.
    query_paths = [img_path for img_path in glob("demo_imgs/*.jpg")]
    query_names = np.array([img_path.split('/')[1].split('.')[0].split('-')[0]  for img_path in query_paths])
    query_img = np.array([F.to_tensor(Image.open(img_path).convert("RGB")).to(device) for img_path in query_paths])
    
    # For each image on query_img, obtain its dimensions and pass it to a dictionary with keyword
    # "boxes" to the list query_target.
    query_target = np.array([{"boxes": torch.tensor([[0, 0, img.shape[2]-1, img.shape[1]-1]]).to(device)} for img in query_img])

    query_groups = {}
    query_feats = []
    for name in np.unique(query_names):
        query_groups[name] = np.where(np.array(query_names)==name)[0]

        query_all_feat = model(query_img[query_groups[name]], query_target[query_groups[name]])
        query_feats.append(torch.mean(torch.stack(query_all_feat), dim=0))

    print("query feats", query_feats) 

    if args.video:
        # Open the video file.
        cap = cv2.VideoCapture(args.video)
    else:
        # Open Webcam
        cap = cv2.VideoCapture(0)
    n_frame = 1
    # Create a new video file in mp4 format.
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    out = cv2.VideoWriter('demo_imgs/output.mp4', fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))

    # Check if the video is opened successfully.
    if not cap.isOpened():
        raise IOError("Video file not found")
    # For each frame in the video, do the following:
    while(cap.isOpened()):
        print(f"Processing frame {n_frame}")
        # Read the next frame.
        ret, frame = cap.read()
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        # frame = cv2.cvtCol=or(frame, cv2.COLOR_BGR2RGB)
        video_img = [F.to_tensor(frame).to(device)]
        video_output = model(video_img)[0]
        detections = video_output["boxes"]
        gallery_feats = video_output["embeddings"].to(device)

        similarities = []
        for query_feat in query_feats:
            similarities.append(gallery_feats.mm(query_feat.view(-1, 1)).squeeze())

        print("Similarities", similarities, np.unique(query_names))
        similarities = torch.vstack((similarities))
        # print("Similarities", similarities, query_names)
        # If similarities is not empty, keep the maximum value in all the vertical axis.
        # if similarities.shape[0] > 0:
        #     similarities = torch.max(similarities, dim=0)[0]

        n_frame += 1

        graph = []
        for indx_i, pfeat in enumerate(query_feats):
            for indx_j, gfeat in enumerate(gallery_feats):
                graph.append((indx_i, indx_j, (pfeat * gfeat).sum()))
        km_res, max_val = run_kuhn_munkres(graph)
        print("function output", km_res, max_val)

        # # revise the similarity between query person and its matching
        # for indx_i, indx_j, _ in km_res:
        #     # 0 denotes the query roi
        #     if indx_i == 0:
        #         print("Similarities", similarities)
        #         break

        # Update the video frame.
        frame = update_video_frame(frame, detections, km_res, np.unique(query_names))
        # Write the updated frame to the video file.
        out.write(frame)
        cv2.imshow('frame', frame)

        # Release cap and out.
    # After the loop release the cap object
    cap.release()
    # Destroy all the windows
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a person search network.")
    parser.add_argument("--cfg", dest="cfg_file", help="Path to configuration file.")
    parser.add_argument("--ckpt", required=True, help="Path to checkpoint to resume or evaluate.")
    parser.add_argument("--video", help="Path to video to evaluate.")
    parser.add_argument(
        "opts", nargs=argparse.REMAINDER, help="Modify config options using the command-line"
    )
    args = parser.parse_args()
    with torch.no_grad():
        main(args)