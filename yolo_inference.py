import argparse 
import cv2
import numpy as np 
import os 

import torch 
from torchvision import transforms

from emotic import Emotic 
from inference import infer
from yolo_utils import prepare_yolo, rescale_boxes, non_max_suppression

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0, help='gpu id')
    parser.add_argument('--experiment_path', type=str, default="/Users/kang/Documents/Github/BentoML_serving/EmotionRecognition/", help='Path of experiment files (results, models, logs)')
    parser.add_argument('--model_dir', type=str, default='models', help='Folder to access the models')
    parser.add_argument('--result_dir', type=str, default='results', help='Path to save the results')
    parser.add_argument('--img_file', type=str, help='Test img file')
    # Generate args
    args = parser.parse_args()
    return args


def get_bbox(yolo_model, device, image_context, yolo_image_size=416, conf_thresh=0.8, nms_thresh=0.4):
  ''' Use yolo to obtain bounding box of every person in context image. 
  :param yolo_model: Yolo model to obtain bounding box of every person in context image. 
  :param device: Torch device. Used to send tensors to GPU (if available) for faster processing. 
  :yolo_image_size: Input image size for yolo model. 
  :conf_thresh: Confidence threshold for yolo model. Predictions with object confidence > conf_thresh are returned. 
  :nms_thresh: Non-maximal suppression threshold for yolo model. Predictions with IoU > nms_thresh are returned. 
  :return: Numpy array of bounding boxes. Array shape = (no_of_persons, 4). 
  '''
  test_transform = transforms.Compose([transforms.ToPILImage(),transforms.ToTensor()])
  image_yolo = test_transform(cv2.resize(image_context, (416, 416))).unsqueeze(0).to(device)

  with torch.no_grad():
    detections = yolo_model(image_yolo)
    nms_det  = non_max_suppression(detections, conf_thresh, nms_thresh)[0]
    det = rescale_boxes(nms_det, yolo_image_size, (image_context.shape[:2]))
  
  bboxes = []
  for x1, y1, x2, y2, _, _, cls_pred in det:
    if cls_pred == 0:  # checking if predicted_class = persons. 
      x1 = int(min(image_context.shape[1], max(0, x1)))
      x2 = int(min(image_context.shape[1], max(x1, x2)))
      y1 = int(min(image_context.shape[0], max(15, y1)))
      y2 = int(min(image_context.shape[0], max(y1, y2)))
      bboxes.append([x1, y1, x2, y2])
  return np.array(bboxes)


## image files
def yolo_img(img_file, result_path, model_path, gpu_id, customerId):
  ''' Perform inference on a video. First yolo model is used to obtain bounding boxes of persons in every frame.
  After that the emotic model is used to obtain categoraical and continuous emotion predictions. 
  :param img_file: Path of image file. 
  :param result_path: Directory path to save the results (output video).
  :param model_path: Directory path to load models and val_thresholds to perform inference.
  '''   
  cat = ['affection', 'anger', 'annoyance', 'anticipation', 'aversion', 'confidence', 'disapproval', 'disconnection', 
          'disquietment', 'confusion', 'embarrassment', 'engagement', 'esteem', 'excitement', 'fatigue', 'fear','happiness', 
          'pain', 'peace', 'pleasure', 'sadness', 'sensitivity', 'suffering', 'surprise', 'sympathy', 'yearning']
  
  #보이스피싱 WARNING
  warning_cat = [ 'anger', 'annoyance',  'disapproval', 'disquietment', 'confusion', 'sadness', 'suffering'] 
  
  cat2ind = {}
  ind2cat = {}

  for idx, emotion in enumerate(cat):
      cat2ind[emotion] = idx
      ind2cat[idx] = emotion
  
  vad = ['Valence', 'Arousal', 'Dominance']
  ind2vad = {}
  for idx, continuous in enumerate(vad):
      ind2vad[idx] = continuous
  
  context_mean = [0.4690646, 0.4407227, 0.40508908] 
  context_std = [0.2514227, 0.24312855, 0.24266963] 
  body_mean = [0.43832874, 0.3964344, 0.3706214]
  body_std = [0.24784276, 0.23621225, 0.2323653]
  context_norm = [context_mean, context_std]   #List containing mean and std values for context images. 
  body_norm = [body_mean, body_std]            #List containing mean and std values for body images.

 
  #device = torch.device("cuda:%s" %(str(args.gpu)) if torch.cuda.is_available() else "cpu")
  device = torch.device("cuda:%s" %(str(gpu_id)) if torch.cuda.is_available() else "cpu")

  #YOLO 불러오기
  yolo = prepare_yolo(model_path)
  yolo = yolo.to(device)
  yolo.eval()

  #모델 불러오기
  thresholds = torch.FloatTensor(np.load(os.path.join(result_path, 'val_thresholds.npy'))).to(device) 
  model_context = torch.load(os.path.join(model_path,'model_context1.pth')).to(device)
  model_body = torch.load(os.path.join(model_path,'model_body1.pth')).to(device)
  emotic_model = torch.load(os.path.join(model_path,'model_emotic1.pth')).to(device)

  model_context.eval()
  model_body.eval()
  emotic_model.eval()

  models = [model_context, model_body, emotic_model]

  print ('Starting testing on img')

  image_context = cv2.cvtColor(cv2.imread(img_file), cv2.COLOR_BGR2RGB)  
  warn_num = 0
  nagative_cat = []
  try: 
    bbox_yolo = get_bbox(yolo, device, image_context) 

    for pred_idx, pred_bbox in enumerate(bbox_yolo):
      pred_cat, pred_cont = infer(context_norm, body_norm, ind2cat, ind2vad, device, thresholds, models, image_context=image_context, bbox=pred_bbox, to_print=False)
      # Emotic category
      for i, emotion in enumerate(pred_cat):
        if emotion in warning_cat:
          nagative_cat.append(emotion)
          print(emotion) 
      print(len(pred_cat))
  
  
  except Exception:
      pass

  return { "customerId" : customerId , "emotions" :nagative_cat, "total" : len(pred_cat)}; 



def check_paths(args):
  ''' Check (create if they don't exist) experiment directories.
  :param args: Runtime arguments as passed by the user.
  :return: result_dir_path, model_dir_path.
  ''' 
  if args.img_file is not None: 
    if not os.path.exists(args.img_file):
      raise ValueError('video file does not exist. Please pass a valid video file')
  model_path = os.path.join(args.experiment_path, args.model_dir)
  if not os.path.exists(model_path):
    raise ValueError('model path %s does not exist. Please pass a valid model_path' %(model_path))
  result_path = os.path.join(args.experiment_path, args.result_dir)
  if not os.path.exists(result_path):
    os.makedirs(result_path)
  return result_path, model_path



if __name__=='__main__':
  result_path, model_path = check_paths(args)

