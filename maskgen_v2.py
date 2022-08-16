import json
import streamlit as st
import imageio
import numpy as np
import gdown
from pathlib import Path
# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.utils.logger import setup_logger
from detectron2.evaluation.evaluator import DatasetEvaluator
setup_logger()

@st.cache
def download_model_weights():
    path = Path('model_final.pth')
    if not path.is_file():
        print('Downloading model weights.')
        url = 'https://drive.google.com/uc?id=1yStOq7PtfnjiiSD5uTJ3VLxSLQpG1kcH'
        output = 'model_final.pth'
        gdown.download(url, output, quiet=False)
    else:
        print('Model weights already cached.')

download_model_weights()

st.sidebar.header('Selector de umbral')
threshold = st.sidebar.slider(
    '',
    min_value=0.0,
    max_value=1.0,
    value = 0.5
)

# @st.cache
def get_predictor(threshold):
    if 'val' not in DatasetCatalog.list():
        with open('coco_val.json') as json_file:
            val_dict = json.load(json_file)
        DatasetCatalog.register('val', lambda d:val_dict)
    MetadataCatalog.get('val').set(thing_classes=['pineapple'])
    metadata=MetadataCatalog.get('val')

    cfg=get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.WEIGHTS = 'model_final.pth'
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold
    cfg.MODEL.DEVICE='cpu'
    predictor = DefaultPredictor(cfg)
    return predictor

predictor=get_predictor(threshold)
class_dict={0:'pineapple'}

st.image('iguana.png',width=100)
st.title('Contador de piñas')

uploaded_file = st.file_uploader(
    'Upload an image',
    ['png', 'jpg', 'jpeg']
)

amount = ''

if uploaded_file is not None:

    upload = uploaded_file.read()
    image = imageio.imread(upload,pilmode='L')
    image=np.array(image).reshape((image.shape[0],image.shape[1],1))

    # Make prediction and generate stats dataframe
    outputs = predictor(image)
    amount = len(outputs['instances'])
    # Visualization of the prediction
    v = Visualizer(image[:, :, ::-1], metadata = MetadataCatalog.get('val'), instance_mode=ColorMode.IMAGE_BW)
    out_pred = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    pred_image=out_pred.get_image()[:, :, ::-1]
    st.image(pred_image)

if amount != '':
    st.write(f'Hay {amount} piñas en la imagen.')