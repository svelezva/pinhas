import json
import gdown
#import pandas as pd
import streamlit as st
import imageio
import numpy as np
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
    url = 'https://drive.google.com/file/d/1yStOq7PtfnjiiSD5uTJ3VLxSLQpG1kcH/view?usp=sharing'
    # url = 'https://drive.google.com/uc?id=1Ws-cTAJ55ebfo6tA-TBr6k3IHs93Jzed'
    output = 'detectron.RLE.pth'
    gdown.download(url, output, quiet=False)

download_model_weights()

@st.cache
def get_predictor():
    # with open('coco_val.json') as json_file:
    #     val_dict = json.load(json_file)
    if 'val' not in DatasetCatalog.list():
        with open('coco_val.json') as json_file:
          val_dict = json.load(json_file)
        DatasetCatalog.register('val', lambda d:val_dict)
    MetadataCatalog.get('val').set(thing_classes=['pineapple'])
    metadata=MetadataCatalog.get('val')

    cfg=get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.INPUT.MASK_FORMAT='bitmask'
    cfg.MODEL.WEIGHTS = 'detectron_RLE.pth'
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.DEVICE='cpu'
    predictor = DefaultPredictor(cfg)
    return predictor

predictor=get_predictor()
cell_type_dict={0:'pineapple'}

st.image('iguana.png',width=100)
st.title('Segmentacion piñular')

#st.sidebar.header('User input features')
#display_mode = st.sidebar.radio(
#    'Select what you want to see?',
#    options=['All cells', 'Single cell','Cell types']
#)

uploaded_file = st.file_uploader(
    'Upload an image',
    ['png', 'jpg']
)

if uploaded_file is not None:

    upload = uploaded_file.read()
    image = imageio.imread(upload,pilmode='L')
    image=np.array(image).reshape((image.shape[0],image.shape[1],1))

    # Make prediction and generate stats dataframe
    outputs = predictor(image)
    instances=outputs['instances']
    id_centers=[center for center in instances.pred_boxes.get_centers().numpy()]
    areas=[mask.sum() for mask in instances.pred_masks.numpy()]
    scores=instances.scores.numpy()
    cell_types=[cell_type_dict[i] for i in instances.pred_classes.numpy()]
    #stats_df=pd.DataFrame(
    #    list(zip(id_centers,areas,scores,cell_types)),
    #    columns=['center','area','score','cell_type']
    #    )

    # Visualization of the prediction
    v = Visualizer(image[:, :, ::-1], metadata = MetadataCatalog.get('val'), instance_mode=ColorMode.IMAGE_BW)
    if display_mode=='All cells':
        out_pred = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    elif display_mode=='Single cell':
        cell_index=st.sidebar.number_input(
            'Input the index of the cell',0,len(outputs['instances'])-1
        )
        out_pred = v.draw_instance_predictions(outputs["instances"][cell_index].to("cpu"))
    else:
        indexes=list(stats_df.index)
        selected_types=st.sidebar.multiselect(
            'Select the types you want to visualize:',
            ['shsy5y','astro','cort']
            )

        indexes=list(stats_df[stats_df['cell_type'].isin(selected_types)].index)
        out_pred = v.draw_instance_predictions(outputs["instances"][indexes].to("cpu"))
    pred_image=out_pred.get_image()[:, :, ::-1]
    st.image(pred_image)

    # Visualize the dataframes
    st.markdown("""
    The following dataframe displays some stats on the annotations:
    """)
    st.dataframe(stats_df)

    st.markdown("""
    A summary of the data:
    """)
    st.dataframe(stats_df.describe())
