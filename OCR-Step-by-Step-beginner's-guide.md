This work has been done by [Ahmed Raafat](https://a-raafat.atlassian.net/wiki/people/5fad0c96048052006b5393d6?ref=confluence) under the supervision of Dr.Hany Ahmed. Here we discuss the steps on how to use and edit PaddleOCR in order to make a working OCR module. We will start first by introducing PaddleOCR and by the end of the guide we will be able to separate all modules and run a demo for each.

**Table of content**

/\*<!\[CDATA\[\*/ div.rbtoc1679300624507 {padding: 0px;} div.rbtoc1679300624507 ul {list-style: disc;margin-left: 0px;} div.rbtoc1679300624507 li {margin-left: 0px;padding-left: 0px;} /\*\]\]>\*/

*   [What is PaddleOCR](#OCRStep-by-Stepbeginner'sguide-WhatisPaddleOCR)
*   [Prerequisites](#OCRStep-by-Stepbeginner'sguide-Prerequisites)
*   [PaddleOCR Installation](#OCRStep-by-Stepbeginner'sguide-PaddleOCRInstallation)
    *   [Sample test on PaddleOCR](#OCRStep-by-Stepbeginner'sguide-SampletestonPaddleOCR)
        *   [Important notes](#OCRStep-by-Stepbeginner'sguide-Importantnotes)
    *   [Sample test on PPStructure](#OCRStep-by-Stepbeginner'sguide-SampletestonPPStructure)
        *   [Important note](#OCRStep-by-Stepbeginner'sguide-Importantnote)
    *   [Models list](#OCRStep-by-Stepbeginner'sguide-Modelslist)
        *   [PaddleOCR models](#OCRStep-by-Stepbeginner'sguide-PaddleOCRmodels)
        *   [PPStructure models](#OCRStep-by-Stepbeginner'sguide-PPStructuremodels)
*   [Separation of models](#OCRStep-by-Stepbeginner'sguide-Separationofmodels)
    *   [Text Detection module](#OCRStep-by-Stepbeginner'sguide-TextDetectionmodule)
        *   [Args](#OCRStep-by-Stepbeginner'sguide-Args)
        *   [Loading the model](#OCRStep-by-Stepbeginner'sguide-Loadingthemodel)
        *   [Model Inference](#OCRStep-by-Stepbeginner'sguide-ModelInference)
    *   [Angle Classifier module](#OCRStep-by-Stepbeginner'sguide-AngleClassifiermodule)
        *   [Args](#OCRStep-by-Stepbeginner'sguide-Args.1)
        *   [Loading the model](#OCRStep-by-Stepbeginner'sguide-Loadingthemodel.1)
        *   [Model Inference](#OCRStep-by-Stepbeginner'sguide-ModelInference.1)
    *   [Text Recognition module](#OCRStep-by-Stepbeginner'sguide-TextRecognitionmodule)
        *   [Args](#OCRStep-by-Stepbeginner'sguide-Args.2)
        *   [Loading the model](#OCRStep-by-Stepbeginner'sguide-Loadingthemodel.2)
        *   [Model Inference](#OCRStep-by-Stepbeginner'sguide-ModelInference.2)
    *   [Layout / Table detection module](#OCRStep-by-Stepbeginner'sguide-Layout/Tabledetectionmodule)
        *   [Args](#OCRStep-by-Stepbeginner'sguide-Args.3)
        *   [Loading the model](#OCRStep-by-Stepbeginner'sguide-Loadingthemodel.3)
        *   [Model Inference](#OCRStep-by-Stepbeginner'sguide-ModelInference.3)
    *   [Table structure recognition module](#OCRStep-by-Stepbeginner'sguide-Tablestructurerecognitionmodule)
        *   [Args](#OCRStep-by-Stepbeginner'sguide-Args.4)
        *   [Loading the model](#OCRStep-by-Stepbeginner'sguide-Loadingthemodel.4)
        *   [Model Inference](#OCRStep-by-Stepbeginner'sguide-ModelInference.4)
    *   [Key Information Extraction module](#OCRStep-by-Stepbeginner'sguide-KeyInformationExtractionmodule)
        *   [Files and folders needed](#OCRStep-by-Stepbeginner'sguide-Filesandfoldersneeded)
        *   [Loading the model](#OCRStep-by-Stepbeginner'sguide-Loadingthemodel.5)
        *   [Model Inference](#OCRStep-by-Stepbeginner'sguide-ModelInference.5)
    *   [Paragraphs module](#OCRStep-by-Stepbeginner'sguide-Paragraphsmodule)
        *   [Algorithm walkthrough](#OCRStep-by-Stepbeginner'sguide-Algorithmwalkthrough)
    *   [Formatter Module](#OCRStep-by-Stepbeginner'sguide-FormatterModule)
        *   [Template schema](#OCRStep-by-Stepbeginner'sguide-Templateschema)
        *   [Rendering to OCR formats](#OCRStep-by-Stepbeginner'sguide-RenderingtoOCRformats)
            *   [HOCR](#OCRStep-by-Stepbeginner'sguide-HOCR)
            *   [ALTO XML](#OCRStep-by-Stepbeginner'sguide-ALTOXML)
            *   [PAGEXML](#OCRStep-by-Stepbeginner'sguide-PAGEXML)
*   [Model Training / Evaluation / Testing](#OCRStep-by-Stepbeginner'sguide-ModelTraining/Evaluation/Testing)
    *   [Yaml config file](#OCRStep-by-Stepbeginner'sguide-Yamlconfigfile)
        *   [Global parameters](#OCRStep-by-Stepbeginner'sguide-Globalparameters)
        *   [Optimizer Parameter](#OCRStep-by-Stepbeginner'sguide-OptimizerParameter)
        *   [Architecture parameters](#OCRStep-by-Stepbeginner'sguide-Architectureparameters)
        *   [Loss parameters](#OCRStep-by-Stepbeginner'sguide-Lossparameters)
        *   [Postprocess parameters](#OCRStep-by-Stepbeginner'sguide-Postprocessparameters)
        *   [Metrics parameters](#OCRStep-by-Stepbeginner'sguide-Metricsparameters)
        *   [Train parameters](#OCRStep-by-Stepbeginner'sguide-Trainparameters)
        *   [Evaluation parameters](#OCRStep-by-Stepbeginner'sguide-Evaluationparameters)
        *   [Testing parameters](#OCRStep-by-Stepbeginner'sguide-Testingparameters)
        *   [Customize everything](#OCRStep-by-Stepbeginner'sguide-Customizeeverything)
    *   [Datasets](#OCRStep-by-Stepbeginner'sguide-Datasets)
        *   [Text detection](#OCRStep-by-Stepbeginner'sguide-Textdetection)
        *   [Text recognition](#OCRStep-by-Stepbeginner'sguide-Textrecognition)
        *   [Angle Classification](#OCRStep-by-Stepbeginner'sguide-AngleClassification)
        *   [Key information extraction](#OCRStep-by-Stepbeginner'sguide-Keyinformationextraction)
*   [Data Annotation and Synthesis](#OCRStep-by-Stepbeginner'sguide-DataAnnotationandSynthesis)
    *   [Data Annotation using PPOCRLabel](#OCRStep-by-Stepbeginner'sguide-DataAnnotationusingPPOCRLabel)
    *   [Text Image Synthesis and augmentation](#OCRStep-by-Stepbeginner'sguide-TextImageSynthesisandaugmentation)

# What is PaddleOCR

PaddleOCR has all the state-of-the-art for OCR tools, which is essential in any end to end OCR module. We will discuss the following tools:

*   Text detection
    
*   Angle classification
    
*   Text recognition
    
*   Layout Analysis
    
*   Table detection and recognition
    
*   Key information extraction
    

# Prerequisites

*   Python
    
*   paddlepaddle
    

# PaddleOCR Installation

In order to be able to install PaddleOCR, you first need to create a conda environment and then clone their repository.

```
conda create -n paddle python=3.9
conda activate paddle

#pip install paddlepaddle #For CPU computing
pip install paddlepaddle-gpu # Or use this for GPU computing

git clone https://github.com/PaddlePaddle/PaddleOCR.git
cd PaddleOCR/
pip install -r requirements.txt
```

## Sample test on PaddleOCR

To check if everything is working properly. please run the python code below. Do not worry about the models dir at the moment, paddle will download the required models based on the `lang` parameter you give.

To check all the other parameters, click [here](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.6/doc/doc_en/inference_args_en.md)

```
from paddleocr import PaddleOCR

ocr = PaddleOCR(use_gpu=False, 
              precision='fp32', 
              det_algorithm='DB', 
              #det_model_dir='', 
              det_limit_side_len=1500, 
              det_limit_type='max', 
              det_db_thresh=0.3, 
              det_db_box_thresh=0.6, 
              det_db_unclip_ratio=1.5,
              det_db_score_mode='slow', 
              rec_algorithm='SVTR_LCNet', 
              #rec_model_dir='', 
              rec_image_shape='3, 48, 320', 
              rec_batch_num=6, 
              max_text_length=25, 
              #rec_char_dict_path='', 
              use_space_char=True, 
              vis_font_path='simfang.ttf', 
              drop_score=0.1, 
              use_angle_cls=True, 
              #cls_model_dir='', 
              cls_image_shape='3, 48, 192', 
              label_list=['0', '180'], 
              cls_batch_num=6, 
              cls_thresh=0.9, 
              enable_mkldnn=False, 
              cpu_threads=10,  
              lang='en', 
              det=True, 
              rec=True, 
              type='ocr', 
              ocr_version='PP-OCRv3')

img_path = "path-to-your-image"

result = ocr.ocr(img_path)
```

The output given will contain the coordinates of each text box detected, its recognition text and its recognition confidence. such as the following output example:

`result = [[[[504.0, 254.0], [707.0, 254.0], [707.0, 299.0], [504.0, 299.0]], ('IMAGE', 0.9968768358230591)]]`

The coordinates are in quad-polygon shape with 2x4 points, top left, top right, bottom right and bottom left. The confidence score given is the recognition score.

### Important notes

*   In order to use another algorithm for detection or recognition, you need to change it in the configs given in the PaddleOCR module you call for example: you want to change the detection algorithm to the best algorithm for detection “DB++” you will do the following changes:  
    `det_algorithm='DB++'`
    
*   But this will not work and give error that DB++ is not recognized. `ppocr ERROR: det_algorithm must in ['DB']`  
    **Solution :** You have to edit the python script paddleocr.py and add this in line 49: `SUPPORT_DET_MODEL = ['DB', "DB++"]`. Now you will be able to use DB++ Or you can add `paddleocr.SUPPORT_DET_MODEL = ['DB', "DB++"]` in your code
    

If you want to draw the output of results as image and check the boxes that got detected, you should run the below script

```
from PIL import Image, ImageDraw
import cv2
import numpy as np

im = cv2.imread("TechSmith-Blog-ExtractText.png")
image = Image.fromarray(im)
result = ocr.ocr(im)

boxes = [line[0] for line in result]
txts = [line[1][0] for line in result]
scores = [line[1][1] for line in result]
im_show = image.copy()

poly = Image.new('RGBA', image.size)
pdraw = ImageDraw.Draw(poly)

for bx in boxes:
    [x1,y1],[x2,y2],[x3,y3],[x4,y4] = bx
    color1 = np.random.randint(0,255)
    color2 = np.random.randint(0,255)
    color3 = np.random.randint(0,255)
    cup_poly = [(x1,y1),(x2,y2),(x3,y3),(x4,y4)]
    pdraw.polygon(cup_poly,
              fill=(color1,color2,color3,100),outline=(0,0,0,255))

im_show.paste(poly,mask=poly)
im_show.save('out.jpg')

from IPython.display import Image 
display(im_show)
```

## Sample test on PPStructure

PP-Structure is a document analysis system which aims to complete tasks related to document understanding such as layout analysis and table recognition, and key information extraction.

*   In the layout analysis task, the image first goes through the layout analysis model to divide the image into different areas such as text, table, and figure, and then analyze these areas separately. For example, the table area is sent to the form recognition module for structured recognition, and the text area is sent to the OCR engine for text recognition. Finally, the layout recovery module restores it to a word or pdf file with the same layout as the original image
    
*   In the key information extraction task, the OCR engine is first used to extract the text content, and then the SER (semantic entity recognition) module obtains the semantic entities in the image, and finally the RE (relationship extraction) module obtains the correspondence between the semantic entities, thereby extracting the required key information.
    

This will be elaborated in details in next sections.

You first need to install the recovery requirements in order to be able to visualize the output in excel or docx format.

1.  Go to ppstructure/recovery
    
2.  run `pip install -r requirements.txt`
    

Using the script below, you can use layout analysis module

```
%cd /content/PaddleOCR
import os
import numpy as np
import cv2
from PIL import Image
from paddleocr import PPStructure,draw_structure_result,save_structure_res
from ppstructure.recovery.recovery_to_doc import sorted_layout_boxes, convert_info_docx

table_engine = PPStructure(use_gpu=False, 
                           precision='fp32', 
                           det_algorithm='DB++', 
                           #det_model_dir='',
                           det_limit_side_len=1500, 
                           det_limit_type='max', 
                           det_db_thresh=0.3, 
                           det_db_box_thresh=0.6, 
                           det_db_unclip_ratio=1.5, 
                           det_db_score_mode='fast', 
                           rec_algorithm='SVTR_LCNet', 
                           #rec_model_dir='', 
                           rec_image_shape='3, 48, 320', 
                           rec_batch_num=6, 
                           max_text_length=25,
                           #rec_char_dict_path='',
                           use_space_char=True, 
                           vis_font_path='simfang.ttf', 
                           drop_score=0.5,
                           #cls_model_dir='', 
                           cls_image_shape='3, 48, 192', 
                           label_list=['0', '180'], 
                           cls_batch_num=6, 
                           cls_thresh=0.9, 
                           enable_mkldnn=False, 
                           cpu_threads=10, 
                           table_max_len=488, 
                           table_algorithm='TableAttn', 
                           #table_model_dir='', 
                           merge_no_span_structure=True, 
                           #table_char_dict_path='', 
                           #layout_model_dir='',
                           #layout_dict_path='',
                           layout_score_threshold=0.5, 
                           layout_nms_threshold=0.5,
                           image_orientation=False, 
                           layout=True, 
                           table=True, 
                           ocr=True, 
                           recovery=True, 
                           save_pdf=True, 
                           lang='en', 
                           det=True, 
                           rec=True, 
                           type='ocr', 
                           ocr_version='PP-OCRv3', 
                           structure_version='PP-StructureV2')

save_folder="output"
img_path = 'merge-multiple-tables-excel.png'
img = cv2.imread(img_path)
result = table_engine(img)
```

Now you have the results of the layout analysis, you can display it out and convert it into docx using the following snippet.

```
save_structure_res(result, save_folder,os.path.basename(img_path).split('.')[0])

for line in result:
    line.pop('img')

font_path = 'simfang.ttf' # font provieded in PaddleOCR
image = Image.open(img_path).convert('RGB')
im_show = draw_structure_result(image, result,font_path="StyleText/fonts/en_standard.ttf")
im_show = Image.fromarray(im_show)
from IPython.display import Image 
display(im_show)

im_show.save('result2.jpg')
```

### Important note

*   You will find the downloaded model behaving very bad on tables. So if you want to download a better model for tables, please download it from [here](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.6/ppstructure/docs/models_list_en.md) under the name `picodet_lcnet_x1_0_fgd_layout_table` Or just download the [inference model](https://paddleocr.bj.bcebos.com/ppstructure/models/layout/picodet_lcnet_x1_0_fgd_layout_table_infer.tar). But note that it detects tables only. Also download the [dictionary](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.6/ppocr/utils/dict/layout_dict/layout_table_dict.txt) file, which is the label for the layout model (in this case it is only table).
    
*   After downloading the model and dictionary, write their path in PPStructure object under `layout_model_dir=''` and `layout_dict_path=''`and re-inference.
    
*   After extracting the tar model, make sure you rename the model to inference instead of model, otherwise paddle will re-download the default models again.
    

## Models list

Here you will find all the models you need.

### PaddleOCR models

Models list can be found [here](https://github.com/PaddlePaddle/PaddleOCR/blob/18ddb6d5f9bdc2c1b0aa7f6e399ec0f76119dc87/doc/doc_ch/models_list.md)

We are using the models below:

*   Detection [model](https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_det_infer.tar) : ch\_PP-OCRv3\_det
    
*   Classification [model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_infer.tar) : ch\_ppocr\_mobile\_v2.0\_cls
    
*   Recognition [model](https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_rec_infer.tar) : en\_PP-OCRv3\_rec
    

### PPStructure models

Model list can be found [here](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.6/ppstructure/docs/models_list_en.md)

We are using the models below:

*   Table detection [model](https://paddleocr.bj.bcebos.com/ppstructure/models/layout/picodet_lcnet_x1_0_fgd_layout_table_infer.tar) : picodet\_lcnet\_x1\_0\_fgd\_layout\_table
    
*   Table structure recognition [model](https://paddleocr.bj.bcebos.com/ppstructure/models/slanet/en_ppstructure_mobile_v2.0_SLANet_infer.tar) : en\_ppstructure\_mobile\_v2.0\_SLANet
    

# Separation of models

In this section we will discuss each module in paddle and how to separate it and use it alone without running end to end OCR. We will also discuss how to format paragraph from lines detected using an intelligent rule based algorithm.

First we need to define our folder structure that we will need in any of the separated modules

```
-- Separated_module/
  -- tools/
    -- infer/
      -- predict.py
  -- inference/
    -- en_PP-OCRv3_det_infer/
      -- inference.pdiparams
      -- inference.pdiparams.info
      -- inference.pdmodel
      
-- ppocr/
  -- data/
    -- __init__.py
    -- simple_dataset.py
    -- imgaug/
      -- __init__.py
      -- table_ops.py
      -- operators.py
  -- utils/
    -- logging.py
    -- en_dict.txt
  -- postprocess/
    -- __init__.py
    -- cls_postprocess.py
    -- db_postprocess.py
    -- rec_postprocess.py
    -- table_postprocess.py
    -- picodet_postprocess.py
    
-- utility.py
```

*   **Separated module** is the module we will separate. it will contain 2 main folders.
    
    *   **Inference folder** will contain the model parameters to load and inference
        
    *   **tools** will contain the separated module as a class that we will use to create model object from it
        
*   **ppocr**
    
    *   **data** contains the dataloader in `simple_dataset.py`and preprocessing script inside `imgaug` folder
        
    *   **utils** contains a logging script for letting us know the status and a dictionary containing english characters that will be used by the recognition model
        
    *   **postprocessing** contains the postprocessing python scripts needed by all the modules
        

Note that in the ppocr, the script `__init__.py`contains the imports of the other scripts included in the folder, for example: in post processing the `__init__.py` defines / imports Detection post processing, recognition post processing, table post processing, etc.. So if you want to add a new postprocessing script in the ppocr folder, make sure you unhash it in the `__init__.py`

## Text Detection module

In this part, we will only add the model and prediction script for the detection module, and keep the same folder of `ppocr` in order to use the post processing inside it. You will also find `predict_det.py` in `PaddleOCR/tools`

```
Detection/
-- tools/
  -- infer/
    -- predict_det.py
-- inference/
  -- en_PP-OCRv3_det_infer/
    -- inference.pdiparams
    -- inference.pdiparams.info
    -- inference.pdmodel
```

Inside `predict_det.py` you will find the detection object from class `TextDetector` that you use to predict. By passing it the correct arguments that it needs, you will define your model and it will build everything for you and become ready to predict

### Args

Any module needs arguments such as preprocessing and post-processing, for the detection module, we need the following arguments:

```
class args_det:
    det_model_dir = 'Detection/inference/en_PP-OCRv3_det_infer/'
    det_algorithm = "DB++"
    det_limit_side_len = 3840
    det_limit_type = "max"
    det_db_thresh=0.3
    det_db_box_thresh=0.6
    det_db_unclip_ratio=1.5
    det_db_score_mode='slow'
    det_box_type = 'quad'
    use_dilation=False
    benchmark = False
    use_onnx = False
    precision = "fp32"
    use_gpu = False
    warmup = False
    use_npu = False
    use_xpu = False
    enable_mkldnn = True
    cpu_threads = 8
```

### Loading the model

To load the model and create an object, we use the following.

```
import utility
from Detection.tools.infer.predict_det import TextDetector

def load_model(args, model_type="det"):
    """
    Loads the model
    args:
        args(class): a class with arguments to load the model
    return:
        the model predictor (paddle inference model type), input_tensor, output_tensors, config
    """
    return utility.create_predictor(args, model_type, logging)

det_model = load_model(args_det,  model_type="det") #Outputs the model predictor (paddle inference model type), paddle input_tensor, paddle output_tensors, paddle config
text_detector = TextDetector(args_det, det_model)   #Outputs TextDetector object
```

Now `det_model` contains the loaded model and `text_detector` contains the Detector object

### Model Inference

In order to get the predictions, you can directly call the object and give it the image as input, like the following example:

```
img = np.array(img_val)
dt_boxes, elapse = text_detector(img)

#dt_boxes will have the shape of [no.textboxes, 4, 2]
#elapse is a float number which resembles the time taken for detection
```

Now you can display the boxes on the image as shows previously in section [Sample test on PaddleOCR](#Sample-test-on-PaddleOCR)

The output is `dt_boxes` which consists of boxes containing text (can be a single word or a text line)

**Input** : OpenCV (numpy array) Image

**Output** : boxes (list of boxes and the time elapsed)

## Angle Classifier module

In this module, we will check the angle / direction of the detected boxes, so that if it is upside down, we will detect the angle and rotate it to become a 0 degree angle. You will find `predict_cls.py` in `PaddleOCR/tools`.

```
Classifier/
-- tools/
  -- infer/
    -- predict_cls.py
-- inference/
  -- ch_ppocr_mobile_v2.0_cls_infer/
    -- inference.pdiparams
    -- inference.pdiparams.info
    -- inference.pdmodel
```

Notice it is the same structure as detection module.

Inside `predict_cls.py` you will find the classifier object from class `TextClassifier` that you use to predict. By passing it the correct arguments that it needs, you will define your model and it will build everything for you and become ready to predict

### Args

The arguments for the classifier module will be

```
class args_cls:
    cls_image_shape='3, 48, 192'
    cls_batch_num=6 
    cls_thresh=0.9
    label_list = ['0', '180']
    cls_model_dir = 'Classifier/inference/ch_ppocr_mobile_v2.0_cls_infer'
    enable_mkldnn = True
    cpu_threads = 8
```

Notice here the `label_list = ['0', '180']` this means it predicts the angle of the boxes and outputs one of the 2 labels given, either 0 or 180.

### Loading the model

To load the model and create an object, we use the following which is exactly as before.

```
import utility
from Classifier.tools.infer.predict_cls import TextClassifier

def load_model(args, model_type="det"):
    """
    Loads the model
    args:
        args(class): a class with arguments to load the model
    return:
        the model predictor (paddle inference model type), input_tensor, output_tensors, config
    """
    return utility.create_predictor(args, model_type, logging)

cls_model = load_model(args_cls,  model_type="cls") #Outputs the model predictor (paddle inference model type), paddle input_tensor, paddle output_tensors, paddle config
text_classifier = TextClassifier(args_cls, cls_model)   #Outputs TextClassifier object
```

Now `cls_model` contains the loaded model and `text_classifier` contains the Classifier object

### Model Inference

In order to predict the results, we pass a batch of cropped images to the classifier object, it takes input batch and outputs the corrected image boxes after fixing it by rotating the image using the angle predicted, also outputs the predicted angle and the time elapsed for inference.

Please check this sudo code to understand

```
import utility

ori_im = cv2.imread(img_path)
img_crop_list = [] # A list that will contain the batch of images each image 
                   # Each image in the list contains the text line that got detected
for box in dt_boxes:
  img_crop = utility.get_rotate_crop_image(ori_im, box)
  img_crop_list.append(img_crop)
```

*   **img\_crop\_list** : A list that contains the cropped images (numpy arrays)
    
*   **dt\_boxes :** The detected boxes from the detector module
    
*   **get\_rotate\_crop\_image :**A utility function that takes the original image as input and a box, then outputs the cropped image using the box coordinates given
    

After you have the batch of cropped images in `img_crop_list`, you can directly call the object using the below snippet

```
output = text_classifier(img_crop_list)
```

Now you only have to give the list of cropped images to the classifier object and take the output of corrected images, their angles and the elapsed time.

**Input** : list of cropped images in numpy arrays

**Output :** list of 3 items

*   item 1 : Contains the cropped images after being rotated and corrected.
    
*   item 2 : Contains the predicted angles depending on the label list given
    
*   item 3 : Contains the float value of the elapsed time for angle classification inference
    

## Text Recognition module

In this module, we will do the same as before and add the prediction script and the model. You will find `predict_rec.py` in `PaddleOCR/tools`.

```
Recognition/
-- tools/
  -- infer/
    -- predict_rec.py
-- inference/
  -- en_PP-OCRv3_rec_infer/
    -- inference.pdiparams
    -- inference.pdiparams.info
    -- inference.pdmodel
```

Inside `predict_rec.py` you will find the text recognition object from class `TextRecognizer` that you use to predict. By passing it the correct arguments that it needs, you will define your model and it will build everything for you and become ready to predict.

### Args

The arguments for the recognition module will be

```
class args_rec:
    rec_image_shape = '3, 48, 320'
    rec_batch_num = 6
    rec_algorithm = "SVTR_LCNet"
    rec_char_dict_path = 'ppocr/utils/en_dict.txt'
    rec_model_dir="Recognition/inference/en_PP-OCRv3_rec_infer"
    precision = "fp32"
    use_space_char = True
```

### Loading the model

To load the model and create an object, we use the following.

```
import utility
from Recognition.tools.infer.predict_rec import TextRecognizer

def load_model(args, model_type="det"):
    """
    Loads the model
    args:
        args(class): a class with arguments to load the model
    return:
        the model predictor (paddle inference model type), input_tensor, output_tensors, config
    """
    return utility.create_predictor(args, model_type, logging)

rec_model = load_model(args_rec,  model_type="rec") #Outputs the model predictor (paddle inference model type), paddle input_tensor, paddle output_tensors, paddle config
text_recognizer = TextRecognizer(args_rec, rec_model)   #Outputs TextDetector object
```

Now `rec_model` contains the loaded model and `text_recognizer` contains the recognition model object

### Model Inference

The inference here is straight-forward like the previous modules, you only need to pass the cropped image containing the text to be recognized in an array shape of (1, img.size) where img.size is in (height , width), the output will be the text recognized and the elapsed time.

Note that you can also send the object a batch of cropped images to recognize in a batch, but keep in mind it predicts image by image, it doesn’t parallelize the process. so the cropped images will be recognized one by one.

```
output = text_recognizer(img_cropped_list))
```

**Input :** A list of cropped images that contain text

**Output :** A list of recognition results with the confidence score

## Layout / Table detection module

### Args

### Loading the model

### Model Inference

## Table structure recognition module

### Args

### Loading the model

### Model Inference

## Key Information Extraction module

In this module, we will discuss the different ways that we can perform KIE then explain which way was used.

General KIE methods are based on Named Entity Recognition (NER), but such methods only use text information and ignore location and visual feature information, which leads to limited accuracy. In recent years, most scholars have started to combine mutil-modal features to improve the accuracy of KIE model. The main methods are as follows:

*   Token based methods. These methods refer to the NLP methods such as Bert, which encode the position, vision and other feature information into the multi-modal model, and conduct pre-training on large-scale datasets, so that in downstream tasks, only a small amount of annotation data is required to obtain excellent results. such as VI-LayoutXLM, LayoutXLM, LayoutLMV2
    
*   SDMGR (Spatial Dual-Modality Graph Reasoning) The key information extraction is solved by iteratively propagating messages along graph edges and reasoning the categories of graph nodes.
    
*   End to end based methods: these methods put the existing OCR character recognition and KIE information extraction tasks into a unified network for common learning, and strengthen each other in the learning process. Such as [TRIE](https://arxiv.org/pdf/2005.13118.pdf)
    

We used the VI-LayoutXLM which is Token based method and we trained it for SER while using our 140 categories.

### Files and folders needed

This one is a bit different than other modules explained above, we need to have a yaml configuration file which we will read the parameters from. The other modules explained can be implemented this way too.

*   Yaml configuration file `ser_vi_layoutxlm_xfund_zh.yml`
    

This file contains the Architecture and Postprocessing parameters as below. It takes other parameters but they are only needed in training/evaluation

```
Architecture:
  model_type: kie
  algorithm: &algorithm "LayoutXLM"
  Transform:
  Backbone:
    name: LayoutXLMForSer
    pretrained: True
    checkpoints: ./utils/model #path of the model
    # one of base or vi
    mode: vi
    num_classes: &num_classes 141
    
PostProcess:
  name: VQASerTokenLayoutLMPostProcess
  class_path: &class_path utils/class_list_nanonets.txt
```

*   ppocr folder should include the modeling folder which contains following folders:
    
    *   architectures
        
    *   backbones
        
    *   heads
        
    *   transformers
        
    *   transforms
        

These folders are essential in building the transformer architecture and also building the sentencepiece tokenizer which tokenizes the sentences into sub-words.

Also it should contain the postprocessing folder which uses `vqa_token_ser_layoutlm_postprocess.py` that is used in post processing to convert between text-label and text-index.

We have also added a script called `kie_utils.py` which contains some functions

*   `predict_kie`
    

```
def predict_kie(model, im_rb, img_name, txts, boundaries):
    new_label_dict_list = prepare_input_dict(txts, boundaries)

    data={}
    data['image'] = im_rb
    data['label'] = json.dumps(new_label_dict_list)

    im_file = BytesIO(im_rb)
    img_val = Image.open(im_file).convert('RGB')
    img = np.array(img_val)

    output_kie = model(data)[0][0]

    posted = postprocess(output_kie)
    #print(posted)
    img_out, results = draw_kie(img, posted)
    
    final_values = {"img":img_name}
    final_values["results"] = results

    df = create_df(final_values)

    return final_values, img_out, df
```

This is the main predict function that takes the output of the detector and recognition results as input and then it outputs for us the dataframe and the image that demonstrates the output.

Firstly, we prepare the input for the model using the following function

```
def prepare_input_dict(txts, boundaries):
    label_dict_list = []
    for text, points in zip(txts, boundaries):
        label_dict = {}
        label_dict["transcription"] = text[0]
        label_dict["points"] = points.tolist()
        label_dict["difficult"] = False
        label_dict["label"] = "other"

        label_dict_list.append(label_dict)
    
    return divide_sent2words(label_dict_list)
```

This function prepares the input for the model, we give it most importantly the transcription and the points, other attributes as difficult and label are not important.

Before returning this list of dictionaries, we pass it to a tokenizer made by us to divide the sentence into words, and also divide the co-ordinates to be fit for each word. This is done using the function `divide_sent2words`

```
def divide_sent2words(label_dict_list):
        
    new_label_dict_list = []
    for label_dict in label_dict_list:
        words_label_list = []

        text_line = label_dict["transcription"]
        text_line_len = len(text_line)

        num_words = len(text_line.split())
        text_words_list = text_line.split()

        if num_words == 1: #only 1 word
            points = fix_points(label_dict['points'])
            label_dict['orig_points'] = points
            new_label_dict_list.append(label_dict)

            continue
```

We first get the transcription and count the number of words. and check if the number of words is 1, then just append the whole dictionary without any tokenization and return it.

```
        points = fix_points(label_dict["points"])
        width_line = get_width(points)
        width_per_char = np.ceil(width_line/text_line_len)

        global_chars_counter = 0

        for text_word in text_words_list:
            last_word = False
            word_dict = copy.deepcopy(label_dict)
            word_dict['transcription'] = text_word
            word_dict['orig_points'] = points

            word_len = len(text_word)
            width_needed = word_len * width_per_char
            start_point = global_chars_counter * width_per_char

            if text_word == text_words_list[-1]:
                last_word = True
                
            new_points = get_new_points(points, width_needed, start_point, width_per_char, last_word)
            word_dict['points'] = new_points
            words_label_list.append(word_dict)
            global_chars_counter+=len(text_word)
        
        for word_label in words_label_list:
            new_label_dict_list.append(word_label)

    return new_label_dict_list
```

First we fix the points, because some points came with a bit of orientation (we basically just take the maximum of all points), then we compute the width of the line using its x-axis and divide it by the number of characters to get the width per character (this means we have **assumption** that every character has somewhat same width).

After that we loop over every word, compute its characters length then compute the width needed so we can trim the coordinates. The trimming is done using the function `get_new_points`.

We also keep the original points information with us to know the position of this word in which sentence and it will have key of `orig_points`

At this point, we have our data ready to be passed to the model as input.

We take the model output and do some postprocessing and we draw the output on the image

In the post processing function we:

*   Remove some unused keys.
    
*   Loop over every token and compute the height intersection with the previous token
    
    *   If the tokens have same label and also have height intersection, we concatenate them together. else we just put it in the dictionary alone (CURRENTLY NEEDS IMPROVEMENTS)
        

Then after the post processing and merging similar tokens together, we draw the output on the image using `draw_kie`

```
def draw_kie(img, output_kie):

    out_list=[]
    new_label_list=[]
    for label in output_kie:
        dict_label={}
        if "VALUE" in label['pred']:

            new_label_list.append(label)
            LABEL = label['pred'].split("VALUE")[0][:-1]
            POINTS = label["points"]
            TEXT = label["transcription"]

            dict_label["label"] = LABEL
            dict_label["text"] = TEXT
            dict_label["boundary"] = POINTS

            out_list.append(dict_label)
 
    img_res = draw_ser_results(img, new_label_list)
    im_out = Image.fromarray(img_res)
    
    return im_out, out_list
```

We loop over the output tokens, and take the points and transcription and build up a list of dictionaries, then we pass this list to a function named `draw_ser_results` that will draw the elements in the list on the image given.

### Loading the model

In order to load the model we first load the configuration from yaml file

```
def load_config(file_path):
    """
    Load config from yml/yaml file.
    Args:
        file_path (str): Path of the config file to be loaded.
    Returns: global config
    """
    _, ext = os.path.splitext(file_path)
    assert ext in ['.yml', '.yaml'], "only support yaml files for now"
    config = yaml.load(open(file_path, 'rb'), Loader=yaml.Loader)
    return config
```

Then we load the SER predictor object using the Yaml configs

```
from infer_kie_token_ser import SerPredictor

kie_config = load_config(kie_config_path)
kie_model = SerPredictor(kie_config)

output_kie = kie_model(data)[0][0] #data should be prepared with the format mentioned
```

### Model Inference

As mentioned before we all have to do is create the object then call it to inference on the data after being prepared in a list of dictionaries.

then we create the dataframe to show the output using the function `create_df` which simply just adds the data extracted into a dataframe to be then transformed into a csv file.

## Paragraphs module

This module is an intelligent rule based algorithm. It takes the boxes that are detected from the text detector module as input then outputs the paragraphs in a ready layout format for the formatter module.

It consists of some functions, we will go into each one in details.

### Algorithm walkthrough

This is the main function called `get_parag_dict_from_boxes` it takes input of paddle detector output and 2 thresholds and then outputs, we will discuss it step by step:

```
def get_parag_dict_from_boxes(boxes, most_frequent, final_factor):
    '''
    
    Main pipeline function that does all the operations and logic

    Inputs :
            boxes : Boxes that came out from paddleocr with shape Nx4x2
            most_frequent : we get the most frequent height of lines to be able to know the appropriate threshold to cut/end the paragraphs --> max(avg_height,most_freq)
            final_factor : Just a scaling factor for the height of the width intersected boxes
            
    Output :
            paragraphs : A dictionary of paragraphs, each paragraph contains the lines

    '''

    yx_boxes = sort_boxes_yx(boxes)[0:] # Sorting boxes in Y then X
    paragraphs = {}


    i = 0
    p_num = 1
    x_thresh = 0.3
    y_thresh = 0.3

    paragraphs["p1"] = [yx_boxes[0]] # First box in the list to be a a start of paragraph
    yx_boxes.remove(yx_boxes[0]) # Removing the added first box
```

At the above snipped we first take the boxes and sort them on y-axis first then x-axis.

`yx_boxes` : is the list of boxes after sorting

`p_num` : is a paragraph index iterator

`x_thresh` : is the threshold for width intersection after filtration (explained later)

`y_thresh` : is the threshold for height intersection (explained later)

We then create a paragraph dictionary to return it at the end as a dictionary of paragraphs

We start our paragraph with the first box and then remove it from the list of boxes `yx_boxes`and then we are going to loop over them and add them depending on some rules

```
def sort_boxes_yx(boxes):
    '''
    
    This function sorts all the boxes on y-axis first, then x-axis

    Inputs :
            boxes : Boxes that came out from paddleocr with shape Nx4x2

    Output :
            Sorted boxes in shape Nx1x8

    '''

    new_boxes = []
    for box in boxes:
        [x1,y1],[x2,y2],[x3,y3],[x4,y4] = box
        new_boxes.append([x1,y1,x2,y2,x3,y3,x4,y4])

    return sorted(new_boxes, key= lambda x:[x[1], x[0]])
```

Sort boxes just takes the boxes and sorts based on y-axis value first then x-axis value and returns Nx1x8 boxes

```
    while(i < len(yx_boxes)):

        x1,y1,x2,y2,x3,y3,x4,y4 = calculate_parag_points(paragraphs["p"+str(p_num)])
        line_1_deg = math.degrees(math.atan2(y2-y1, x2-x1))
        line_2_deg = math.degrees(math.atan2(y3-y4, x3-x4))
        avg_line_angle = (line_1_deg+line_2_deg)/2
        
        if avg_line_angle > 2 or avg_line_angle < -2 :
            p_num+=1
            paragraphs["p"+str(p_num)] = [yx_boxes[0]]
            yx_boxes.remove(yx_boxes[0])
            continue
```

We start looping over the list of boxes but we do some checks at first.

We calculate the paragraph points using `calculate_parag_points` and then check the angle of the paragraph box (actually it will just contain 1 box so we are just getting the box angle) using math.degrees.

If the box has slight rotation (slanted box) we just consider it as it is a paragraph on its own and do not add any other boxes with it.

*   We increment the paragraph index
    
*   Take the new box from the box list and add it to a new paragraph
    
*   Remove the taken new box from the box list
    

```
def calculate_parag_points(parag):
    '''
    
    Function that takes a paragraph of lines and computes a bigger box that has all the lines inside. i.e 
    computes the paragraph coordinates

    Inputs :   
            parag : A paragraph dictionary values (array of Nx1x8)

    Output :
            minx : Minimum value of the X1 coordinate in all lines
            miny : Minimum value of the Y1 coordinate in all lines
            maxx : Maximum value of the X2 coordinate in all lines
            maxy : Maximum value of the Y2 coordinate in all lines

    '''
    parag_arr = np.array(parag)

    minx1 = np.min(parag_arr[:,0]) # Computing min x1 from all x1 in the paragraph lines 
    miny1 = np.min(parag_arr[:,1])
    maxx2 = np.max(parag_arr[:,2])
    miny2 = np.min(parag_arr[:,3])
    maxx3 = np.max(parag_arr[:,4])
    maxy3 = np.max(parag_arr[:,5])
    minx4 = np.min(parag_arr[:,6])
    maxy4 = np.max(parag_arr[:,7])

    return [minx1, miny1, maxx2, miny2, maxx3, maxy3, minx4, maxy4]
```

Function that takes the minimum and maximum of all points across all the included boxes in the paragraph and creates a paragraph box that includes all of them.

```
        threshold = get_average_height2(paragraphs["p"+str(p_num)], most_frequent)
        nearest_boxes = get_nearest_boxes2(paragraphs["p"+str(p_num)], yx_boxes, threshold)

        if len(nearest_boxes) == 0: # No nearest boxes found
            p_num+=1
            paragraphs["p"+str(p_num)] = [yx_boxes[0]]
            yx_boxes.remove(yx_boxes[0])
            continue
```

Currently the function `get_average_height2` is redundant and we just take the average height or most frequent height that was computed.

Then get the nearest boxes for the current paragraph from the remaining boxes list `yx_boxes` using the most frequent / average height.

If we have no nearest boxes returned, then we just close the paragraph and start a new one using the next box from the boxes list

```
def get_nearest_boxes2(parag, boxes, threshold):

    nearest_boxes = []
    combined_parag = calculate_parag_points(parag) # Computes big box for the paragraph
    
    threshold=threshold
    for box in boxes:
        if box[1] - combined_parag[-1] < threshold and box[1] > combined_parag[1]:#or box[1] - parag[-1][1] < threshold * 2.5:
            nearest_boxes.append(box)
    return nearest_boxes
```

`get_nearest_boxes2` This function will get all the boxes that are near to the paragraph, and here we mean any box that has the y-axis distance between them is less than the threshold

![alt text](https://a-raafat.atlassian.net/wiki/download/attachments/589834/image-20230101-131431.png?api=v2)

The blue box will not be added, while both green boxes will be added to the nearest boxes

```
##########
        try:
                
            parag_box = calculate_parag_points(paragraphs["p"+str(p_num)])
            width_boxes_for_threshold , _= get_width_inter_boxes(nearest_boxes, parag_box, thresh = 0.9)
            
            arr = np.array(width_boxes_for_threshold)
            #print(arr)
            y1s = arr[:,1]
            y2s = arr[:,5]

            heights = y2s - y1s
            heights[heights<0]=0
            avg_height_width_boxes = np.average(heights)
            most_freq_width_boxes= np.bincount(heights.astype(int)).argmax()

            threshold2 = most_freq_width_boxes * final_factor
            nearest_boxes = get_nearest_boxes2(paragraphs["p"+str(p_num)], yx_boxes, threshold2)
        except:
            pass
        ###########
```

![alt text](https://a-raafat.atlassian.net/wiki/download/attachments/589834/image-20230101-183025.png?api=v2)

Try here is used because sometimes the boxes that has width intersection with our paragraph can be more than 1 box. So we get the nearest boxes from them depending on the average / most frequent height.

Then we compute again and find the nearest boxes using the new threshold that came from the width intersected boxes (should be more accurate threshold)

Therefore, we handle this using try-except. But the main idea here is just **filtration** for the nearest boxes that we computed. so we want to get the boxes that have intersection over width from the nearest boxes with the paragraph boxes

![alt text](https://a-raafat.atlassian.net/wiki/download/attachments/589834/image-20230101-132701.png?api=v2)

The bottom right box is the only one that has intersection in width with the paragraph, because it lies between **its** x-axis points. Therefore by doing this, we have filtered the blue and the bottom left green box.

```
def get_width_inter_boxes(nearest_boxes, parag, thresh=0.9):
    '''
    Function that gets all the boxes that has width intersection more than 90% with the current paragraph

    Inputs :
            nearest_boxes : A list of boxes that came from get_nearest_boxes function
            parag : A box coordinate which came from calculate_parag_points function

    Output :
            width_inter_boxes : List of boxes that has width intersection more than 90% with the paragraph
            index : A list of indices :: Currently not used

    '''
    width_inter_boxes = []
    index = []
    for idx, box in enumerate(nearest_boxes):
        if get_width_intersection(box, parag) > thresh:
            width_inter_boxes.append(box)
            index.append(idx)
    return width_inter_boxes, index
```

In this function we loop over every box from the nearest boxes and then check if its IOW (intersection over width) is more than the threshold. We use `get_width_intersection` function.

```
def get_width_intersection(line1,line2):
    '''
    
    This function computes the width intersection over union 

    Inputs : 
            line1 : box of coordinate shape 1x8
            line2 : box of coordinate shape 1x8

    Output :
            Intersection over Width from 2 boxes float value

    '''

    x1= line1[0]
    x2= line1[2]
    x3= line2[0]
    x4= line2[2]
    x_inter1 = max(x1, x3)
    x_inter2 = min(x2, x4)
    width_inter = x_inter2 - x_inter1
    if width_inter<0:
        width_inter=0
    width_box1 = abs(x2 - x1)+0.00001
    width_box2 = abs(x4 - x3)+0.00001
    return max(width_inter/width_box1,width_inter/width_box2) 
```

Intersection over width function, computes the intersection over width value between 2 boxes.

![alt text](https://a-raafat.atlassian.net/wiki/download/attachments/589834/image-20230101-135536.png?api=v2)

Then we divide the width intersection by each the width of the boxes and take the maximum to compute the IOW which will be a value between 0 - 1 which will be then compared to the threshold (0.9 currently).

The 0.9 threshold has been set because we want to make sure that the box is very much included in our paragraph

```
        parag_box = calculate_parag_points(paragraphs["p"+str(p_num)])
        
        unsorted_width_inter_boxes, _ = get_width_inter_boxes(nearest_boxes, parag_box, thresh = 0.9)

        if len(unsorted_width_inter_boxes) == 0: # No nearest boxes found
            p_num+=1
            paragraphs["p"+str(p_num)] = [yx_boxes[0]]
            yx_boxes.remove(yx_boxes[0])
            continue
```

Now we have a different nearest boxes list so we have to re-compute the width intersection boxes list.

If there is no nearest boxes left, close the paragraph and create a new one.

```
        first_nearest_box = sort_for_me(parag_box, unsorted_width_inter_boxes)# sorted(unsorted_width_inter_boxes, key=lambda x : x[5])[0]

        breaker= False
        # check if first box is height intersection with any of the nearest boxes
        for bb in unsorted_width_inter_boxes:
                if bb!= first_nearest_box:                
                    if get_height_intersection(bb,first_nearest_box) > y_thresh:
                        breaker = True
                        break

        
```

After that, we take the first nearest box (which is the nearest box to our paragraph, and check weather it has height intersection with any nearest box excluding itself.

Notice that the following scenario will happen.

![alt text](https://a-raafat.atlassian.net/wiki/download/attachments/589834/image-20230103-151052.png?api=v2)

The green boxes that are close to the paragraph will still be included in the nearest boxes list and they might have height intersection in each other, so here we handle this case by the above note. if we found any of the nearest boxes has intersection in height, we immediately break the loop for time optimization. before we explain the next point, we will first mention the intersection in height function

```
def get_height_intersection(line1,line2):
    '''
    
    This function computes the height intersection over union 

    Inputs : 
            line1 : box of coordinate shape 1x8
            line2 : box of coordinate shape 1x8

    Output :
            Intersection over Height from 2 boxes float value

    '''

    y1= line1[1]
    y2= line1[5]
    y3= line2[1]
    y4= line2[5]
    y_inter1 = max(y1, y3)
    y_inter2 = min(y2, y4)
    height_inter = y_inter2 - y_inter1
    if height_inter<0:
        height_inter=0
    height_box1 = abs(y2 - y1) + 0.00001
    height_box2 = abs(y4 - y3) + 0.00001
    return max(height_inter/height_box1,height_inter/height_box2)
```

![alt text](https://a-raafat.atlassian.net/wiki/download/attachments/589834/image-20230103-163051.png?api=v2)

```
        if breaker:
            sorted_width_inter_boxes,_ = get_width_inter_boxes(unsorted_width_inter_boxes, parag_box, thresh = 0.9)
            p_num+=1
            try:
                paragraphs["p"+str(p_num)] = [sorted_width_inter_boxes[0]]
                yx_boxes.remove(sorted_width_inter_boxes[0])
            except:
                paragraphs["p"+str(p_num)] = [first_nearest_box]
                yx_boxes.remove(first_nearest_box)
            continue
```

This part we check if we broke from the loop and found there is a height intersection with any of the nearest boxes. So again we close the paragraph and take the first box (nearest box) for the added paragraph. The try except may be redundant by it was just added as a sanity check. The next part shows if there was no breaking loop happened and the nearest box was actually not intersecting in height with any nearest box

```
        if len(nearest_boxes) == 0: # No nearest boxes found
            p_num+=1
            paragraphs["p"+str(p_num)] = [yx_boxes[0]]
            yx_boxes.remove(yx_boxes[0])
            continue

        
```

if no nearest boxes left, then just close the paragraph

But what is our status here?.

If the code ran till here, then this means we have a stack of boxes that have width intersection with the paragraph and non of them have height intersection with any of each other

```
        elif len(nearest_boxes) >= 1: #  1 box or more found as nearest
            
            parag_box = calculate_parag_points(paragraphs["p"+str(p_num)]) # Calculating the paragraph box
            width_inter = get_width_intersection(parag_box, first_nearest_box) # Checking if this 1 box has Width IOU with parag box

            all_boxes = collect_all_boxes("p"+str(p_num), paragraphs, yx_boxes)
            width_inter_bb_parag,_ = get_width_inter_boxes(all_boxes, parag_box, thresh = x_thresh)
            width_inter_bb_line,_ = get_width_inter_boxes(all_boxes, first_nearest_box, thresh = x_thresh)
```

`collect_all_boxes` is a function to get all the boxes in the image excluding the current paragraph. this means we will take all the remaining boxes from the box list `yx_boxes` and also all the boxes in the paragraphs added (excluding the current one).

Then we get the following:

*   width intersection boxes between the current paragraph and all boxes
    
*   width intersection boxes between the first nearest box and all boxes
    
*   `width_inter` which is the value of IOW of the paragraph and first nearest box
    

```
def collect_all_boxes(current_parag_key, all_parag, remaining_boxes):
    '''
    Function to collect all boxes excluding the current paragraph

    Inputs : 
            current_parag_key : Current key to exclude string
            all_parag : Dictionary containing all paragraphs
            remaining_boxes : a list of the remaining boxes inside the while loop

    Output :
            all_boxes : list of all boxes in the image (combined into paragraph or not)
            
    '''

    all_boxes = []

    for box in remaining_boxes:
        all_boxes.append(box)
    
    for key, val in all_parag.items():
        if key!=current_parag_key:
            box = calculate_parag_points(val)
            all_boxes.append(box)

    return all_boxes
```

This is the function that collects all the boxes excluding the boxes inside our current paragraph. check the following image for illustration.

![alt text](https://a-raafat.atlassian.net/wiki/download/attachments/589834/image-20230103-215032.png?api=v2)

```
            new_parag_from_parag = False
            new_parag_from_line = False
            
            try:
                    
                width_inter_bb_line.remove(first_nearest_box)
                width_inter_bb_line.remove(parag_box)
                width_inter_bb_parag.remove(first_nearest_box)
                width_inter_bb_parag.remove(parag_box)
            except:
                pass
```

At this state, we have 2 lists:

*   `width_inter_bb_line` Contains the width intersection boxes with first line
    
*   `width_inter_bb_parag` Contains the width intersection boxes with the current paragraph
    

![alt text](https://a-raafat.atlassian.net/wiki/download/attachments/589834/image-20230109-134435.png?api=v2)![alt text](https://a-raafat.atlassian.net/wiki/download/attachments/589834/image-20230109-134702.png?api=v2)

Notice that the box green box got reverted to blue, this is because the width intersection with the first box is less than the threshold (0.9).

Also notice that the paragraph box will be included in the width intersection boxes with the first box, this is why we remove the paragraph box from `width_inter_bb_line` and also remove the first box from `width_inter_bb_parag`

```
              for bb in width_inter_bb_parag:
                if bb!= first_nearest_box:
                        
                    if get_height_intersection(bb,first_nearest_box) > y_thresh :
                        if get_width_intersection(bb, first_nearest_box) > 0.9:
                            new_parag_from_parag = False
                        else:
                            new_parag_from_parag = True
                        break
```

Then we start looping over each list and get the height intersection of every box with the first nearest box.

Remember we are now comparing the width intersection boxes of paragraphs with our first nearest box, This will solve the issues in the image below.

![alt text](https://a-raafat.atlassian.net/wiki/download/attachments/589834/image-20230109-145912.png?api=v2)

In case 1, we might have a box that has intersection in height but doesn't intersect in width, this means they are different columns. Notice that this light blue box came **only** from the width intersection boxes with paragraphs (`width_inter_bb_parag`)

While in case 2, we might have boxes that has intersection in height, but also intersection in width, so in this case we set `new_parag_from_parag` to False, this means we will not create a new paragraph. Otherwise case 1 happened and `new_parag_from_parag` is set to True and we create a new paragraph

```
              for bb in width_inter_bb_line:
                if bb!= first_nearest_box:
                        
                    if get_height_intersection(bb,parag_box) > y_thresh:
                        new_parag_from_line = True
                        break
```

Then we start looping over the 2nd list which is `width_inter_bb_line` and check the height intersection with the current paragraph box, if it exists then we will set the flag `new_parag_from_line` to True and create a new paragraph. This condition solves the issue below.

![alt text](https://a-raafat.atlassian.net/wiki/download/attachments/589834/image-20230109-153033.png?api=v2)

Notice that the paragraph has width intersection with the first nearest box but if we don't handle this case. The nearest box will be included in the paragraph which is incorrect. Therefore, We check the other width intersection boxes with the first nearest box.

Note that the blue box will only be included in the `width_inter_bb_line` because it has width intersection with the first nearest box.

In the case above the flag `new_parag_from_line` will be set to True. which means we will create a new paragraph from the first nearest box

```
            if width_inter > 0.9 and not (new_parag_from_line or new_parag_from_parag) :
                paragraphs["p"+str(p_num)]+= [first_nearest_box]
                yx_boxes.remove(first_nearest_box)
                continue
            else:
                p_num+=1
                paragraphs["p"+str(p_num)] = [first_nearest_box]
                yx_boxes.remove(first_nearest_box)
                continue

    return paragraphs
```

Finally we have a check over 2 conditions:

*   If the width intersection between the paragraph box and first nearest box is more than 0.9 (almost totally included)
    
*   Both the flags `new_parag_from_line` and `new_parag_from_parag` are false (means do not create a new paragraph)
    

If these conditions are met, then add this first nearest box, otherwise close the paragraph and create a new one with the first nearest box and remove the first nearest box from the boxes list.

## Formatter Module

In this module, we take the output from all the previous modules, and create a well known format such as HOCR, ALTO XML and PAGE XML, where we put all the information we could grasp from the image, such as paragraphs and the words in the paragraphs and their coordinates and recognition. We will show an example for each format

### Template schema

In order to create our formats, we will need :

1.  Template for each format
    
2.  An environment to render the template and create the format from the template.
    

The schema is difficult to explain all the templates but we will show the general idea.

You can find all the templates in this [link](https://github.com/mittagessen/kraken/tree/master/kraken/templates), let’s take the HOCR template for example

```
{% macro render_line(line) -%}
		<span class="ocr_line" id="line_{{ line.index }}" title="bbox {{ line.bbox|join(' ') }}; x_bboxes {{ line.cuts|sum(start=[])|map('join', ' ')|join(' ') }}{% if line.boundary -%}; poly {{ line.boundary|sum(start=[])|join(' ') }}{% endif %}">
		{% for segment in line.recognition %}
			<span class="ocrx_word" id="segment_{{ segment.index }}" title="bbox {{ segment.bbox|join(' ') }}; x_confs {{ segment.confidences|join(' ') }}{% if segment.boundary -%}; poly {{ segment.boundary|sum(start=[])|join(' ') }}{% endif %}">{{ segment.text }}</span>
		{% endfor -%}
		</span>
		<br/>
{%- endmacro -%}
```

In this part we are creating a function called `render_line` which does the following:

*   Takes the line as input and writes the `<span class`part and then writes the index of this line
    
*   Writes the bounding box and polygon boundary for this line
    
*   Loops over every word inside this line and write its coordinates, confidence and the actual text
    

After that there is our main loop as follows:

```
<!DOCTYPE html>
<html>
	<head>
		<meta http-equiv="Content-Type" content="text/html; charset=utf-8"/>
		<meta name="ocr-system" content="Beyond Limits Inc."/>
		<meta name="ocr-capabilities" content="ocr_page ocrx_block ocr_line ocrx_word ocrp_poly"/>
		{% if page.scripts %}
		<meta name="ocr-scripts" content="{{ page.scripts|join(' ') }}"/>
		{% endif %}
	</head>
	<body>
		<div class="ocr_page" title="bbox 0 0 {{ page.size|join(' ') }}; image {{ page.name }}" style="writing-mode: {{ page.writing_mode }};">
			{% for entity in page.entities -%}
			{% if entity.type == "region" -%}
			<div class="ocrx_block" id="region_{{ entity.index }}" data-region-type="{{ entity.region_type }}" title="bbox {{ entity.bbox|join(' ') }}; poly {{ entity.boundary|sum(start=[])|join(' ') }}">
				{% for line in entity.lines -%}
				{{ render_line(line) }}
				{% endfor %}
			</div>
			{% else -%}
			{{ render_line(entity) }}
			{% endif -%}
			{% endfor -%}
		</div>
	</body>
</html>
```

This main loop does the following:

*   It writes the first 12 lines then at line 13 it starts looping on the page entities which are our paragraphs
    
*   then checks if the type of paragraph is a region or a table or anything else. in our work they are all currently as region
    
*   If the entity type is region, it writes line 15 then starts looping over every line inside the paragraph and pass the line to the `render_line` function
    

### Rendering to OCR formats

In order to render the template above, we need an environment that takes the template and outputs the format, therefore, we will use Jinja2 environment

#### HOCR

The recognized text is stored in normal text nodes of the HTML file.​

Additional information is given in the properties such as:​

*   Different layout elements such as “ocr\_page”, "ocrx\_block", "ocr\_line", "ocrx\_word"​
    
*   Geometric information for each detected textbox with a bounding box "bbox"​ and polygon “poly”
    
*   Confidence values for each word "x\_confs" followed by the recognized text
    

![alt text](https://a-raafat.atlassian.net/wiki/download/attachments/589834/Screen%20Shot%202022-12-18%20at%206.12.08%20PM.png?api=v2)

#### ALTO XML

*   **<Description>** section contains metadata about the ALTO file itself and processing information on how the file was created.
    
*   **<Layout>** section contains the content information. It is subdivided into <Page> elements.
    
    *   **<TextBlock>** contains the paragraphs information inside the <Page>
        
        *   **<Shape>** Contains the shape information of the paragraphs
            
            *   **<Polygon>** Contains the coordinates of the shape which is polygon
                
        *   **<TextLine>** Contains the line information inside the paragraphs
            
            *   **<Shape>** Contains the shape information of the line
                
                *   **<Polygon>** Contains the coordinates of the shape which is polygon
                    
            *   **<String>** Contains the the information of the textline such as:
                
                *   **CONTENT** Which is the actual recognized text
                    
                *   **HPOS** The horizontal coordinates for the word (Position on X-axis)
                    
                *   **VPOS** The vertical coordinates for the word (Position on Y-axis)
                    
                *   **WIDTH** The width of the word
                    
                *   **HEIGHT** The height of the word
                    
                *   **WC** the confidence of the word
                    

![alt text](https://a-raafat.atlassian.net/wiki/download/attachments/589834/Screen%20Shot%202022-12-18%20at%206.14.55%20PM.png?api=v2)

#### PAGEXML

(Page Analysis and Ground truth Elements) XML format.​

*   **<Metadata>** Some information about the how the file was created
    
*   **<Page>** Contains the information about the page such as file name and the shape of page
    
    *   **<TextRegion>** Contains the region information which is our paragraphs inside the Page
        
        *   **<Coords>** Contains the coordinates information for the paragraph
            
        *   **<TextLine>** Contains the Textline information inside the paragraph
            
            *   **<Coords>** Contains the coordinates information for the line
                
            *   **<Word>** Contains the word information inside the text line
                
                *   **<Coords>** Contains the coordinates information for the word
                    
                *   **<TextEquiv>** Contains the actual text and confidence for the word
                    

![alt text](https://a-raafat.atlassian.net/wiki/download/attachments/589834/Screen%20Shot%202022-12-18%20at%206.15.52%20PM.png?api=v2)

# Model Training / Evaluation / Testing

In order to train any paddle module, you must do the following steps. All modules are trainable using the same way.

First you need to set the configuration file. The configuration files can be found in the`configs` folder

```
configs/
  - cls
  - det
  - e2e
  - kie
  - rec
  - sr
  - table
```

Inside each of these folders you will find the already made configuration for every algorithm/architecture you need to try. The below example explains what is inside the configs yaml file. We will mention the important ones that we need

## Yaml config file

You can check all the parameters from [here](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.6/doc/doc_en/config_en.md)

### Global parameters

Here we define the global parameters that the model will need

```
Global:
  use_gpu: true                                # if you have compiled paddle with cpu only,
                                               # turn this to false
  
  epoch_num: 500                               # epoch numbers
  
  print_batch_step: 10                         # print the loss every 10 iterations
  
  save_model_dir: ./output/v3_en_mobile        # where to save the trained model
  
  save_epoch_step: 3                           # save the model parameters every 3 epochs
  
  eval_batch_step: [0, 2000]                   # evaluate on dev set every 2000 iterations
  
  cal_metric_during_train: true                # whether to evaluate the metric during 
                                               # the training process
                                               
  pretrained_model:                            # if you want to do fine tuning and 
                                               #load paramters from a model
  
  checkpoints:                                 # Used to load parameters after 
                                               #interruption to continue training
  
  character_dict_path: ppocr/utils/en_dict.txt # Used in recognition, If it is None,
                                               # model can only recognize number 
                                               # and lower letters
                                               
  max_text_length: &max_text_length 25         # Used in recognition, Set the maximum 
                                               # length of text	
                                               
  use_space_char: true                         # Used in recognition, If you want to have
                                               # the space character recognized
```

### Optimizer Parameter

These parameters will be used to adjust the training parameters for the optimization / learning rate

```
Optimizer:
  name: Adam
  beta1: 0.9
  beta2: 0.999
  lr:
    name: Cosine
    learning_rate: 0.001
    warmup_epoch: 5
  regularizer:
    name: L2
    factor: 3.0e-05
```

Optimizers supported : Momentum, Adam, RMSProp

Learning rate decay function supported : Linear, Cosine, Step, Piecewise

Regularizers supported : L1, L2

### Architecture parameters

```
Architecture:
  model_type:                             # This will contain which module such as rec / det
  algorithm:                              # Algorithm name
  Transform:                              # Used in recognition only for now
  Backbone:                               # Define the backbone
  Neck:                                   # Define the neck
  Head:                                   # Define the head
```

*   **Algorithm**
    

This is the algorithm name you are going to use. You can find all the algorithms with all comparisons [here](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.6/doc/doc_en/algorithm_overview_en.md)

*   **Backbone**
    

You can use any backbone you want, or you can even create your own backbone and add it in the backbones folder. You can find all backbones [here](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.6/ppocr/modeling/backbones/__init__.py)

*   **Neck**
    

This is the middle part of the network which is called Neck. You can find all the supported necks [here](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.6/ppocr/modeling/necks/__init__.py)

*   **Head**
    

The last part of the network, All the supported heads are found [here](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.6/ppocr/modeling/heads/__init__.py)

### Loss parameters

You can find all the loss names [here](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.6/ppocr/losses/__init__.py)

```
Loss:
  name: MultiLoss
  loss_config_list:
    - CTCLoss:
    - SARLoss:
```

### Postprocess parameters

All supported postprocess function names can be found [here](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.6/ppocr/postprocess/__init__.py)

```
PostProcess:  
  name: CTCLabelDecode
```

### Metrics parameters

All supported metrics can be found [here](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.6/ppocr/metrics/__init__.py)

```
Metric:
  name: RecMetric
  main_indicator: acc
  ignore_space: False
```

### Train parameters

Here we need to define the training parameters

```
Train:
  dataset:
    name: SimpleDataSet                   # Currently supports SimpleDataset and LMDBDataset
    
    data_dir: ./train_data/               # The base directory path for the images
   
    label_file_list:                      # File containing images path and labels in 
    - ./train_data/train_list.txt         # correct chosen dataset format
    
    transforms:                           # List of methods to transform images and labels
    - DecodeImage:
        img_mode: BGR                     # You can put the parameters or leave it blank
    - RecAug:                             # To use the default parameters
    - MultiLabelEncode:
    - RecResizeImg:
        image_shape: [3, 48, 320]
    - KeepKeys:                           # The dataloader will keep these keys in the dict
        keep_keys:
        - image
        - label_ctc
        - label_sar
        - length
        - valid_ratio
  loader:
    shuffle: true
    batch_size_per_card: 128
    drop_last: true                       # Whether to discard last incomplete mini-batch
    num_workers: 4                        # The number of sub-processes used to load data
```

After you set all the parameters you will need to run the following

```
python3 tools/train.py -c configs/{model_type}/{yaml_file}.yml
```

You can also resume training using

```
python3 tools/train.py -c configs/{model_type}/{yaml_file}.yml \
                       -o Global.checkpoints=./your/trained/model
```

### Evaluation parameters

It will be the same as training parameters but only remove the augmentations and make `drop_last` in loader to be false and change the `label_file_list` to point on the evaluation dataset file

You can then run the following

```
python3 tools/eval.py -c configs/{model_type}/{yaml_file}.yml  \
                      -o Global.checkpoints="{path/to/weights}/best_accuracy" 
```

### Testing parameters

You can add the testing parameters in Global parameters part or you can just call the below script

```
python3 tools/infer_{model_type}.py -c configs/{model_type}/{yaml_file}.yml \
                                    -o Global.infer_img="./doc/imgs_en/img_10.jpg" \
                                       Global.pretrained_model="./output/best_accuracy" \
                                       Global.save_res_path="./saved_output_path"
```

`Global.infer_img` Can be a single image or a folder containing image

### Customize everything

In order to build your own backbone, head, neck, algorithm, metrics … etc Please follow this [guide](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.6/doc/doc_en/add_new_algorithm_en.md)

## Datasets

In order to prepare your dataset, as explained above, you need to prepare the data in either SimpleDataset or IMDBDataset format

### Text detection

The annotation file formats supported by the PaddleOCR text detection algorithm are as follows, separated by "\\t"

```
" Image filename path             Image annotation information encoded by json.dumps"
ch4_test_images/img_61.jpg    [{"transcription": "MASA", "points": [[310, 104], [416, 141], [418, 216], [312, 179]]}]
```

The image annotation after **json.dumps()** encoding is a list containing multiple dictionaries.

The `points` in the dictionary represent the coordinates (x, y) of the four points of the text box, arranged clockwise from the point at the upper left corner.

`transcription` represents the text of the current text box. **When its content is "###" it means that the text box is invalid and will be skipped during training. So transcription here is just to know if there is a text in this box.**

Some dataset links:

*   [ICDAR 2015](https://rrc.cvc.uab.es/?ch=4&com=downloads) original data
    
    *   Paddle format [Train](https://paddleocr.bj.bcebos.com/dataset/train_icdar2015_label.txt)
        
    *   Paddle format [Test](https://paddleocr.bj.bcebos.com/dataset/test_icdar2015_label.txt)
        
*   [ctw1500](https://paddleocr.bj.bcebos.com/dataset/ctw1500.zip)
    
*   [total text](	https://paddleocr.bj.bcebos.com/dataset/total_text.tar)
    

### Text recognition

By default, the image path and image label are split with \\t, if you use other methods to split, it will cause training error

```
" Image file name           Image annotation "

train_data/rec/train/word_001.jpg   JOINT
```

Some dataset links:

*   [ICDAR 2015](https://rrc.cvc.uab.es/?ch=4&com=downloads) original data
    
    *   Paddle format [Train](https://paddleocr.bj.bcebos.com/dataset/train_icdar2015_label.txt)
        
    *   Paddle format [Test](https://paddleocr.bj.bcebos.com/dataset/test_icdar2015_label.txt)
        
*   en benchmark(MJ, SJ, IIIT, SVT, IC03, IC13, IC15, SVTP, and CUTE.) [LINK](https://github.com/clovaai/deep-text-recognition-benchmark#download-lmdb-dataset-for-traininig-and-evaluation-from-here)  
    LMDB format, which can be loaded directly with [lmdb\_dataset.py](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.6/ppocr/data/lmdb_dataset.py)
    

### Angle Classification

By default, the image path and image label are split with `\t`, if you use other methods to split, it will cause training error

0 and 180 indicate that the angle of the image is 0 degrees and 180 degrees, respectively.

```
" Image file name           Image annotation "

train/word_001.jpg   0
train/word_002.jpg   180
```

### Key information extraction

By default, the image path and image label are split with `\t`, if you use other methods to split, it will cause training error

```
" image path                 annotation information "
img_0.jpg   [{"transcription": "Western", "label": "other", "points": [[104, 114], [530, 114], [530, 175], [104, 175]], "id": 1, "linking": []}, {"transcription": "Date:", "label": "question", "points": [[126, 267], [266, 267], [266, 305], [126, 305]], "id": 7, "linking": [[7, 13]]}, {"transcription": "2020.6.15", "label": "answer", "points": [[321, 239], [537, 239], [537, 285], [321, 285]], "id": 13, "linking": [[7, 13]]}]
```

*   **transcription** is the text value
    
*   **label** the classification label
    
*   **points** text box coordinates in 1x8
    
*   **id** transcription index for RE model training
    
*   **linking** a mapping for question to answer \[question\_id, answer\_id\] for RE model training
    

You are also going to need dictionary file `class_list.txt` which will contain the classes types inside

```
OTHER
QUESTION
ANSWER
HEADER
```

In the annotation file, the annotation information of the `label` field of the text line content of each annotation needs to belong to the dictionary content.

The category information in the annotation file is not case sensitive. For example, 'HEADER' and 'header' will be seen as the same category ID.

In the dictionary file, it is recommended to put the `other` category (other textlines that need not be paid attention to can be labeled as `other`) on the first line. When parsing, the category ID of the 'other' category will be resolved to 0, and the textlines predicted as `other` will not be visualized later.

Some dataset links:

*   [wildreciepts](https://paddleocr.bj.bcebos.com/ppstructure/dataset/wildreceipt.tar)
    
*   [FUNSD](https://guillaumejaume.github.io/FUNSD/dataset.zip)
    

For FUNSD you will need to transform the dataset into paddle format using

`ppstructure/kie/tools/trans_funsd_label.py`

# Data Annotation and Synthesis

## Data Annotation using PPOCRLabel

PPOCRLabelv2 is a semi-automatic graphic annotation tool suitable for OCR field. We use the OCR for recognition and detection of the available text boxes. We currently use it only for Key information extraction task.

*   installation
    

`python3 -m pip install paddlepaddle`

`pip3 install PPOCRLabel`

`pip3 install trash-cli`

In order to run it just type in the terminal`PPOCRLabel --kie True`

1.  Click 'Open Dir' in Menu/File to select the folder of the pictures
    
2.  If you do not have the text boxes coordinates or recognition, you can simply click Re-recognition and you will get the text for all the boxes.
    
3.  You can also create a box and choose its label for the KIE task
    
4.  After you are done with the image, you can click check to go to the next image.
    

Finally you export the annotation and you will find a file named `Label.txt` with the same format that we need for Key information extraction task training.

You also get `fileState.txt`which shows which files you clicked check button and confirmed the annotation.

This means you can pass it the `Label.txt` file and it will recognize it for the images and load it.

## Text Image Synthesis and augmentation

There are multiple sources we can use such as:

*   [TRDG](https://github.com/Belval/TextRecognitionDataGenerator)
    
*   [SynthText](https://github.com/ankush-me/SynthText)
    

TRDG gives us almost real-like text images and it is very easy to use. You can generate random text or give it a text to generate it. Also it can generate Handwritten dataset

SynthText can produce very hard images, which means the background may have almost same color as the text that normal human can’t notice.

We also have [BSRGAN](https://github.com/cszn/BSRGAN) for text image annotation
