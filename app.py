import streamlit as st
from paddleocr import PaddleOCR
from PIL import Image
from utils import *

@st.cache(allow_output_mutation=True)
def load_model():
    ocr = PaddleOCR(use_gpu=False, use_xpu=False, ir_optim=True, use_tensorrt=False, min_subgraph_size=15, 
    shape_info_filename=None, precision='fp32', gpu_mem=500, image_dir=None, det_algorithm='DB', 
    det_model_dir='/code/models/det/en_PP-OCRv3_det_infer/', 
    det_limit_side_len=3840, det_limit_type='max', det_db_thresh=0.3, det_db_box_thresh=0.6, det_db_unclip_ratio=1.5, max_batch_size=10, 
    use_dilation=False, det_db_score_mode='slow', det_east_score_thresh=0.8, det_east_cover_thresh=0.1, det_east_nms_thresh=0.2, det_sast_score_thresh=0.5, 
    det_sast_nms_thresh=0.2, det_sast_polygon=False, det_pse_thresh=0, det_pse_box_thresh=0.85, det_pse_min_area=16, det_pse_box_type='box', 
    det_pse_scale=1, scales=[8, 16, 32], alpha=1.0, beta=1.0, fourier_degree=5, det_fce_box_type='box', rec_algorithm='SVTR_LCNet', 
    rec_model_dir='/code/models/rec/en_PP-OCRv3_rec_infer/', 
    rec_image_shape='3, 48, 320', rec_batch_num=6, max_text_length=25, 
    #rec_char_dict_path='/Users/hmansour/opt/miniconda3/lib/python3.9/site-packages/paddleocr/ppocr/utils/en_dict.txt', 
    use_space_char=True, vis_font_path='simfang.ttf', drop_score=0.1, e2e_algorithm='PGNet', e2e_model_dir=None, 
    e2e_limit_side_len=768, e2e_limit_type='max', e2e_pgnet_score_thresh=0.5, e2e_char_dict_path='./ppocr/utils/ic15_dict.txt', 
    e2e_pgnet_valid_set='totaltext', e2e_pgnet_mode='fast', use_angle_cls=False, 
    cls_model_dir='/code/models/cls/ch_ppocr_mobile_v2.0_cls_infer', 
    cls_image_shape='3, 48, 192', label_list=['0', '180'], cls_batch_num=6, cls_thresh=0.9, enable_mkldnn=False, cpu_threads=10, 
    use_pdserving=False, warmup=False, sr_model_dir=None, sr_image_shape='3, 32, 128', sr_batch_num=1, 
    draw_img_save_dir='inference_results', save_crop_res=False, crop_res_save_dir='./output2', use_mp=False, 
    total_process_num=1, process_id=0, benchmark=False, save_log_path='./log_output/', show_log=True, use_onnx=False, 
    output='./output', table_max_len=488, table_algorithm='TableAttn', 
    #table_model_dir='/Users/hmansour/.paddleocr/whl/table/en_ppstructure_mobile_v2.0_SLANet_infer', 
    merge_no_span_structure=True, 
    #table_char_dict_path='/Users/hmansour/opt/miniconda3/lib/python3.9/site-packages/paddleocr/ppocr/utils/dict/table_structure_dict.txt', 
    #layout_model_dir='/Users/hmansour/.paddleocr/whl/layout/picodet_lcnet_x1_0_fgd_layout_infer', 
    #layout_dict_path='/Users/hmansour/opt/miniconda3/lib/python3.9/site-packages/paddleocr/ppocr/utils/dict/layout_dict/layout_publaynet_dict.txt', 
    layout_score_threshold=0, layout_nms_threshold=0, kie_algorithm='LayoutXLM', 
    ser_dict_path='../train_data/XFUND/class_list_xfun.txt', ocr_order_method=None, mode='structure', 
    image_orientation=True, layout=False, table=False, ocr=True, recovery=False, save_pdf=False, 
    lang='en', det=True, rec=True, type='ocr', ocr_version='PP-OCRv3', structure_version='PP-Structurev2')

    return ocr


def load_image(image_file, save=True):
    img = Image.open(image_file)
    img_size = len(img.fp.read()) / 1000 / 1000
    img = img.convert("RGB")
    if img_size > 10 :
        st.write("Error, Image size is more than 10 MB, please choose smaller image")
        return "STOP"

    if save:
        img.save("test.png")
    
    return img

st.set_page_config(
    page_title="POC for OCR text detection and recognition", page_icon=":speaking_head_in_silhouette:", layout="wide")

# Now let’s widen our app layout with this handy CSS hack! Let’s set the width to 1200 pixels:
def _max_width_():
    max_width_str = f"max-width: 1200px;"
    st.markdown(
        f"""
    <style>
    .reportview-container .main .block-container{{
        {max_width_str}
    }}
    </style>
    """,
        unsafe_allow_html=True,
    )

st.title("OCR demo")
_max_width_()
tab1, tab2 = st.tabs(["Upload", "Result"])
button = False

with tab1:
    st.write('Please upload an image with size less than 10 MB')
    image_file = st.file_uploader("Upload Images", type=["png","jpg","jpeg","jp2","bmp"])

    if image_file is not None:
        im = load_image(image_file)
        
        if im != "STOP":

            button = st.button(label="Start Processing")

            if button:
                with st.spinner(text="It may take few seconds, please wait..."):
                    model = load_model()
                    im_lines, im_parag, result = run_ocr(model, "test.png", recognition=True)
                    
                st.write("Done processing! Pleae check the results tab")


with tab2:
    if image_file is not None and button and im != "STOP"  :
        st.write("Lines and paragraphs outputs")
        st.image([im_lines, im_parag], width=600)

        mapper = grep_text_mapper(result)
        boxes = [line[0] for line in result]

        for box in boxes:
            cropped_im, text = grep_textbox(mapper, im, box)
            st.image(cropped_im)
            st.write(text)

