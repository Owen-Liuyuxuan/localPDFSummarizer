import streamlit as st
# st.set_page_config(layout='wide')
from PIL import Image
import numpy as np
from PIL import Image
import requests
import fitz 
import io 
from PIL import Image 
import numpy as np
import layoutparser as lp
from transformers import AutoModel, AutoTokenizer
import os
import uuid




@st.cache_resource()
def load_model():
    model_dict = {}
    model_dict['layout'] = lp.EfficientDetLayoutModel("lp://PubLayNet/tf_efficientdet_d0/config")
    model_dict['chat_tokenizer'] = AutoTokenizer.from_pretrained("THUDM/chatglm2-6b", trust_remote_code=True)
    model_dict['chat_model'] = AutoModel.from_pretrained("THUDM/chatglm2-6b-int4", trust_remote_code=True).to(f'cuda:0')
    model_dict['chat_model'].eval()
    return model_dict

def extract_image_text_pdf(pdf_doc):
    extracted_text = ""
    rendered_imgs  = []
    for page_index in range(len(pdf_doc)): 
        page = pdf_doc.load_page(page_index)
        resolution = 300
        pix = page.get_pixmap(matrix=fitz.Matrix(resolution / 72, resolution / 72))
        img = Image.open(io.BytesIO(pix.tobytes()))
        rendered_imgs.append(img)
        text = page.get_text()
        extracted_text += text
    print(len(extracted_text))
    return extracted_text, rendered_imgs
    

@st.cache_data()
def fetch_process_pdf_file_from_url(pdf_url):
    response = requests.get(pdf_url)
    # Generate a random UUID
    random_uuid = str(uuid.uuid4())
    pdf_path = f"{random_uuid}.pdf"
    try:
        with open(pdf_path, 'wb') as f:
            f.write(response.content)
        doc = fitz.open(pdf_path)
        extracted_text, rendered_imgs = extract_image_text_pdf(doc)
        doc.close()
    except Exception as e:
        raise e
    finally:
        os.remove(pdf_path)
    return extracted_text, rendered_imgs

@st.cache_data()
def process_upload_pdf_file(uploaded_file_placeholder):
    random_uuid = str(uuid.uuid4())
    pdf_path = f"{random_uuid}.pdf"
    try:
        with open(pdf_path, "wb") as f:
            f.write(uploaded_file_placeholder.getbuffer())
        doc = fitz.open(pdf_path)
        extracted_text, rendered_imgs = extract_image_text_pdf(doc)
        doc.close()
    except Exception as e:
        raise e
    finally:
        os.remove(pdf_path)
    return extracted_text, rendered_imgs

def detect_and_crop_images(_models, images):
    model = _models['layout']
    output_image_clips = dict(
        figures=[],
        tables=[],
    )
    for page_image in images:
        layout = model.detect(page_image)
        numpy_page_image = np.array(page_image)
        for block in layout:
            block_type = None
            if block.type.lower() == 'figure':
                block_type = 'figures'
            if block.type.lower() == 'table':
                block_type = 'tables'
            if block_type:
                width = int(block.block.x_2) - int(block.block.x_1)
                height = int(block.block.y_2) - int(block.block.y_1)
                if width < 50 or height < 50:
                    continue
                output_image_clips[block_type].append(
                    numpy_page_image[int(block.block.y_1):int(block.block.y_2),
                                     int(block.block.x_1):int(block.block.x_2)]
                )
    return output_image_clips

def process_pdf_text(pdf_text, max_length):
    ## crop the pdf_text
    if len(pdf_text) > max_length- 100:
        text = pdf_text[:max_length-100]
    text += '---- Please summarize the main idea and contribution of the paper, assuming you are an expert in the field. Present the result as a comprehensive report. ----'
    return text

def summarize_text(model_dict, message_placeholder, pdf_text, max_length, top_p, temperature):
    tokenizer = model_dict['chat_tokenizer']
    model = model_dict['chat_model']
    history, past_key_values = st.session_state.history, st.session_state.past_key_values
    processed_texts = process_pdf_text(pdf_text, max_length)
    for response, history, past_key_values in model.stream_chat(tokenizer, processed_texts, history, past_key_values, max_length=max_length, top_p=top_p, temperature=temperature, return_past_key_values=True):
        st.session_state.history = history
        st.session_state.past_key_values = past_key_values
        message_placeholder.markdown(response)
    
    response += " \n"
    st.session_state.history = history
    st.session_state.past_key_values = past_key_values

    translation_promt_text = "请将上述英文文本翻译成中文"
    for cn_response, history, past_key_values in model.stream_chat(tokenizer, translation_promt_text, history, past_key_values, max_length=max_length, top_p=top_p, temperature=temperature, return_past_key_values=True):
        st.session_state.history = history
        st.session_state.past_key_values = past_key_values
        message_placeholder.markdown(response + "\n" + "## CN Translation \n" + cn_response)
    
    st.session_state.history = history
    st.session_state.past_key_values = past_key_values



### Starting Webpage

st.set_page_config(
    page_title="PDF Summarizer",
    page_icon=":robot:",
    layout='wide'
)

model_dictionary = load_model()

st.title("PDF Paper summarizer")

max_length = st.sidebar.slider(
    'max_length', 0, 8192, 4096, step=1
)
top_p = st.sidebar.slider(
    'top_p', 0.0, 1.0, 0.8, step=0.01
)
temperature = st.sidebar.slider(
    'temperature', 0.0, 1.0, 0.8, step=0.01
)

uploaded_file = st.file_uploader("Choose an pdf...", type=["pdf"])
website_link = st.text_area(label="PDF website link",
                           height=30,
                           placeholder="Please either upload a pdf or provide a link to the pdf")
button = st.button("send", key="predict")

if 'history' not in st.session_state:
    st.session_state.history = []

if 'past_key_values' not in st.session_state:
    st.session_state.past_key_values = None

models = load_model()

if button:
    if uploaded_file is not None:
        texts, images = process_upload_pdf_file(uploaded_file)

    elif website_link is not None:
        texts, images = fetch_process_pdf_file_from_url(website_link)

    else:
        st.warning("Please either upload a pdf or provide a link to the pdf")
        raise ValueError("Please either upload a pdf or provide a link to the pdf")

    fig_tab_dict = detect_and_crop_images(models, images)

    with st.expander("Paper Figures"):
        st.image(fig_tab_dict['figures'])
    
    with st.expander("Paper Tables"):
        st.image(fig_tab_dict['tables'])

    with st.expander("Original Paper"):
        st.image(images)

    with st.chat_message(name="paper summarizer", avatar="assistant"):
        message_placeholder = st.empty()

    summarize_text(models, message_placeholder, texts, max_length, top_p, temperature)
