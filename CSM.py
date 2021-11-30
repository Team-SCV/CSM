import streamlit as st
import pandas as pd
import numpy as np
import os
from PIL import Image
from PIL import ImageFont, ImageDraw
import base64
from time import sleep
import datetime
import plotly.express as px
import plotly.graph_objects as go
# import winsound

import torch, cv2
from glob import glob
from tqdm import tqdm

from collections import Counter
from cell_box import cell_box_main

def style_button_row(clicked_button_ix, n_buttons):
    def get_button_indices(button_ix):
        return {
            'nth_child': button_ix,
            'nth_last_child': n_buttons - button_ix + 1
        }

    clicked_style = f"""
    div[data-testid*="stHorizontalBlock"] > div:nth-child(%(nth_child)s):nth-last-child(%(nth_last_child)s) button {{
        border-color: #0283EE;
        color: #0283EE;
        box-shadow: rgba(0, 176, 240, 0.5) 0px 0px 0px 0.2rem;
        outline: currentcolor none medium;
    }}
    """
    unclicked_style = f"""
    div[data-testid*="stHorizontalBlock"] > div:nth-child(%(nth_child)s):nth-last-child(%(nth_last_child)s) button {{
        pointer-events: none;
        cursor: not-allowed;
        opacity: 0.65;
        filter: alpha(opacity=65);
        -webkit-box-shadow: none;
        box-shadow: none;
    }}
    """

    style = ""
    for ix in range(n_buttons):
        ix += 1
        if ix == clicked_button_ix:
            style += clicked_style % get_button_indices(ix)
        else:
            style += unclicked_style % get_button_indices(ix)
    st.markdown(f"<style>{style}</style>", unsafe_allow_html=True)

st.sidebar.title("CSM for us")

sidebar_button = f"""
                <style>
                div.stButton > button{{
                width:13em;
                border-radius:20px 20px 20px 20px;
                font-size:20px;
                border-color: #00B0F0;
                color: #002F8E;
                box-shadow: rgba(0, 176, 240, 0.5) 0px 0px 0px 0.2rem;}}
                <style>
                """

st.markdown(sidebar_button, unsafe_allow_html=True)

def my_selectbox(model_list, key):
    return st.selectbox("모델 선택", tuple(model_list), key)

def cctv_button(num):
    return st.button("CCTV #"+num)

def list_chunk(lst, n):
    return [lst[i:i+n] for i in range(0, len(lst), n)]

def Demo(model_name, uploaded_file):
    main_bg = 'background.png'
    main_bg_ext = 'png'
    st.title("")
    st.markdown(
        f"""
                <style>
                    .reportview-container .main .block-container{{
                        max-width: {1600}px;
                        padding-top: {0.5}rem;
                        padding-right: {1}rem;
                        padding-left: {10}rem;
                        padding-bottom: {1}rem;
                    }}
                    .reportview-container .main {{
                        color: black;
                        background-color: white;
                        background-image: url(data:image/{main_bg_ext};base64,{base64.b64encode(open(os.getcwd() + os.sep + main_bg, "rb").read()).decode()});
                    }}

                </style>
                """,
        unsafe_allow_html=True,
    )

    model = torch.hub.load('ultralytics/yolov5', 'custom', path=f'Model/{model_name}')

    label_dict = {
        'danger': 'safe',
        'fire': 'danger',
        'safe': 'fire'
    }

    Demo_title = '<p style="font-family:sans-serif; color: #1D56C0; font-weight:bold; font-size: 42px;">CSM for us 시연해보기</p>'
    st.markdown(Demo_title, unsafe_allow_html=True)
    st.title("")
    st.write("")

    Demo_contents = '<p style="font-family:sans-serif; color: #1D56C0; font-size: 17px;">- 사용중인 모델 :  {}</p>'.format(model_name)
    st.markdown(Demo_contents, unsafe_allow_html=True)

    st.write("")
    if uploaded_file:
            image = Image.open(uploaded_file)
    d_col1, d_col2, d_col3 = st.columns([1, 1, 1])

    with d_col1:
        Demo_subtitle1 = '<p style="font-family:sans-serif; font-weight:bold; color: #0283EE; font-size: 27px;">입력 이미지</p>'
        st.markdown(Demo_subtitle1, unsafe_allow_html=True)
        st.image(image, use_column_width=True)

    with d_col2:
        Demo_subtitle2 = '<p style="font-family:sans-serif; font-weight:bold; color: #0283EE; font-size: 27px;">결과</p>'
        st.markdown(Demo_subtitle2, unsafe_allow_html=True)

        if uploaded_file:
            results = model(image, size=640)

            res = results.pandas().xyxy[0]
            res['name'] = res['name'].map(label_dict)

            for each in res.iloc:
                xmin, ymin, xmax, ymax, conf, name = int(each['xmin']), int(each['ymin']), int(each['xmax']), int(
                    each['ymax']), round(each['confidence'], 3), each['name']
                point1 = (xmin, ymin)
                point2 = (xmax, ymax)
                font = ImageFont.truetype("DejaVuSans.ttf", size=30)
                draw = ImageDraw.Draw(image)
                if name == 'safe':
                    draw.rectangle((point1, point2), outline=(124, 252, 0), width=5)
                    if point1[1] - 25 < 0:
                        draw.text((point1[0], 0), f'{name}: {conf}', fill=(124, 252, 0), font=font)
                    else:
                        draw.text((point1[0], point1[1] - 25), f'{name}: {conf}', fill=(124, 252, 0), font=font)
                elif name == 'danger':
                    draw.rectangle((point1, point2), outline=(255, 212, 0), width=5)
                    if point1[1] - 25 < 0:
                        draw.text((point1[0], 0), f'{name}: {conf}', fill=(255, 212, 0), font=font)
                    else:
                        draw.text((point1[0], point1[1] - 25), f'{name}: {conf}', fill=(255, 212, 0), font=font)
                elif name == 'fire':
                    draw.rectangle((point1, point2), outline=(155, 17, 30), width=5)
                    if point1[1] - 25 < 0:
                        draw.text((point1[0], 0), f'{name}: {conf}', fill=(155, 17, 30), font=font)
                    else:
                        draw.text((point1[0], point1[1] - 25), f'{name}: {conf}', fill=(155, 17, 30), font=font)

            st.image(image)

    with d_col3:
        Demo_subtitle2 = '<p style="font-family:sans-serif; color: #0283EE; font-weight:bold; font-size: 27px;">인식 결과</p>'
        st.markdown(Demo_subtitle2, unsafe_allow_html=True)
        if uploaded_file:
            safe    = Counter(res['name'])['safe']
            danger  = Counter(res['name'])['danger']
            fire    = Counter(res['name'])['fire']

            Demo_result1 = '<p style="font-family:sans-serif; color: black; font-weight:bold; font-size: 20px;">- 안전장비 착용 준수자 : {}명</p>'.format(safe)
            st.markdown(Demo_result1, unsafe_allow_html=True)
            st.markdown("")
            st.markdown("")
            Demo_result2 = '<p style="font-family:sans-serif; color: black; font-weight:bold; font-size: 20px;">- 안전장비 착용 미준수자 : {}명</p>'.format(danger)
            st.markdown(Demo_result2, unsafe_allow_html=True)
            st.markdown("")
            st.markdown("")
            Demo_result3 = '<p style="font-family:sans-serif; color: black; font-weight:bold; font-size: 20px;">- 화재 탐지 결과 : {}건 탐지</p>'.format(fire)
            st.markdown(Demo_result3, unsafe_allow_html=True)

add_side_button_home = st.sidebar.button("홈  🏡")
add_side_button_service = st.sidebar.button("서비스 안내  📖")

demo_expander = st.sidebar.expander("데모  🦱", expanded=True)

with demo_expander:
    model_list = os.listdir(os.getcwd() + os.sep + 'Model')
    select_box0 = my_selectbox(model_list, key=1)
    st.write(f"선택한 모델 : {select_box0}")
    uploaded_file = st.file_uploader("이미지를 입력해주세요...", type="jpg")
    demo = st.button("시연하기")

if demo:
    Demo(select_box0, uploaded_file)

def Monitoring(key, model):
    main_bg = "background.png"
    main_bg_ext = "png"

    st.markdown(
        f"""
                <style>
                    .reportview-container .main .block-container{{
                        max-width: {1050}px;
                        padding-top: {0.5}rem;
                        padding-right: {1}rem;
                        padding-left: {1}rem;
                        padding-bottom: {1}rem;
                    }}
                    .reportview-container .main {{
                        color: black;
                        background-color: white;
                        background-image: url(data:image/{main_bg_ext};base64,{base64.b64encode(open(os.getcwd() + os.sep + main_bg, "rb").read()).decode()});
                    }}

                </style>
                """,
        unsafe_allow_html=True,
    )

    st.markdown("""
    <style>

    </style>
        """, unsafe_allow_html=True)

    csv_path = os.getcwd() + os.sep + 'Monitoring_csv'
    csv_list = os.listdir(csv_path)

    if key == "CCTV #1":
        CCTV_title = '<p style="font-family:sans-serif; font-weight:bold; color: #1D56C0; font-size: 42px;">실시간 AI 모니터링</p>'
        st.markdown(CCTV_title, unsafe_allow_html=True)
        st.title("")
        st.write("")
        warn = "위험경보 작동 중"
        bell_col1, bell_col2 = st.columns([2.7, 1.3])

        with bell_col1:
            CCTV_contents = '<p style="font-family:sans-serif; color: #1D56C0; font-size: 17px;">- 사용중인 모델 :  {}</p>'.format(model)
            st.markdown(CCTV_contents, unsafe_allow_html=True)

        st.write("")

        m_col1, m_col2 = st.columns([2.8, 1.2])

        with m_col1:
            CCTV_subtitle = '<p style="font-family:sans-serif; font-weight:bold; color: #0283EE; font-size: 27px;">{} 영상</p>'.format(key)
            st.markdown(CCTV_subtitle, unsafe_allow_html=True)

            file_ = open(os.getcwd() + os.sep + "Monitoring_gif" + os.sep + "video1.gif", "rb") ## 수정
            contents = file_.read()
            data_url_1 = base64.b64encode(contents).decode("utf-8")
            file_.close()

            st.markdown(
                f'<img src="data:image/gif;base64,{data_url_1}" alt="scene1 gif"  width = 650 height = 450>',
                unsafe_allow_html=True,
            )

        with m_col2:

            c1_df = pd.read_csv(csv_path + os.sep +csv_list[0]) ## 수정
            # safe_df = c1_df['safe']
            danger_df = c1_df['danger']
            fire_df = c1_df['fire']

            danger_list = list(danger_df)
            fire_list = list(fire_df)

            danger_chunk_list = list_chunk(danger_list, 3)
            danger_chunk_list = danger_chunk_list[0:-1]

            fire_chunk_list = list_chunk(fire_list, 3)
            fire_chunk_list = fire_chunk_list[0:-1]

            CCTV_subtitle2 = '<p style="font-family:sans-serif; font-weight:bold; color: #0283EE; font-size: 27px;">탐지 결과</p>'
            st.markdown(CCTV_subtitle2, unsafe_allow_html=True)

            case_placeholder_0 = st.empty()
            time_placeholder_0 = st.empty()

            danger_count_placeholder_0 = st.empty()
            fire_count_placeholder_0 = st.empty()

            danger_placeholder_0 = st.empty()
            fire_placeholder_0 = st.empty()

            fire_attention_placeholder_0 = st.empty()

            case_placeholder_1 = st.empty()

            danger_mean_count = 0
            fire_mean_count = 0

            for i in range(len(danger_chunk_list)):

                case_placeholder_0.write("-----------------")

                now = datetime.datetime.now()
                nowDatetime = now.strftime('%Y-%m-%d %H:%M:%S')
                time_placeholder_0.markdown(f"탐지 시간 : {nowDatetime}")

                mean_danger_count = round(np.mean(danger_chunk_list[i]), 1)
                mean_fire_count = round(np.mean(fire_chunk_list[i]), 1)

                danger_count_placeholder_0.markdown(f"초당 평균 안전장비 착용 미준수자 : {mean_danger_count}명", unsafe_allow_html=True)
                fire_count_placeholder_0.markdown(f"초당 평균 화재 감지 횟수 : {mean_fire_count}회", unsafe_allow_html=True)

                if mean_danger_count >= 2:
                    danger_comment = '<p style="color:red; font-weight:bold;">안전장비 착용 상태 주의 요망</p>'
                    danger_placeholder_0.markdown(f"- 안전장비 탐지 결과: {danger_comment}", unsafe_allow_html=True)
                else:
                    no_danger_comment = '<p style="color:blue; font-weight:bold;">안전장비 착용 상태 양호</p>'
                    danger_placeholder_0.markdown(f"- 안전장비 탐지 결과:  {no_danger_comment}", unsafe_allow_html=True)

                if mean_fire_count == 0:
                    no_fire_comment = '<p style="color:blue; font-weight:bold;">화재발생 전조 없음</p>'
                    fire_placeholder_0.markdown(f"- 화재 탐지 결과: {no_fire_comment}", unsafe_allow_html=True)
                else:
                    fire_comment = '<p style="color:red; font-weight:bold;">화재발생 전조 감지</p>'
                    fire_placeholder_0.markdown(f"- 화재 탐지 결과: {fire_comment}", unsafe_allow_html=True)

                if fire_mean_count > 2:
                    fire_attention_comment = '<p style="color:red; font-weight:bold;">화재발생, 영상 및 발생위치 확인 바람!!</p>'
                    fire_attention_placeholder_0.markdown(f"- 특이사항 : {fire_attention_comment}", unsafe_allow_html=True)
                    # frequency = 2500  # Set Frequency To 2500 Hertz
                    # duration = 1000  # Set Duration To 1000 ms == 1 second
                    # winsound.Beep(frequency, duration)
                else:
                    fire_attention_placeholder_0.markdown(f"- 특이사항 : 없음", unsafe_allow_html=True)

                case_placeholder_1.write("-----------------")

                danger_mean_count = danger_mean_count + mean_danger_count
                fire_mean_count = fire_mean_count + mean_fire_count

                sleep(0.38)

                case_placeholder_0.empty()
                time_placeholder_0.empty()
                danger_count_placeholder_0.empty()
                fire_count_placeholder_0.empty()
                danger_placeholder_0.empty()
                fire_placeholder_0.empty()
                fire_attention_placeholder_0.empty()
                case_placeholder_1.empty()

            case_placeholder_0.write("-----------------")
            time_placeholder_0.markdown(f"탐지 시간 : {nowDatetime}")

            mean_danger_count = round(np.mean(danger_chunk_list[-1]), 1)
            mean_fire_count = round(np.mean(fire_chunk_list[-1]), 1)

            danger_count_placeholder_0.markdown(f"평균 안전장비 착용 미준수자 : {mean_danger_count}명", unsafe_allow_html=True)
            fire_count_placeholder_0.markdown(f"평균 화재 감지 횟수 : {mean_fire_count}회", unsafe_allow_html=True)

            if mean_danger_count >= 3:
                danger_comment = '<p style="color:red; font-weight:bold;">안전장비 착용 상태 주의 요망</p>'
                danger_placeholder_0.markdown(f"- 안전장비 탐지 결과: {danger_comment}", unsafe_allow_html=True)
            else:
                no_danger_comment = '<p style="color:blue;">안전장비 착용 상태 양호</p>'
                danger_placeholder_0.markdown(f"- 안전장비 탐지 결과:  {no_danger_comment}", unsafe_allow_html=True)

            if mean_fire_count == 0:
                no_fire_comment = '<p style="color:blue; font-weight:bold;">화재발생 전조 없음</p>'
                fire_placeholder_0.markdown(f"- 화재 탐지 결과: {no_fire_comment}", unsafe_allow_html=True)
            else:
                fire_comment = '<p style="color:red; font-weight:bold;">화재발생 전조 감지</p>'
                fire_placeholder_0.markdown(f"- 화재 탐지 결과: {fire_comment}", unsafe_allow_html=True)

            if fire_mean_count > 2:
                fire_attention_comment = '<p style="color:red; font-weight:bold;">화재발생, 영상 및 발생위치 확인 바람!!</p>'
                fire_attention_placeholder_0.markdown(f"- 특이사항 : {fire_attention_comment}", unsafe_allow_html=True)
                # frequency = 2500  # Set Frequency To 2500 Hertz
                # duration = 1000  # Set Duration To 1000 ms == 1 second
                # winsound.Beep(frequency, duration)
            else:
                fire_attention_placeholder_0.markdown(f"- 특이사항 : 없음", unsafe_allow_html=True)

            case_placeholder_1.write("-----------------")
            danger_mean_count = 0
            fire_mean_count = 0

    if key == "CCTV #2":

        CCTV_title = '<p style="font-family:sans-serif; font-weight:bold; color: #1D56C0; font-size: 42px;">실시간 AI 모니터링</p>'
        st.markdown(CCTV_title, unsafe_allow_html=True)
        st.title("")
        st.write("")
        warn = "위험경보 작동 중"
        bell_col1, bell_col2 = st.columns([2.7, 1.3])

        with bell_col1:
            CCTV_contents = '<p style="font-family:sans-serif; color: #1D56C0; font-size: 17px;">- 사용중인 모델 :  {}</p>'.format(model)
            st.markdown(CCTV_contents, unsafe_allow_html=True)

        st.write("")

        m_col1, m_col2 = st.columns([2.8, 1.2])

        with m_col1:
            CCTV_subtitle = '<p style="font-family:sans-serif; font-weight:bold; color: #0283EE; font-size: 27px;">{} 영상</p>'.format(key)
            st.markdown(CCTV_subtitle, unsafe_allow_html=True)

            file_ = open(os.getcwd() + os.sep + "Monitoring_gif" + os.sep + "video2.gif", "rb") ## 수정
            contents = file_.read()
            data_url_1 = base64.b64encode(contents).decode("utf-8")
            file_.close()

            st.markdown(
                f'<img src="data:image/gif;base64,{data_url_1}" alt="scene1 gif"  width = 650 height = 450>',
                unsafe_allow_html=True,
            )

        with m_col2:

            c1_df = pd.read_csv(csv_path + os.sep +csv_list[1]) ## 수정
            # safe_df = c1_df['safe']
            danger_df = c1_df['danger']
            fire_df = c1_df['fire']

            danger_list = list(danger_df)
            fire_list = list(fire_df)

            danger_chunk_list = list_chunk(danger_list, 3)
            danger_chunk_list = danger_chunk_list[0:-1]

            fire_chunk_list = list_chunk(fire_list, 3)
            fire_chunk_list = fire_chunk_list[0:-1]

            CCTV_subtitle2 = '<p style="font-family:sans-serif; font-weight:bold; color: #0283EE; font-size: 27px;">탐지 결과</p>'
            st.markdown(CCTV_subtitle2, unsafe_allow_html=True)

            case_placeholder_0 = st.empty()
            time_placeholder_0 = st.empty()

            danger_count_placeholder_0 = st.empty()
            fire_count_placeholder_0 = st.empty()

            danger_placeholder_0 = st.empty()
            fire_placeholder_0 = st.empty()

            fire_attention_placeholder_0 = st.empty()

            case_placeholder_1 = st.empty()

            danger_mean_count = 0
            fire_mean_count = 0

            for i in range(len(danger_chunk_list)):

                case_placeholder_0.write("-----------------")

                now = datetime.datetime.now()
                nowDatetime = now.strftime('%Y-%m-%d %H:%M:%S')
                time_placeholder_0.markdown(f"탐지 시간 : {nowDatetime}")

                mean_danger_count = round(np.mean(danger_chunk_list[i]), 1)
                mean_fire_count = round(np.mean(fire_chunk_list[i]), 1)

                danger_count_placeholder_0.markdown(f"초당 평균 안전장비 착용 미준수자 : {mean_danger_count}명", unsafe_allow_html=True)
                fire_count_placeholder_0.markdown(f"초당 평균 화재 감지 횟수 : {mean_fire_count}회", unsafe_allow_html=True)

                if mean_danger_count >= 3:
                    danger_comment = '<p style="color:red; font-weight:bold;">안전장비 착용 상태 주의 요망</p>'
                    danger_placeholder_0.markdown(f"- 안전장비 탐지 결과: {danger_comment}", unsafe_allow_html=True)
                else:
                    no_danger_comment = '<p style="color:blue; font-weight:bold;">안전장비 착용 상태 양호</p>'
                    danger_placeholder_0.markdown(f"- 안전장비 탐지 결과:  {no_danger_comment}", unsafe_allow_html=True)

                if mean_fire_count == 0:
                    no_fire_comment = '<p style="color:blue; font-weight:bold;">화재발생 전조 없음</p>'
                    fire_placeholder_0.markdown(f"- 화재 탐지 결과: {no_fire_comment}", unsafe_allow_html=True)
                else:
                    fire_comment = '<p style="color:red; font-weight:bold;">화재발생 전조 감지</p>'
                    fire_placeholder_0.markdown(f"- 화재 탐지 결과: {fire_comment}", unsafe_allow_html=True)

                if fire_mean_count > 2:
                    fire_attention_comment = '<p style="color:red; font-weight:bold;">화재발생, 영상 및 발생위치 확인 바람!!</p>'
                    fire_attention_placeholder_0.markdown(f"- 특이사항 : {fire_attention_comment}", unsafe_allow_html=True)
                    # frequency = 2500  # Set Frequency To 2500 Hertz
                    # duration = 1000  # Set Duration To 1000 ms == 1 second
                    # winsound.Beep(frequency, duration)
                else:
                    fire_attention_placeholder_0.markdown(f"- 특이사항 : 없음", unsafe_allow_html=True)

                case_placeholder_1.write("-----------------")

                danger_mean_count = danger_mean_count + mean_danger_count
                fire_mean_count = fire_mean_count + mean_fire_count

                sleep(0.37)

                case_placeholder_0.empty()
                time_placeholder_0.empty()
                danger_count_placeholder_0.empty()
                fire_count_placeholder_0.empty()
                danger_placeholder_0.empty()
                fire_placeholder_0.empty()
                fire_attention_placeholder_0.empty()
                case_placeholder_1.empty()

            case_placeholder_0.write("-----------------")
            time_placeholder_0.markdown(f"탐지 시간 : {nowDatetime}")

            mean_danger_count = round(np.mean(danger_chunk_list[-1]), 1)
            mean_fire_count = round(np.mean(fire_chunk_list[-1]), 1)

            danger_count_placeholder_0.markdown(f"평균 안전장비 착용 미준수자 : {mean_danger_count}명", unsafe_allow_html=True)
            fire_count_placeholder_0.markdown(f"평균 화재 감지 횟수 : {mean_fire_count}회", unsafe_allow_html=True)

            if mean_danger_count >= 3:
                danger_comment = '<p style="color:red; font-weight:bold;">안전장비 착용 상태 주의 요망</p>'
                danger_placeholder_0.markdown(f"- 안전장비 탐지 결과: {danger_comment}", unsafe_allow_html=True)
            else:
                no_danger_comment = '<p style="color:blue; font-weight:bold;">안전장비 착용 상태 양호</p>'
                danger_placeholder_0.markdown(f"- 안전장비 탐지 결과:  {no_danger_comment}", unsafe_allow_html=True)

            if mean_fire_count == 0:
                no_fire_comment = '<p style="color:blue; font-weight:bold;">화재발생 전조 없음</p>'
                fire_placeholder_0.markdown(f"- 화재 탐지 결과: {no_fire_comment}", unsafe_allow_html=True)
            else:
                fire_comment = '<p style="color:red; font-weight:bold;">화재발생 전조 감지</p>'
                fire_placeholder_0.markdown(f"- 화재 탐지 결과: {fire_comment}", unsafe_allow_html=True)

            if fire_mean_count > 2:
                fire_attention_comment = '<p style="color:red; font-weight:bold;">화재발생, 영상 및 발생위치 확인 바람!!</p>'
                fire_attention_placeholder_0.markdown(f"- 특이사항 : {fire_attention_comment}", unsafe_allow_html=True)
                # frequency = 2500  # Set Frequency To 2500 Hertz
                # duration = 1000  # Set Duration To 1000 ms == 1 second
                # winsound.Beep(frequency, duration)
            else:
                fire_attention_placeholder_0.markdown(f"- 특이사항 : 없음", unsafe_allow_html=True)

            case_placeholder_1.write("-----------------")
            danger_mean_count = 0
            fire_mean_count = 0

    if key == "CCTV #3":
        CCTV_title = '<p style="font-family:sans-serif; font-weight:bold; color: #1D56C0; font-size: 42px;">실시간 AI 모니터링</p>'
        st.markdown(CCTV_title, unsafe_allow_html=True)
        st.title("")
        st.write("")
        warn = "위험경보 작동 중"
        bell_col1, bell_col2 = st.columns([2.7, 1.3])

        with bell_col1:
            CCTV_contents = '<p style="font-family:sans-serif; color: #1D56C0; font-size: 17px;">- 사용중인 모델 :  {}</p>'.format(model)
            st.markdown(CCTV_contents, unsafe_allow_html=True)

        st.write("")

        m_col1, m_col2 = st.columns([2.8, 1.2])

        with m_col1:
            CCTV_subtitle = '<p style="font-family:sans-serif; font-weight:bold; color: #0283EE; font-size: 27px;">{} 영상</p>'.format(key)
            st.markdown(CCTV_subtitle, unsafe_allow_html=True)

            file_ = open(os.getcwd() + os.sep + "Monitoring_gif" + os.sep + "video3.gif", "rb") ## 수정
            contents = file_.read()
            data_url_1 = base64.b64encode(contents).decode("utf-8")
            file_.close()

            st.markdown(
                f'<img src="data:image/gif;base64,{data_url_1}" alt="scene1 gif"  width = 650 height = 450>',
                unsafe_allow_html=True,
            )

        with m_col2:

            c1_df = pd.read_csv(csv_path + os.sep +csv_list[2]) ## 수정
            # safe_df = c1_df['safe']
            danger_df = c1_df['danger']
            fire_df = c1_df['fire']

            danger_list = list(danger_df)
            fire_list = list(fire_df)

            danger_chunk_list = list_chunk(danger_list, 3)
            danger_chunk_list = danger_chunk_list[0:-1]

            fire_chunk_list = list_chunk(fire_list, 3)
            fire_chunk_list = fire_chunk_list[0:-1]

            CCTV_subtitle2 = '<p style="font-family:sans-serif; font-weight:bold; color: #0283EE; font-size: 27px;">탐지 결과</p>'
            st.markdown(CCTV_subtitle2, unsafe_allow_html=True)

            case_placeholder_0 = st.empty()
            time_placeholder_0 = st.empty()

            danger_count_placeholder_0 = st.empty()
            fire_count_placeholder_0 = st.empty()

            danger_placeholder_0 = st.empty()
            fire_placeholder_0 = st.empty()

            fire_attention_placeholder_0 = st.empty()

            case_placeholder_1 = st.empty()

            danger_mean_count = 0
            fire_mean_count = 0

            for i in range(len(danger_chunk_list)):

                case_placeholder_0.write("-----------------")

                now = datetime.datetime.now()
                nowDatetime = now.strftime('%Y-%m-%d %H:%M:%S')
                time_placeholder_0.markdown(f"탐지 시간 : {nowDatetime}")

                mean_danger_count = round(np.mean(danger_chunk_list[i]), 1)
                mean_fire_count = round(np.mean(fire_chunk_list[i]), 1)

                danger_count_placeholder_0.markdown(f"초당 평균 안전장비 착용 미준수자 : {mean_danger_count}명", unsafe_allow_html=True)
                fire_count_placeholder_0.markdown(f"초당 평균 화재 감지 횟수 : {mean_fire_count}회", unsafe_allow_html=True)

                if mean_danger_count >= 3:
                    danger_comment = '<p style="color:red; font-weight:bold;">안전장비 착용 상태 주의 요망</p>'
                    danger_placeholder_0.markdown(f"- 안전장비 탐지 결과: {danger_comment}", unsafe_allow_html=True)
                else:
                    no_danger_comment = '<p style="color:blue; font-weight:bold;">안전장비 착용 상태 양호</p>'
                    danger_placeholder_0.markdown(f"- 안전장비 탐지 결과:  {no_danger_comment}", unsafe_allow_html=True)

                if mean_fire_count == 0:
                    no_fire_comment = '<p style="color:blue; font-weight:bold;">화재발생 전조 없음</p>'
                    fire_placeholder_0.markdown(f"- 화재 탐지 결과: {no_fire_comment}", unsafe_allow_html=True)
                else:
                    fire_comment = '<p style="color:red; font-weight:bold;">화재발생 전조 감지</p>'
                    fire_placeholder_0.markdown(f"- 화재 탐지 결과: {fire_comment}", unsafe_allow_html=True)

                if fire_mean_count > 2:
                    fire_attention_comment = '<p style="color:red; font-weight:bold;">화재발생, 영상 및 발생위치 확인 바람!!</p>'
                    fire_attention_placeholder_0.markdown(f"- 특이사항 : {fire_attention_comment}", unsafe_allow_html=True)
                    # frequency = 2500  # Set Frequency To 2500 Hertz
                    # duration = 1000  # Set Duration To 1000 ms == 1 second
                    # winsound.Beep(frequency, duration)
                else:
                    fire_attention_placeholder_0.markdown(f"- 특이사항 : 없음", unsafe_allow_html=True)

                case_placeholder_1.write("-----------------")

                danger_mean_count = danger_mean_count + mean_danger_count
                fire_mean_count = fire_mean_count + mean_fire_count

                sleep(0.37)

                case_placeholder_0.empty()
                time_placeholder_0.empty()
                danger_count_placeholder_0.empty()
                fire_count_placeholder_0.empty()
                danger_placeholder_0.empty()
                fire_placeholder_0.empty()
                fire_attention_placeholder_0.empty()
                case_placeholder_1.empty()

            case_placeholder_0.write("-----------------")
            time_placeholder_0.markdown(f"탐지 시간 : {nowDatetime}")

            mean_danger_count = round(np.mean(danger_chunk_list[-1]), 1)
            mean_fire_count = round(np.mean(fire_chunk_list[-1]), 1)

            danger_count_placeholder_0.markdown(f"평균 안전장비 착용 미준수자 : {mean_danger_count}명", unsafe_allow_html=True)
            fire_count_placeholder_0.markdown(f"평균 화재 감지 횟수 : {mean_fire_count}회", unsafe_allow_html=True)

            if mean_danger_count >= 3:
                danger_comment = '<p style="color:red; font-weight:bold;">안전장비 착용 상태 주의 요망</p>'
                danger_placeholder_0.markdown(f"- 안전장비 탐지 결과: {danger_comment}", unsafe_allow_html=True)
            else:
                no_danger_comment = '<p style="color:blue; font-weight:bold;">안전장비 착용 상태 양호</p>'
                danger_placeholder_0.markdown(f"- 안전장비 탐지 결과:  {no_danger_comment}", unsafe_allow_html=True)

            if mean_fire_count == 0:
                no_fire_comment = '<p style="color:blue; font-weight:bold;">화재발생 전조 없음</p>'
                fire_placeholder_0.markdown(f"- 화재 탐지 결과: {no_fire_comment}", unsafe_allow_html=True)
            else:
                fire_comment = '<p style="color:red; font-weight:bold;">화재발생 전조 감지</p>'
                fire_placeholder_0.markdown(f"- 화재 탐지 결과: {fire_comment}", unsafe_allow_html=True)

            if fire_mean_count > 2:
                fire_attention_comment = '<p style="color:red; font-weight:bold;">화재발생, 영상 및 발생위치 확인 바람!!</p>'
                fire_attention_placeholder_0.markdown(f"- 특이사항 : {fire_attention_comment}", unsafe_allow_html=True)
                # frequency = 2500  # Set Frequency To 2500 Hertz
                # duration = 1000  # Set Duration To 1000 ms == 1 second
                # winsound.Beep(frequency, duration)
            else:
                fire_attention_placeholder_0.markdown(f"- 특이사항 : 없음", unsafe_allow_html=True)

            case_placeholder_1.write("-----------------")
            danger_mean_count = 0
            fire_mean_count = 0

    if key == "CCTV #4":
        CCTV_title = '<p style="font-family:sans-serif; font-weight:bold; color: #1D56C0; font-size: 42px;">실시간 AI 모니터링</p>'
        st.markdown(CCTV_title, unsafe_allow_html=True)
        st.title("")
        st.write("")
        warn = "위험경보 작동 중"
        bell_col1, bell_col2 = st.columns([2.7, 1.3])

        with bell_col1:
            CCTV_contents = '<p style="font-family:sans-serif; color: #1D56C0; font-size: 17px;">- 사용중인 모델 :  {}</p>'.format(model)
            st.markdown(CCTV_contents, unsafe_allow_html=True)

        st.write("")

        m_col1, m_col2 = st.columns([2.8, 1.2])

        with m_col1:
            CCTV_subtitle = '<p style="font-family:sans-serif; font-weight:bold; color: #0283EE; font-size: 27px;">{} 영상</p>'.format(key)
            st.markdown(CCTV_subtitle, unsafe_allow_html=True)

            file_ = open(os.getcwd() + os.sep + "Monitoring_gif" + os.sep + "video4.gif", "rb") ## 수정

            contents = file_.read()
            data_url_1 = base64.b64encode(contents).decode("utf-8")
            file_.close()

            st.markdown(
                f'<img src="data:image/gif;base64,{data_url_1}" alt="scene1 gif"  width = 650 height = 450>',
                unsafe_allow_html=True,
            )

        with m_col2:

            c1_df = pd.read_csv(csv_path + os.sep +csv_list[3]) ## 수정
            # safe_df = c1_df['safe']
            danger_df = c1_df['danger']
            fire_df = c1_df['fire']

            danger_list = list(danger_df)
            fire_list = list(fire_df)

            danger_chunk_list = list_chunk(danger_list, 3)
            danger_chunk_list = danger_chunk_list[0:-1]

            fire_chunk_list = list_chunk(fire_list, 3)
            fire_chunk_list = fire_chunk_list[0:-1]

            CCTV_subtitle2 = '<p style="font-family:sans-serif; font-weight:bold; color: #0283EE; font-size: 27px;">탐지 결과</p>'
            st.markdown(CCTV_subtitle2, unsafe_allow_html=True)

            case_placeholder_0 = st.empty()
            time_placeholder_0 = st.empty()

            danger_count_placeholder_0 = st.empty()
            fire_count_placeholder_0 = st.empty()

            danger_placeholder_0 = st.empty()
            fire_placeholder_0 = st.empty()

            fire_attention_placeholder_0 = st.empty()

            case_placeholder_1 = st.empty()

            danger_mean_count = 0
            fire_mean_count = 0

            for i in range(len(danger_chunk_list)):

                case_placeholder_0.write("-----------------")

                now = datetime.datetime.now()
                nowDatetime = now.strftime('%Y-%m-%d %H:%M:%S')
                time_placeholder_0.markdown(f"탐지 시간 : {nowDatetime}")

                mean_danger_count = round(np.mean(danger_chunk_list[i]), 1)
                mean_fire_count = round(np.mean(fire_chunk_list[i]), 1)

                danger_count_placeholder_0.markdown(f"초당 평균 안전장비 착용 미준수자 : {mean_danger_count}명", unsafe_allow_html=True)
                fire_count_placeholder_0.markdown(f"초당 평균 화재 감지 횟수 : {mean_fire_count}회", unsafe_allow_html=True)

                if mean_danger_count >= 3:
                    danger_comment = '<p style="color:red; font-weight:bold;">안전장비 착용 상태 주의 요망</p>'
                    danger_placeholder_0.markdown(f"- 안전장비 탐지 결과: {danger_comment}", unsafe_allow_html=True)
                else:
                    no_danger_comment = '<p style="color:blue; font-weight:bold;">안전장비 착용 상태 양호</p>'
                    danger_placeholder_0.markdown(f"- 안전장비 탐지 결과:  {no_danger_comment}", unsafe_allow_html=True)

                if mean_fire_count == 0:
                    no_fire_comment = '<p style="color:blue; font-weight:bold;">화재발생 전조 없음</p>'
                    fire_placeholder_0.markdown(f"- 화재 탐지 결과: {no_fire_comment}", unsafe_allow_html=True)
                else:
                    fire_comment = '<p style="color:red; font-weight:bold;">화재발생 전조 감지</p>'
                    fire_placeholder_0.markdown(f"- 화재 탐지 결과: {fire_comment}", unsafe_allow_html=True)

                if fire_mean_count > 2:
                    fire_attention_comment = '<p style="color:red; font-weight:bold;">화재발생, 영상 및 발생위치 확인 바람!!</p>'
                    fire_attention_placeholder_0.markdown(f"- 특이사항 : {fire_attention_comment}", unsafe_allow_html=True)
                    # frequency = 2500  # Set Frequency To 2500 Hertz
                    # duration = 1000  # Set Duration To 1000 ms == 1 second
                    # winsound.Beep(frequency, duration)
                else:
                    fire_attention_placeholder_0.markdown(f"- 특이사항 : 없음", unsafe_allow_html=True)

                case_placeholder_1.write("-----------------")

                danger_mean_count = danger_mean_count + mean_danger_count
                fire_mean_count = fire_mean_count + mean_fire_count

                sleep(0.38)

                case_placeholder_0.empty()
                time_placeholder_0.empty()
                danger_count_placeholder_0.empty()
                fire_count_placeholder_0.empty()
                danger_placeholder_0.empty()
                fire_placeholder_0.empty()
                fire_attention_placeholder_0.empty()
                case_placeholder_1.empty()

            case_placeholder_0.write("-----------------")
            time_placeholder_0.markdown(f"탐지 시간 : {nowDatetime}")

            mean_danger_count = round(np.mean(danger_chunk_list[-1]), 1)
            mean_fire_count = round(np.mean(fire_chunk_list[-1]), 1)

            danger_count_placeholder_0.markdown(f"평균 안전장비 착용 미준수자 : {mean_danger_count}명", unsafe_allow_html=True)
            fire_count_placeholder_0.markdown(f"평균 화재 감지 횟수 : {mean_fire_count}회", unsafe_allow_html=True)

            if mean_danger_count >= 3:
                danger_comment = '<p style="color:red; font-weight:bold;">안전장비 착용 상태 주의 요망</p>'
                danger_placeholder_0.markdown(f"- 안전장비 탐지 결과: {danger_comment}", unsafe_allow_html=True)
            else:
                no_danger_comment = '<p style="color:blue; font-weight:bold;">안전장비 착용 상태 양호</p>'
                danger_placeholder_0.markdown(f"- 안전장비 탐지 결과:  {no_danger_comment}", unsafe_allow_html=True)

            if mean_fire_count == 0:
                no_fire_comment = '<p style="color:blue; font-weight:bold;">화재발생 전조 없음</p>'
                fire_placeholder_0.markdown(f"- 화재 탐지 결과: {no_fire_comment}", unsafe_allow_html=True)
            else:
                fire_comment = '<p style="color:red; font-weight:bold;">화재발생 전조 감지</p>'
                fire_placeholder_0.markdown(f"- 화재 탐지 결과: {fire_comment}", unsafe_allow_html=True)

            if fire_mean_count > 2:
                fire_attention_comment = '<p style="color:red; font-weight:bold;">화재발생, 영상 및 발생위치 확인 바람!!</p>'
                fire_attention_placeholder_0.markdown(f"- 특이사항 : {fire_attention_comment}", unsafe_allow_html=True)
                # frequency = 2500  # Set Frequency To 2500 Hertz
                # duration = 1000  # Set Duration To 1000 ms == 1 second
                # winsound.Beep(frequency, duration)
            else:
                fire_attention_placeholder_0.markdown(f"- 특이사항 : 없음", unsafe_allow_html=True)

            case_placeholder_1.write("-----------------")
            danger_mean_count = 0
            fire_mean_count = 0

moni_expander = st.sidebar.expander("AI 모니터링  🖥️",expanded=True)

with moni_expander:
    model_list = os.listdir(os.getcwd() + os.sep + 'Model')
    select_box1 = my_selectbox(model_list, key=0)
    st.write(f"선택한 모델 : {select_box1}")

    c1 = cctv_button("1")

    c2 = cctv_button("2")

    c3 = cctv_button("3")

    c4 = cctv_button("4")

if c1:
    Monitoring("CCTV #1", select_box1)

if c2:
    Monitoring("CCTV #2", select_box1)

if c3:
    Monitoring("CCTV #3", select_box1)

if c4:
    Monitoring("CCTV #4", select_box1)


def Dash_borad(case, filter):

    main_bg = 'background.png'
    main_bg_ext = 'png'

    # st.markdown(
    #     f"""
    #             <style>
    #                 .reportview-container .main .block-container{{
    #                     max-width: {1200}px;
    #                     padding-top: {0.5}rem;
    #                     padding-right: {1}rem;
    #                     padding-left: {1}rem;
    #                     padding-bottom: {1}rem;
    #                 }}
    #                 .reportview-container .main {{
    #                     color: black;
    #                     background-color: white;
    #                     background-image: url(data:image/{main_bg_ext};base64,{base64.b64encode(open(os.getcwd() + os.sep + main_bg, "rb").read()).decode()});
    #
    #                 }}
    #             </style>
    #             """,
    #     unsafe_allow_html=True,
    # )

    st.markdown(
        f"""
                <style>
                    .reportview-container .main .block-container{{
                        max-width: {1200}px;
                        padding-top: {0.5}rem;
                        padding-right: {1}rem;
                        padding-left: {1}rem;
                        padding-bottom: {1}rem;
                    }}
                    .reportview-container .main {{
                        color: black;
                        background-color: white;
                    }}
                </style>
                """,
        unsafe_allow_html=True,
    )
    Dash_title = '<p style="font-family:sans-serif; font-weight:bold; color: #1D56C0; font-size: 42px;">Dash Board</p>'
    st.markdown(Dash_title, unsafe_allow_html=True)
    dash_data = pd.read_csv(os.getcwd() + os.sep + 'dash_data.csv')

    filter_week_data = []
    week_filter = ''

    if case == "일":
        plot_case = 'day'
        plot_name = '일별'
        plot_name2 = '일'

    if case == "요일":
        ori_week = ['월', '화', '수', '목', '금', '토', '일']

        input_week = filter
        filter = sorted(input_week, key=ori_week.index)
        week_filter = filter

        for i in range(len(ori_week)):
            for j in range(len(filter)):
                if filter[j] in ori_week[i]:
                    index = ori_week.index(filter[j])
                    filter_week_data.append(index + 1)

        plot_case = 'weekday_2'
        plot_name = '요일별'

    if case == "시간":
        plot_case = 'time'
        plot_name = '시간별'
        plot_name2 = '시'

    if len(filter) == 1:
        filter = filter

        if plot_case == 'weekday_2':
            filter_data = dash_data.loc[dash_data[plot_case] == filter_week_data[0]]
        else:
            filter_data = dash_data.loc[dash_data[plot_case] == filter[0]]

    elif len(filter) >= 2:

        if plot_case == 'time':
            dash_data['time'] = pd.to_datetime(dash_data['time'], errors='coerce')
            dash_data['time'] = dash_data['time'].dt.hour

        if plot_case == 'weekday_2':

            if len(filter_week_data) == 2:
                filter_1, filter_2 = filter_week_data
                filter_data = dash_data.loc[(dash_data[plot_case] == filter_1) |\
                                            (dash_data[plot_case] == filter_2)]
            if len(filter_week_data) == 3:
                filter_1, filter_2, filter_3 = filter_week_data
                filter_data = dash_data.loc[(dash_data[plot_case] == filter_1) |\
                                            (dash_data[plot_case] == filter_2) |\
                                            (dash_data[plot_case] == filter_3)]
            if len(filter_week_data) == 4:
                filter_1, filter_2, filter_3, filter_4 = filter_week_data
                filter_data = dash_data.loc[(dash_data[plot_case] == filter_1) |\
                                            (dash_data[plot_case] == filter_2) |\
                                            (dash_data[plot_case] == filter_3) |\
                                            (dash_data[plot_case] == filter_4)]
            if len(filter_week_data) == 5:
                filter_1, filter_2, filter_3, filter_4, filter_5 = filter_week_data
                filter_data = dash_data.loc[(dash_data[plot_case] == filter_1) |\
                                            (dash_data[plot_case] == filter_2) |\
                                            (dash_data[plot_case] == filter_3) |\
                                            (dash_data[plot_case] == filter_4) |\
                                            (dash_data[plot_case] == filter_5)]
            if len(filter_week_data) == 6:
                filter_1, filter_2, filter_3, filter_4, filter_5, filter_6 = filter_week_data
                filter_data = dash_data.loc[(dash_data[plot_case] == filter_1) |\
                                            (dash_data[plot_case] == filter_2) |\
                                            (dash_data[plot_case] == filter_3) |\
                                            (dash_data[plot_case] == filter_4) |\
                                            (dash_data[plot_case] == filter_5) |\
                                            (dash_data[plot_case] == filter_6)]
            if len(filter_week_data) == 7:
                filter_1, filter_2, filter_3, filter_4, filter_5, filter_6, filter_7 = filter_week_data
                filter_data = dash_data.loc[(dash_data[plot_case] == filter_1) |\
                                            (dash_data[plot_case] == filter_2) |\
                                            (dash_data[plot_case] == filter_3) |\
                                            (dash_data[plot_case] == filter_4) |\
                                            (dash_data[plot_case] == filter_5) |\
                                            (dash_data[plot_case] == filter_6) |\
                                            (dash_data[plot_case] == filter_7)]

        else:
            filter_1, filter_2 = filter
            filter_data = dash_data.loc[(dash_data[plot_case] >= filter_1) & (dash_data[plot_case] <= filter_2)]

    if plot_case == 'weekday_2':
        # week_label = []
        # for i in range(len(filter_week_data)):
        #     week_label.append(ori_week[filter_week_data[i] - 1])

        fire = filter_data.loc[:, ['fire', 'weekday']]
        safe = filter_data.loc[:, ['safe', 'weekday']]
        danger = filter_data.loc[:, ['danger', 'weekday']]
        pi_data = filter_data.loc[:, ['fire', 'safe', 'danger', 'weekday']]

        fire_groupby = fire.groupby(['weekday'], as_index = False).agg('sum')
        safe_groupby = safe.groupby(['weekday'], as_index = False).agg('sum')
        danger_groupby = danger.groupby(['weekday'], as_index = False).agg('sum')

        week_label = []
        fire_groupby['weekday_num'] = None

        for i in range(len(filter)):
            for j in range(len(ori_week)):
                if fire_groupby['weekday'][i] in ori_week[j]:
                    index = ori_week.index(fire_groupby['weekday'][i])
                    week_label.append(index + 1)

        for i in range(len(week_label)):
            fire_groupby['weekday_num'][i] = week_label[i]

        fire_groupby = fire_groupby.sort_values(by=['weekday_num'], axis=0)
        fire_groupby = fire_groupby.reset_index(drop=True)

        week_label = []
        safe_groupby['weekday_num'] = None

        for i in range(len(filter)):
            for j in range(len(ori_week)):
                if safe_groupby['weekday'][i] in ori_week[j]:
                    index = ori_week.index(safe_groupby['weekday'][i])
                    week_label.append(index + 1)

        for i in range(len(week_label)):
            safe_groupby['weekday_num'][i] = week_label[i]

        safe_groupby = safe_groupby.sort_values(by=['weekday_num'], axis=0)
        safe_groupby = safe_groupby.reset_index(drop=True)

        week_label = []
        danger_groupby['weekday_num'] = None

        for i in range(len(filter)):
            for j in range(len(ori_week)):
                if danger_groupby['weekday'][i] in ori_week[j]:
                    index = ori_week.index(danger_groupby['weekday'][i])
                    week_label.append(index + 1)

        for i in range(len(week_label)):
            danger_groupby['weekday_num'][i] = week_label[i]

        danger_groupby = danger_groupby.sort_values(by=['weekday_num'], axis=0)
        danger_groupby = danger_groupby.reset_index(drop=True)

        # fire_x = fire_groupby['weekday']
        # fire_y = fire_groupby['fire']
        #
        # safe_x = safe_groupby['weekday']
        # safe_y = safe_groupby['safe']
        #
        # danger_x = danger_groupby['weekday']
        # danger_y = danger_groupby['danger']

        plot_case = 'weekday'

    else:
        fire = filter_data.loc[:, ['fire', plot_case]]
        safe = filter_data.loc[:, ['safe', plot_case]]
        danger = filter_data.loc[:, ['danger', plot_case]]
        pi_data = filter_data.loc[:, ['fire', 'safe', 'danger', plot_case]]

        fire_groupby = fire.groupby([plot_case], as_index = False).agg('sum')
        safe_groupby = safe.groupby([plot_case], as_index = False).agg('sum')
        danger_groupby = danger.groupby([plot_case], as_index = False).agg('sum')


        # fire_x = fire_groupby[plot_case]
        # fire_y = fire_groupby['fire']
        #
        # safe_x = safe_groupby[plot_case]
        # safe_y = safe_groupby['safe']
        #
        # danger_x = danger_groupby[plot_case]
        # danger_y = danger_groupby['danger']

    fire_sum = sum(list(pi_data['fire']))
    safe_sum = sum(list(pi_data['safe']))
    danger_sum = sum((pi_data['danger']))

    cell_box_main(fire_sum, safe_sum, danger_sum)
    st.write("")
    st.write("")

    num_col1, num_col2, num_col3 = st.columns(3)
    with num_col1:
        if len(filter) == 1:

            if plot_case =='weekday':
                dasg_subtitle1 = '<p style="font-family:sans-serif; font-weight:bold; color: #0283EE; font-size: 15px;">{} 화재 탐지 비중{}</p>'.format(plot_name,  filter)
            else:
                dasg_subtitle1 = '<p style="font-family:sans-serif; font-weight:bold; color: #0283EE; font-size: 15px;">{} 화재 탐지 비중[{}{}]</p>'.format(plot_name, filter, plot_name2)

            st.markdown(dasg_subtitle1, unsafe_allow_html=True)

        elif plot_case == 'weekday':
            dasg_subtitle1 = '<p style="font-family:sans-serif; font-weight:bold; color: #0283EE; font-size: 15px;">{} 화재 탐지 비중</p>\
                            <p style="font-family:sans-serif; font-weight:bold; color: #0283EE; font-size: 15px;">{}</p>'.format(plot_name, week_filter)
            st.markdown(dasg_subtitle1, unsafe_allow_html=True)
        else:
            if filter[0] == filter[1]:
                dasg_subtitle1 = '<p style="font-family:sans-serif; font-weight:bold; color: #0283EE; font-size: 15px;">{} 화재 탐지 비중[{}{}]</p>'.format(plot_name, filter_1, plot_name2)
            else:
                dasg_subtitle1 = '<p style="font-family:sans-serif; font-weight:bold; color: #0283EE; font-size: 15px;">{} 화재 탐지 비중[{}~{}{}]</p>'.format(plot_name, filter_1,
                                                                                                                                                     filter_2, plot_name2)
            st.markdown(dasg_subtitle1, unsafe_allow_html=True)

        pie_1 = px.pie(pi_data, values='fire', names=plot_case, color=plot_case,
                     color_discrete_sequence=px.colors.cmocean.dense)

        st.plotly_chart(pie_1, use_container_width=True)

    with num_col2:
        if len(filter) == 1:
            if plot_case =='weekday':
                dasg_subtitle2 = '<p style="font-family:sans-serif; font-weight:bold; color: #0283EE; font-size: 15px;">{} 안전장비 착용 준수자 탐지 비중{}</p>'.format(plot_name,  filter)
            else:
                dasg_subtitle2 = '<p style="font-family:sans-serif; font-weight:bold; color: #0283EE; font-size: 15px;">{} 안전장비 착용 준수자 탐지 비중[{}{}]</p>'.format(plot_name,
                                                                                                                                                               filter, plot_name2)

            st.markdown(dasg_subtitle2, unsafe_allow_html=True)
        elif plot_case == 'weekday':
            # dasg_subtitle2 = '<p style="font-family:sans-serif; font-weight:bold; color: #0283EE; font-size: 25px;">{}~{} </p>'.format(week_filter_1, week_filter_2)
            dasg_subtitle2 = '<p style="font-family:sans-serif; font-weight:bold; color: #0283EE; font-size: 15px;">{} 안전장비 착용 준수자 탐지 비중</p>\
                            <p style="font-family:sans-serif; font-weight:bold; color: #0283EE; font-size: 15px;">{}</p>'.format(plot_name, week_filter)
            st.markdown(dasg_subtitle2, unsafe_allow_html=True)
        else:
            if filter[0] == filter[1]:
                dasg_subtitle2 = '<p style="font-family:sans-serif; font-weight:bold; color: #0283EE; font-size: 15px;">{} 안전장비 착용 준수자 탐지 비중[{}{}]</p>'.format(plot_name, filter_1, plot_name2)
            else:
                dasg_subtitle2 = '<p style="font-family:sans-serif; font-weight:bold; color: #0283EE; font-size: 15px;">{} 안전장비 착용 준수자 탐지 비중[{}~{}{}]</p>'.format(plot_name, filter_1,
                                                                                                                                                              filter_2, plot_name2)
            st.markdown(dasg_subtitle2, unsafe_allow_html=True)

        pie_2 = px.pie(pi_data, values='safe', names=plot_case, color=plot_case,
                     color_discrete_sequence=px.colors.cmocean.dense)

        st.plotly_chart(pie_2, use_container_width=True)

    with num_col3:
        if len(filter) == 1:
            if plot_case =='weekday':
                dasg_subtitle3 = '<p style="font-family:sans-serif; font-weight:bold; color: #0283EE; font-size: 15px;">{} 안전장비 착용 미준수자 탐지 비중{}</p>'.format(plot_name,  filter)
            else:
                dasg_subtitle3 = '<p style="font-family:sans-serif; font-weight:bold; color: #0283EE; font-size: 15px;">{} 안전장비 착용 미준수자 탐지 비중[{}{}]</p>'.format(plot_name,
                                                                                                                                                               filter, plot_name2)
            st.markdown(dasg_subtitle3, unsafe_allow_html=True)
        elif plot_case == 'weekday':
            dasg_subtitle3 = '<p style="font-family:sans-serif; font-weight:bold; color: #0283EE; font-size: 15px;">{} 안전장비 착용 미준수자 탐지 비중</p>\
                            <p style="font-family:sans-serif; font-weight:bold; color: #0283EE; font-size: 15px;">{}</p>'.format(plot_name, week_filter)
            st.markdown(dasg_subtitle3, unsafe_allow_html=True)
        else:
            if filter[0] == filter[1]:
                dasg_subtitle3 = '<p style="font-family:sans-serif; font-weight:bold; color: #0283EE; font-size: 15px;">{} 안전장비 착용 미준수자 탐지 비중[{}{}]</p>'.format(plot_name, filter_1, plot_name2)
            else:
                dasg_subtitle3 = '<p style="font-family:sans-serif; font-weight:bold; color: #0283EE; font-size: 15px;">{} 안전장비 착용 미준수자 탐지 비중[{}~{}{}] </p>'.format(plot_name, filter_1,
                                                                                                                                                                filter_2, plot_name2)
            st.markdown(dasg_subtitle3, unsafe_allow_html=True)

        pie_3 = px.pie(pi_data, values='danger', names=plot_case, color=plot_case,
                     color_discrete_sequence=px.colors.cmocean.dense)
        st.plotly_chart(pie_3, use_container_width=True)

    chart_col1, chart_col2, chart_col3 = st.columns(3)
    with chart_col1:
        if len(filter) == 1:
            if plot_case =='weekday':
                dasg_subtitle4 = '<p style="font-family:sans-serif; font-weight:bold; color: #0283EE; font-size: 15px;">{} 화재 탐지 횟수{}</p>'.format(plot_name,  filter)
            else:
                dasg_subtitle4 = '<p style="font-family:sans-serif; font-weight:bold; color: #0283EE; font-size: 15px;">{} 화재 탐지 횟수[{}{}]</p>'.format(plot_name,
                                                                                                                                                        filter, plot_name2)
            st.markdown(dasg_subtitle4, unsafe_allow_html=True)
        elif plot_case == 'weekday':
            dasg_subtitle4 = '<p style="font-family:sans-serif; font-weight:bold; color: #0283EE; font-size: 15px;">{} 화재 탐지 횟수</p>\
                            <p style="font-family:sans-serif; font-weight:bold; color: #0283EE; font-size: 15px;">{}</p>'.format(plot_name, week_filter)
            st.markdown(dasg_subtitle4, unsafe_allow_html=True)
        else:
            if filter[0] == filter[1]:
                dasg_subtitle4 = '<p style="font-family:sans-serif; font-weight:bold; color: #0283EE; font-size: 15px;">{} 화재 탐지 횟수[{}{}]</p>'.format(plot_name, filter_1, plot_name2)
            else:
                dasg_subtitle4 = '<p style="font-family:sans-serif; font-weight:bold; color: #0283EE; font-size: 15px;">{} 화재 탐지 횟수[{}~{}{}]</p>'.format(plot_name, filter_1,
                                                                                                                                                     filter_2, plot_name2)
            st.markdown(dasg_subtitle4, unsafe_allow_html=True)

        bar_1 = px.bar(
            fire_groupby,
            x= plot_case,
            y= 'fire',
            labels={'fire' : 'Detected Counts : %s' %('fire')},
            color= plot_case,
            color_discrete_sequence=px.colors.cmocean.dense)

        st.plotly_chart(bar_1, use_container_width=True)

    with chart_col2:
        if len(filter) == 1:
            if plot_case =='weekday':
                dasg_subtitle5 = '<p style="font-family:sans-serif; font-weight:bold; color: #0283EE; font-size: 15px;">{} 안전장비 착용 준수자 탐지 횟수{}</p>'.format(plot_name,  filter)
            else:
                dasg_subtitle5 = '<p style="font-family:sans-serif; font-weight:bold; color: #0283EE; font-size: 15px;">{} 안전장비 착용 준수자 탐지 횟수[{}{}]</p>'.format(plot_name,
                                                                                                                                                                filter, plot_name2)
            st.markdown(dasg_subtitle5, unsafe_allow_html=True)
        elif plot_case == 'weekday':
            dasg_subtitle5 = '<p style="font-family:sans-serif; font-weight:bold; color: #0283EE; font-size: 15px;">{} 안전장비 착용 준수자 탐지 횟수</p>\
                            <p style="font-family:sans-serif; font-weight:bold; color: #0283EE; font-size: 15px;">{}</p>'.format(plot_name, week_filter)
            st.markdown(dasg_subtitle5, unsafe_allow_html=True)
        else:
            if filter[0] == filter[1]:
                dasg_subtitle5 = '<p style="font-family:sans-serif; font-weight:bold; color: #0283EE; font-size: 15px;">{} 안전장비 착용 준수자 탐지 횟수[{}{}]</p>'.format(plot_name, filter_1, plot_name2)
            else:
                dasg_subtitle5 = '<p style="font-family:sans-serif; font-weight:bold; color: #0283EE; font-size: 15px;">{} 안전장비 착용 준수자 탐지 횟수[{}~{}{}]</p>'.format(plot_name, filter_1,
                                                                                                                                                              filter_2, plot_name2)
            st.markdown(dasg_subtitle5, unsafe_allow_html=True)

        bar_2 = px.bar(
            safe_groupby,
            x= plot_case,
            y= 'safe',
            labels={'safe' : 'Detected Counts : %s' %('safe')},
            color= plot_case,
            color_discrete_sequence=px.colors.cmocean.dense)

        st.plotly_chart(bar_2, use_container_width=True)

    with chart_col3:
        if len(filter) == 1:
            if plot_case =='weekday':
                dasg_subtitle6 = '<p style="font-family:sans-serif; font-weight:bold; color: #0283EE; font-size: 15px;">{} 안전장비 착용 미준수자 탐지 횟수{}</p>'.format(plot_name,  filter)
            else:
                dasg_subtitle6 = '<p style="font-family:sans-serif; font-weight:bold; color: #0283EE; font-size: 15px;">{} 안전장비 착용 미준수자 탐지 횟수[{}{}]</p>'.format(plot_name,
                                                                                                                                                                    filter, plot_name2)
            st.markdown(dasg_subtitle6, unsafe_allow_html=True)
        elif plot_case == 'weekday':
            dasg_subtitle6 = '<p style="font-family:sans-serif; font-weight:bold; color: #0283EE; font-size: 15px;">{} 안전장비 착용 미준수자 탐지 횟수</p>\
                            <p style="font-family:sans-serif; font-weight:bold; color: #0283EE; font-size: 15px;">{}</p>'.format(plot_name, week_filter)
            st.markdown(dasg_subtitle6, unsafe_allow_html=True)
        else:
            if filter[0] == filter[1]:
                dasg_subtitle6 = '<p style="font-family:sans-serif; font-weight:bold; color: #0283EE; font-size: 15px;">{} 안전장비 착용 미준수자 탐지 횟수[{}{}]</p>'.format(plot_name, filter_1, plot_name2)
            else:
                dasg_subtitle6 = '<p style="font-family:sans-serif; font-weight:bold; color: #0283EE; font-size: 15px;">{} 안전장비 착용 미준수자 탐지 횟수[{}~{}{}]</p>'.format(plot_name, filter_1,
                                                                                                                                                               filter_2, plot_name2)
            st.markdown(dasg_subtitle6, unsafe_allow_html=True)

        bar_3 = px.bar(
            danger_groupby,
            x= plot_case,
            y= 'danger',
            labels={'danger' : 'Detected Counts : %s' %('danger')},
            color= plot_case,
            color_discrete_sequence=px.colors.cmocean.dense)

        st.plotly_chart(bar_3, use_container_width=True)

    filter = ""

dash_expander = st.sidebar.expander("대시보드  📊", expanded=True)
with dash_expander:
    case = st.selectbox("형식 선택", ('일', '요일', '시간'), key=0)

    if case == '일':
        filter = st.slider('일을 선택해주세요.', 1, 30, (10, 18), key=0)

    if case == '요일':
        filter = st.multiselect("요일을 선택해주세요.", ('월', '화', '수', '목', '금', '토', '일'), key=0)

    if case == '시간':
        filter = st.slider('시간을 선택해주세요.', 0, 24, (9, 18), key=0)

    st.write("")
    dash_start = st.button("Dash Board 작성하기")

if dash_start:
    Dash_borad(case, filter)

def Database(case, filter):
    main_bg = 'background.png'
    main_bg_ext = 'png'
    st.markdown(
        f"""
                <style>
                    .reportview-container .main .block-container{{
                        max-width: {1200}px;
                        padding-top: {0.5}rem;
                        padding-right: {1}rem;
                        padding-left: {1}rem;
                        padding-bottom: {1}rem;
                    }}
                    .reportview-container .main {{
                        color: black;
                        background-color: white;
                        background-image: url(data:image/{main_bg_ext};base64,{base64.b64encode(open(os.getcwd() + os.sep + main_bg, "rb").read()).decode()});

                    }}

                </style>
                """,
        unsafe_allow_html=True,
    )
    Dash_title = '<p style="font-family:sans-serif; font-weight:bold; color: #1D56C0; font-size: 42px;">DataBase</p>'
    st.markdown(Dash_title, unsafe_allow_html=True)

    dash_data = pd.read_csv(os.getcwd() + os.sep + 'dash_data.csv')

    if case == "일":
        plot_case = 'day'

    if case == "요일":
        plot_case = 'weekday'

    if case == "시간":
        plot_case = 'time'


    if len(filter) == 1:
        filter = filter
        filter_data = dash_data.loc[dash_data[plot_case] == filter[0]]

    elif len(filter) >= 2:

        if plot_case == 'time':
            dash_data['time'] = pd.to_datetime(dash_data['time'], errors='coerce')
            dash_data['time'] = dash_data['time'].dt.hour

        if plot_case == 'weekday':

            if len(filter) == 2:
                filter_1, filter_2 = filter
                filter_data = dash_data.loc[(dash_data[plot_case] == filter_1) |\
                                            (dash_data[plot_case] == filter_2)]
            if len(filter) == 3:
                filter_1, filter_2, filter_3 = filter
                filter_data = dash_data.loc[(dash_data[plot_case] == filter_1) |\
                                            (dash_data[plot_case] == filter_2) |\
                                            (dash_data[plot_case] == filter_3)]
            if len(filter) == 4:
                filter_1, filter_2, filter_3, filter_4 = filter
                filter_data = dash_data.loc[(dash_data[plot_case] == filter_1) |\
                                            (dash_data[plot_case] == filter_2) |\
                                            (dash_data[plot_case] == filter_3) |\
                                            (dash_data[plot_case] == filter_4)]
            if len(filter) == 5:
                filter_1, filter_2, filter_3, filter_4, filter_5 = filter
                filter_data = dash_data.loc[(dash_data[plot_case] == filter_1) |\
                                            (dash_data[plot_case] == filter_2) |\
                                            (dash_data[plot_case] == filter_3) |\
                                            (dash_data[plot_case] == filter_4) |\
                                            (dash_data[plot_case] == filter_5)]
            if len(filter) == 6:
                filter_1, filter_2, filter_3, filter_4, filter_5, filter_6 = filter
                filter_data = dash_data.loc[(dash_data[plot_case] == filter_1) |\
                                            (dash_data[plot_case] == filter_2) |\
                                            (dash_data[plot_case] == filter_3) |\
                                            (dash_data[plot_case] == filter_4) |\
                                            (dash_data[plot_case] == filter_5) |\
                                            (dash_data[plot_case] == filter_6)]
            if len(filter) == 7:
                filter_1, filter_2, filter_3, filter_4, filter_5, filter_6, filter_7 = filter
                filter_data = dash_data.loc[(dash_data[plot_case] == filter_1) |\
                                            (dash_data[plot_case] == filter_2) |\
                                            (dash_data[plot_case] == filter_3) |\
                                            (dash_data[plot_case] == filter_4) |\
                                            (dash_data[plot_case] == filter_5) |\
                                            (dash_data[plot_case] == filter_6) |\
                                            (dash_data[plot_case] == filter_7)]

        else:
            filter_1, filter_2 = filter
            filter_data = dash_data.loc[(dash_data[plot_case] >= filter_1) & (dash_data[plot_case] <= filter_2)]

    fire = filter_data.loc[:, ['fire', plot_case]]
    safe = filter_data.loc[:, ['safe', plot_case]]
    danger = filter_data.loc[:, ['danger', plot_case]]
    total_data = filter_data.loc[:, ['fire', 'safe', 'danger', plot_case]]

    fire.columns = map(lambda x: str(x).upper(), fire.columns)
    safe.columns = map(lambda x: str(x).upper(), safe.columns)
    danger.columns = map(lambda x: str(x).upper(), danger.columns)
    total_data.columns = map(lambda x: str(x).upper(), total_data.columns)


    fig1 = go.Figure(data=[go.Table(
        header=dict(values=list(total_data.columns),
                    fill_color='lightskyblue',
                    align='center'),
        cells=dict(values=[total_data['FIRE'], total_data['SAFE'], total_data['DANGER'], total_data[plot_case.upper()]],
                   fill_color='lightcyan',
                   align='center'))
    ])

    fig1.update_layout(
        width=1000,
        height=350,
        font_size = 12,
        margin=dict(
            l=0,
            r=0,
            b=0,
            t=0
        )
    )

    st.plotly_chart(fig1)

    db_col1, db_col2, db_col3 = st.columns(3)

    with db_col1:
        fig2 = go.Figure(data=[go.Table(
            header=dict(values=list(fire.columns),
                        fill_color='lightskyblue',
                        align='center'),
            cells=dict(values=[fire['FIRE'], fire[plot_case.upper()]],
                       fill_color='lightcyan',
                       align='center'))
        ])

        fig2.update_layout(
            width=216,
            height=400,
            font_size=12,
            margin=dict(
                l=0,
                r=0,
                b=0,
                t=0
            )
        )

        st.plotly_chart(fig2)

    with db_col2:
        fig3 = go.Figure(data=[go.Table(
            header=dict(values=list(safe.columns),
                        fill_color='lightskyblue',
                        align='center'),
            cells=dict(values=[safe['SAFE'], safe[plot_case.upper()]],
                       fill_color='lightcyan',
                       align='center'))
        ])

        fig3.update_layout(
            width=216,
            height=400,
            font_size=12,
            margin=dict(
                l=0,
                r=0,
                b=0,
                t=0
            )
        )

        st.plotly_chart(fig3)

    with db_col3:
        fig4 = go.Figure(data=[go.Table(
            header=dict(values=list(danger.columns),
                        fill_color='lightskyblue',
                        align='center'),
            cells=dict(values=[danger['DANGER'], danger[plot_case.upper()]],
                       fill_color='lightcyan',
                       align='center'))
        ])

        fig4.update_layout(
            width=216,
            height=400,
            font_size=12,
            margin=dict(
                l=0,
                r=0,
                b=0,
                t=0
            )
        )

        st.plotly_chart(fig4)

    filter = ""

db_expander = st.sidebar.expander("데이터베이스 🗃", expanded=True)
with db_expander:
    case = st.selectbox("형식 선택", ('일', '요일', '시간'), key=1)

    if case == '일':
        filter2 = st.slider('일을 선택해주세요.', 1, 30, (10, 18), key=1)

    if case == '요일':
        filter2 = st.multiselect("요일을 선택해주세요.", ('월', '화', '수', '목', '금', '토', '일'), key=1)

    if case == '시간':
        filter2 = st.slider('시간을 선택해주세요.', 0, 24, (9, 18), key=1)

    st.write("")
    db_start = st.button("Database 확인하기")

if db_start:
    Database(case, filter2)

def home():

    st.markdown(
        f"""
                <style>
                    .reportview-container .main .block-container{{
                        max-width: {1050}px;
                        padding-top: {0.5}rem;
                        padding-right: {1}rem;
                        padding-left: {1}rem;
                        padding-bottom: {1}rem;
                    }}
                    .reportview-container .main {{
                        color: black;
                        background-color: white;
                    }}

                </style>
                """,
        unsafe_allow_html=True,
    )
    st.title("")
    st.write("")

    home_title = '<p style="font-family:sans-serif; font-weight:bold; color: #1D56C0; font-size: 50px;">Welcome to CSM for us</p>'
    st.markdown(home_title, unsafe_allow_html=True)

    home_subtitle = '<p style="font-family:sans-serif; color: #00B0F0; font-size: 21px;">CSM(Construction Safety Monitoring & Management) is a service that think your safety first.</p>'
    st.markdown(home_subtitle, unsafe_allow_html=True)

    home_image = Image.open(os.getcwd() + os.sep + "Home_image" + os.sep + "home_image.png")
    st.image(home_image, width = 1100)

def Service_info():
    st.markdown(
        f"""
                <style>
                    .reportview-container .main .block-container{{
                        max-width: {1050}px;
                        padding-top: {0.5}rem;
                        padding-right: {1}rem;
                        padding-left: {1}rem;
                        padding-bottom: {1}rem;
                    }}
                    .reportview-container .main {{
                        color: black;
                        background-color: white;
                    }}

                </style>
                """,
        unsafe_allow_html=True,
    )

    Service_title = '<p style="font-family:sans-serif; font-weight:bold; color: #1D56C0; font-size: 42px;">About CSM for us</p>'
    st.markdown(Service_title, unsafe_allow_html=True)

    f_col1, f_col2 = st.columns([1.5, 2.5])
    with f_col1:
        safe_image = Image.open(os.getcwd() + os.sep + "Service_image" + os.sep + "service.png")
        st.image(safe_image, use_column_width=True)

    with f_col2:
        Service_subtitle = '<p style="font-family:sans-serif; font-weight:bold; color: #0283EE; font-size: 30px;">서비스 개요</p>'
        st.markdown(Service_subtitle, unsafe_allow_html=True)

        st.title("")

        Service_contents = '<p style="font-family:sans-serif; color: black; font-size: 20px;">"CSM for us"는 공사현장의 작업자와 관리자 모두를 위한</p>'
        st.markdown(Service_contents, unsafe_allow_html=True)

        Service_contents3 = '<p style="font-family:sans-serif; color: black; font-size: 20px;">공사현장 안전관리 솔루션입니다.</p>'
        st.markdown(Service_contents3, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        image1 = Image.open(os.getcwd() + os.sep + "Service_image" + os.sep + "service1.png")
        st.image(image1, use_column_width=True)

    with col2:
        image2 = Image.open(os.getcwd() + os.sep + "Service_image" + os.sep + "service2.png")
        st.image(image2, use_column_width=True)

    with col3:
        image2 = Image.open(os.getcwd() + os.sep + "Service_image" + os.sep + "service3.png")
        st.image(image2, use_column_width=True)

if add_side_button_home:
    home()

if add_side_button_service:
    Service_info()
