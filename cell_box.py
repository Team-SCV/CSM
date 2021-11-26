from typing import List, Optional
import markdown
import streamlit as st

COLOR = "white"
BACKGROUND_COLOR = "black"



def cell_box_main(fire_sum, safe_sum, danger_sum):
    # My preliminary idea of an API for generating a grid
    safe_point = round(safe_sum/(danger_sum+safe_sum+fire_sum) * 100, 2)
    danger_point = round(danger_sum/(danger_sum+safe_sum+fire_sum) * 100, 2)
    fire_point = round(fire_sum/(danger_sum+safe_sum+fire_sum) * 100,2)

    dash_content1 = '<p style="font-family:sans-serif; font-weight:bold; color: black; font-size: 20px;">선택 기간중 항목별 탐지결과</p>'
    dash_content2 = '<p style="font-family:sans-serif; font-weight:bold; color: black; font-size: 16px;">- 안전 장비 착용 준수 탐지 수 : {:,}명</p>'.format(safe_sum)
    dash_content3 = '<p style="font-family:sans-serif; font-weight:bold; color: black; font-size: 16px;">- 안전 장비 착용 미준수 탐지 수 : {:,}명</p>'.format(danger_sum)
    dash_content4 = '<p style="font-family:sans-serif; font-weight:bold; color: black; font-size: 16px;">- 화재 전조 탐지 수 : {:,}건</p>'.format(fire_sum)

    dash_content5 = '<p style="font-family:sans-serif; font-weight:bold; color: black; font-size: 12.5px;">선택 기간중</p>'
    dash_content5_1 = '<p style="font-family:sans-serif; font-weight:bold; color: black; font-size: 12.5px;">공사현장 안전 지수</p>'
    dash_content6 = '<p style="font-family:sans-serif; font-weight:bold; text-align:center; color: black; font-size: 40px;">{}</p>'.format(safe_point)

    dash_content7 = '<p style="font-family:sans-serif; font-weight:bold; color: black; font-size: 12.5px;">선택 기간중</p>'
    dash_content7_1 = '<p style="font-family:sans-serif; font-weight:bold; color: black; font-size: 12.5px;">공사현장 불안전 지수</p>'
    dash_content8 = '<p style="font-family:sans-serif; font-weight:bold; text-align:center; color: black; font-size: 40px;">{}</p>'.format(danger_point)

    dash_content9 = '<p style="font-family:sans-serif; font-weight:bold; color: black; font-size: 12.5px;">선택 기간중</p>'
    dash_content9_1 = '<p style="font-family:sans-serif; font-weight:bold; color: black; font-size: 12.5px;">공사현장 화재 위험 지수</p>'
    dash_content10 = '<p style="font-family:sans-serif; font-weight:bold; text-align:center; color: black; font-size: 40px;">{}</p>'.format(fire_point)


    with Grid("1 1 1 1", color=COLOR, background_color=BACKGROUND_COLOR) as grid:

        g1 = grid.cell(
                class_="a",
                grid_column_start=1,
                grid_column_end=3,
                grid_row_start=3,
                grid_row_end=5,
            )

        g2 = grid.cell(
            class_="b",
            grid_column_start=3,
            grid_column_end=4,
            grid_row_start=3,
            grid_row_end=5,
        )

        g3 = grid.cell(
            class_="c",
            grid_column_start=4,
            grid_column_end=5,
            grid_row_start=3,
            grid_row_end=5,
        )

        g4 = grid.cell(
            class_="d",
            grid_column_start=5,
            grid_column_end=6,
            grid_row_start=3,
            grid_row_end=5,
        )

        g1.markdown(f"{dash_content1}{dash_content2}{dash_content3}{dash_content4}")
        g2.markdown(f"{dash_content5}{dash_content5_1}{dash_content6}")
        g3.markdown(f"{dash_content7}{dash_content7_1}{dash_content8}")
        g4.markdown(f"{dash_content9}{dash_content9_1}{dash_content10}")

class Cell:
    """A Cell can hold text, markdown, plots etc."""

    def __init__(
            self,
            class_: str = None,
            grid_column_start: Optional[int] = None,
            grid_column_end: Optional[int] = None,
            grid_row_start: Optional[int] = None,
            grid_row_end: Optional[int] = None,
    ):
        self.class_ = class_
        self.grid_column_start = grid_column_start
        self.grid_column_end = grid_column_end
        self.grid_row_start = grid_row_start
        self.grid_row_end = grid_row_end
        self.inner_html = ""

    def _to_style(self) -> str:
        return f"""
.{self.class_} {{
    grid-column-start: {self.grid_column_start};
    grid-column-end: {self.grid_column_end};
    grid-row-start: {self.grid_row_start};
    grid-row-end: {self.grid_row_end};
}}
"""

    def text(self, text: str = ""):
        self.inner_html = text

    def markdown(self, text):
        self.inner_html = markdown.markdown(text)

    def _to_html(self):
        return f"""<div class="box {self.class_}">{self.inner_html}</div>"""


class Grid:
    """A (CSS) Grid"""

    def __init__(
            self,
            template_columns="1 1 1 1",
            gap="30px",
            background_color=COLOR,
            color=BACKGROUND_COLOR,
    ):
        self.template_columns = template_columns
        self.gap = gap
        self.background_color = background_color
        self.color = color
        self.cells: List[Cell] = []

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        st.markdown(self._get_grid_style(), unsafe_allow_html=True)
        st.markdown(self._get_cells_style(), unsafe_allow_html=True)
        st.markdown(self._get_cells_html(), unsafe_allow_html=True)

    def _get_grid_style(self):
        return f"""
<style>
    .wrapper {{
    display: grid;
    justify-content :space-between;
    grid-template-columns: 180px 180px 180px 180px 180px;
    grid-gap : 0px 10px;
    background-color: white;
    color: black;
    }}

    .box {{
    background: #83a4d4;  /* fallback for old browsers */
    background: -webkit-linear-gradient(to left, #b6fbff, #83a4d4);  /* Chrome 10-25, Safari 5.1-6 */
    background: linear-gradient(to left, #b6fbff, #83a4d4); /* W3C, IE 10+/ Edge, Firefox 16+, Chrome 26+, Opera 12+, Safari 7+ */
    color: black;
    border-radius: 5px;
    font-weight: bold;
    padding: 20px;
    font-size: 100%;
    }}
</style>
"""

    def _get_cells_style(self):
        return (
                "<style>"
                + "\n".join([cell._to_style() for cell in self.cells])
                + "</style>"
        )

    def _get_cells_html(self):
        return (
                '<div class="wrapper">'
                + "\n".join([cell._to_html() for cell in self.cells])
                + "</div>"
        )

    def cell(
            self,
            class_: str = None,
            grid_column_start: Optional[int] = None,
            grid_column_end: Optional[int] = None,
            grid_row_start: Optional[int] = None,
            grid_row_end: Optional[int] = None,
    ):
        cell = Cell(
            class_=class_,
            grid_column_start=grid_column_start,
            grid_column_end=grid_column_end,
            grid_row_start=grid_row_start,
            grid_row_end=grid_row_end,
        )
        self.cells.append(cell)
        return cell