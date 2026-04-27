from typing import List, Dict, Tuple, Optional
from pathlib import Path
import hashlib
import os
import math
from unstructured.partition.pdf import partition_pdf
from unstructured.documents.elements import Image, Text, Title, NarrativeText, Table
from llama_index.core.schema import Document, TextNode
from utils.logger import setup_logger

logger = setup_logger(__name__)


class MultimodalPDFProcessor:
    """多模态PDF处理器 - 支持提取图片、文本和表格"""

    def __init__(self, image_output_dir: str = "file/images"):
        self.image_output_dir = image_output_dir
        os.makedirs(image_output_dir, exist_ok=True)

    def extract_images_and_text(self, pdf_path: str) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """从PDF中提取图片和对应的文档内容"""
        logger.info(f"开始解析PDF文件: {pdf_path}")

        # 解析PDF
        raw_pdf_elements = partition_pdf(
            filename=pdf_path,  # 指定要解析的 PDF 文件路径
            extract_images_in_pdf=True,  # 是否提取 PDF 中的图片（将其作为 ImageBlock 返回）
            infer_table_structure=True,  # 是否尝试识别并解析表格结构（会输出结构化的 Table 类型元素）
            strategy='hi_res',  # 提取策略：使用高分辨率模式（hi_res）+ OCR，可处理图像型或复杂布局 PDF
            extract_image_block_output_dir=self.image_output_dir  # 图片输出目录，提取出的图片会保存到这个路径
        )

        images = []
        texts = []
        tables = []

        logger.info(f"总共解析到 {len(raw_pdf_elements)} 个元素")

        # 遍历所有元素
        for i, element in enumerate(raw_pdf_elements):
            element_info = {
                'type': type(element).__name__,
                'page': getattr(element.metadata, 'page_number', None) if hasattr(element, 'metadata') else None,  # 获取页码
                'bbox': getattr(element.metadata, 'coordinates', None) if hasattr(element, 'metadata') else None,  # 获取坐标系
                'content': str(element),
                'element_id': f"{Path(pdf_path).stem}_{i}"
            }

            # 处理图片元素
            if isinstance(element, Image):
                image_info = self._process_image_element(element, element_info, pdf_path)
                if image_info:
                    images.append(image_info)

            # 处理文本元素
            elif isinstance(element, (Text, Title, NarrativeText)):
                texts.append(element_info)

            # 处理表格元素
            elif isinstance(element, Table):
                tables.append(element_info)

        logger.info(f"提取完成 - 图片: {len(images)}, 文本: {len(texts)}, 表格: {len(tables)}")
        return images, texts, tables

    def _process_image_element(self, element: Image, element_info: Dict, pdf_path: str) -> Optional[Dict]:
        """处理图片元素"""
        image_info = element_info.copy()

        # 获取图片的base64数据
        if hasattr(element.metadata, 'image_base64') and element.metadata.image_base64:
            image_info['image_data'] = element.metadata.image_base64
            image_info['image_format'] = getattr(element.metadata, 'image_mime_type', 'image/png')

        # 获取图片路径
        elif hasattr(element.metadata, 'image_path') and element.metadata.image_path:
            image_info['image_path'] = element.metadata.image_path

        # 检查其他可能的图片数据属性
        elif hasattr(element.metadata, 'image_data') and element.metadata.image_data:
            image_info['image_data'] = element.metadata.image_data

        return image_info if any(key in image_info for key in ['image_data', 'image_path']) else None

    def euclidean_distance(self, p1: float, p2: float) -> float:
        """计算欧几里得距离"""
        return math.sqrt((p1 - p2) ** 2)

    def get_context_around_image(self, images: List[Dict], texts: List[Dict]) -> List[Dict]:
        """获取图片周围的文本内容作为上下文"""
        for image in images:
            image_page = image['page']  # 获取页码
            image_bbox = image['bbox']  # 获取坐标

            if not image_page or not image_bbox:
                continue

            # 找到同一页最近的文本
            nearest_text = None  # 创建一个文档对应的副本
            min_dist = float("inf")  # 创建一个无穷大的浮点数

            # 计算图片和哪个文档最接近
            for text in texts:
                text_page = text['page']
                text_bbox = text['bbox']

                # 检查是否在相同页面
                if text_page == image_page and text_bbox and image_bbox:
                    try:
                        # 图片的坐标
                        image_top = int(image_bbox.points[0][1])  # 图片的上面坐标
                        image_bottom = int(image_bbox.points[1][1])  # 图片的下面左边

                        # 文本的坐标
                        text_top = int(text_bbox.points[0][1])  # 文本的上面坐标
                        text_bottom = int(text_bbox.points[1][1])  # 文本的下面左边

                        # 计算距离
                        if image_top > text_bottom:  # 文字在图片上面
                            dist = self.euclidean_distance(image_top, text_bottom)
                        else:  # 文字在图片下面
                            dist = self.euclidean_distance(text_top, image_bottom)

                        if dist < min_dist:
                            min_dist = dist
                            # 将文档进行赋值
                            nearest_text = text

                    except (IndexError, ValueError, AttributeError) as e:
                        logger.warning(f"处理坐标时出错: {e}")
                        continue

            # 创建图片-文本对
            if nearest_text:
                # 将当前图片映射到文档中
                nearest_text["image_path"] = image["image_path"]

        # 最终返回内容
        results = []
        # 每个块内容
        text_chunk = {}
        # 每个块重复数据
        text_chunk_overlap = ""
        # 每个块的长度
        text_length = 0
        for text in texts:
            # 如果文本长度为0，创建一个新的文本块，将文本和图片信息(如果有)填充
            if text_length == 0:
                text_chunk["content"] = text_chunk_overlap + text["content"]
                text_chunk["image_path"] = text['image_path'] if 'image_path' in text else None
                text_length += len(text["content"])
            else:
                # 如果如果文本长度不为0，将文本和图片信息(如果有)填充
                text_chunk["content"] += text["content"]
                # 如果包含图片路径就存入最终的文档块中
                if 'image_path' in text:
                    text_chunk["image_path"] = text['image_path']
                text_length += len(text["content"])
                # 如果文本块的长度大于或等于512将文本块存入最终的返回值中，将文本块内容清空，文本块长度设置为0
                if text_length >= 512:
                    results.append(text_chunk)
                    text_chunk_overlap = text_chunk["content"][-50:]
                    text_chunk = {}
                    text_length = 0
        return results

    def create_multimodal_nodes(self, pdf_path: str) -> List[TextNode]:
        """创建多模态文档"""
        images, texts, tables = self.extract_images_and_text(pdf_path)
        # 处理图片和文字之间的关联关系
        texts = self.get_context_around_image(images, texts)

        text_nodes = []

        # 创建纯文本文档，组装llamaIndex支持的文档节点数据
        for text in texts:
            doc = TextNode(
                text=text['content'],
                metadata={
                    "image_path": text['image_path'] if 'image_path' in text else None,
                    "file_name": Path(pdf_path).name,
                    "doc_id": hashlib.md5(Path(pdf_path).name.encode('utf-8')).hexdigest()
                }
            )
            text_nodes.append(doc)

        # # 创建表格文档
        # for table in tables:
        #     doc = Document(
        #         text=table['content'],
        #         metadata={
        #             'source': pdf_path,
        #             'page': table['page'],
        #             'type': 'table',
        #             'element_id': table['element_id']
        #
        #         }
        #     )
        #     text_documents.append(doc)
        return text_nodes
