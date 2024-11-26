from docling.document_converter import DocumentConverter
from docling.datamodel.base_models import InputFormat
from docling.document_converter import (
    DocumentConverter,
    PdfFormatOption,
    WordFormatOption,
)
from docling.pipeline.simple_pipeline import SimplePipeline
from docling.pipeline.standard_pdf_pipeline import StandardPdfPipeline
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    TableFormerMode,
    TableStructureOptions,
)
from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend

pipeline_options = PdfPipelineOptions()
pipeline_options.do_ocr = True
pipeline_options.do_table_structure = True
pipeline_options.table_structure_options.mode = TableFormerMode.ACCURATE

table_structure_options = TableStructureOptions()
table_structure_options.do_cell_matching = True

doc_converter = DocumentConverter(
    allowed_formats=[
        InputFormat.PDF,
        InputFormat.IMAGE,
        InputFormat.DOCX,
        InputFormat.HTML,
        InputFormat.PPTX,
    ],  # whitelist formats, non-matching files are ignored.
    format_options={
        InputFormat.PDF: PdfFormatOption(
            pipeline_options=pipeline_options,  # pipeline options go here.
            backend=PyPdfiumDocumentBackend,  # optional: pick an alternative backend
            table_structure_options=table_structure_options,
        ),
        InputFormat.DOCX: WordFormatOption(
            pipeline_cls=SimplePipeline  # default for office formats and HTML
        ),
    },
)
