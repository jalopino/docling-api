import gc
import logging
import sys
from abc import ABC, abstractmethod
from io import BytesIO
from typing import List, Tuple

from celery.result import AsyncResult
from docling.datamodel.base_models import InputFormat, DocumentStream
from docling.datamodel.pipeline_options import VlmConvertOptions, VlmPipelineOptions
from docling.document_converter import PdfFormatOption, DocumentConverter
from docling.pipeline.vlm_pipeline import VlmPipeline
from docling_core.types.doc import ImageRefMode
from fastapi import HTTPException

from document_converter.schema import BatchConversionJobResult, ConversationJobResult, ConversionResult
from document_converter.utils import handle_csv_file

logging.basicConfig(level=logging.INFO)
IMAGE_RESOLUTION_SCALE = 4

# Granite Docling VLM pipeline (replaces EasyOCR): single vision-language model for document understanding.
_VLM_OPTIONS = VlmPipelineOptions(vlm_options=VlmConvertOptions.from_preset("granite_docling"))


def _release_conversion_memory() -> None:
    """Release memory after a conversion to avoid growth across requests."""
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


class DocumentConversionBase(ABC):
    @abstractmethod
    def convert(self, document: Tuple[str, BytesIO], **kwargs) -> ConversionResult:
        pass

    @abstractmethod
    def convert_batch(self, documents: List[Tuple[str, BytesIO]], **kwargs) -> List[ConversionResult]:
        pass


class DoclingDocumentConversion(DocumentConversionBase):
    """Document conversion using Docling with Granite Docling VLM pipeline.

    Uses the vision-language model (Granite Docling) for end-to-end document understanding;
    no separate OCR pipeline. extract_tables and image_resolution_scale are kept for API
    compatibility but are not used by the VLM pipeline.
    """

    @staticmethod
    def _document_to_markdown(conv_res) -> str:
        """Export document to text-only markdown with page breaks between pages."""
        content_md = conv_res.document.export_to_markdown(
            image_mode=ImageRefMode.PLACEHOLDER,
            page_break_placeholder="\n\n==== PAGE BREAK ====\n\n",
        )
        content_md = content_md.replace("<!-- image -->", "")
        return content_md

    def convert(
        self,
        document: Tuple[str, BytesIO],
        extract_tables: bool = False,
        image_resolution_scale: int = IMAGE_RESOLUTION_SCALE,
    ) -> ConversionResult:
        filename, file = document
        doc_converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_cls=VlmPipeline,
                    pipeline_options=_VLM_OPTIONS,
                )
            }
        )

        if filename.lower().endswith('.csv'):
            file, error = handle_csv_file(file)
            if error:
                return ConversionResult(filename=filename, error=error)

        file.seek(0)
        conv_res = doc_converter.convert(
            DocumentStream(name=filename, stream=file),
            raises_on_error=False,
            max_num_pages=sys.maxsize,
        )
        doc_filename = conv_res.input.file.stem

        if conv_res.errors:
            logging.error(f"Failed to convert {filename}: {conv_res.errors[0].error_message}")
            _release_conversion_memory()
            return ConversionResult(filename=doc_filename, error=conv_res.errors[0].error_message)

        content_md = self._document_to_markdown(conv_res)
        del conv_res
        del doc_converter
        _release_conversion_memory()
        return ConversionResult(filename=doc_filename, markdown=content_md, images=[])

    def convert_batch(
        self,
        documents: List[Tuple[str, BytesIO]],
        extract_tables: bool = False,
        image_resolution_scale: int = IMAGE_RESOLUTION_SCALE,
    ) -> List[ConversionResult]:
        doc_converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_cls=VlmPipeline,
                    pipeline_options=_VLM_OPTIONS,
                )
            }
        )

        for filename, file in documents:
            file.seek(0)
        conv_results = doc_converter.convert_all(
            [DocumentStream(name=filename, stream=file) for filename, file in documents],
            raises_on_error=False,
            max_num_pages=sys.maxsize,
        )

        results = []
        for conv_res in conv_results:
            doc_filename = conv_res.input.file.stem

            if conv_res.errors:
                logging.error(f"Failed to convert {conv_res.input.name}: {conv_res.errors[0].error_message}")
                results.append(ConversionResult(filename=conv_res.input.name, error=conv_res.errors[0].error_message))
                del conv_res
                continue

            content_md = self._document_to_markdown(conv_res)
            results.append(ConversionResult(filename=doc_filename, markdown=content_md, images=[]))
            del conv_res
            _release_conversion_memory()

        del doc_converter
        del conv_results
        _release_conversion_memory()
        return results


class DocumentConverterService:
    def __init__(self, document_converter: DocumentConversionBase):
        self.document_converter = document_converter

    def convert_document(self, document: Tuple[str, BytesIO], **kwargs) -> ConversionResult:
        result = self.document_converter.convert(document, **kwargs)
        if result.error:
            logging.error(f"Failed to convert {document[0]}: {result.error}")
            raise HTTPException(status_code=500, detail=result.error)
        return result

    def convert_documents(self, documents: List[Tuple[str, BytesIO]], **kwargs) -> List[ConversionResult]:
        return self.document_converter.convert_batch(documents, **kwargs)

    def convert_document_task(
        self,
        document: Tuple[str, bytes],
        **kwargs,
    ) -> ConversionResult:
        document = (document[0], BytesIO(document[1]))
        return self.document_converter.convert(document, **kwargs)

    def convert_documents_task(
        self,
        documents: List[Tuple[str, bytes]],
        **kwargs,
    ) -> List[ConversionResult]:
        documents = [(filename, BytesIO(file)) for filename, file in documents]
        return self.document_converter.convert_batch(documents, **kwargs)

    def get_single_document_task_result(self, job_id: str) -> ConversationJobResult:
        """Get the status and result of a document conversion job.

        Returns:
        - IN_PROGRESS: When task is still running
        - SUCCESS: When conversion completed successfully
        - FAILURE: When task failed or conversion had errors
        """

        task = AsyncResult(job_id)
        if task.state == 'PENDING':
            return ConversationJobResult(job_id=job_id, status="IN_PROGRESS")

        elif task.state == 'SUCCESS':
            result = task.get()
            # Check if the conversion result contains an error
            if result.get('error'):
                return ConversationJobResult(job_id=job_id, status="FAILURE", error=result['error'])

            return ConversationJobResult(job_id=job_id, status="SUCCESS", result=ConversionResult(**result))

        else:
            return ConversationJobResult(job_id=job_id, status="FAILURE", error=str(task.result))

    def get_batch_conversion_task_result(self, job_id: str) -> BatchConversionJobResult:
        """Get the status and results of a batch conversion job.

        Returns:
        - IN_PROGRESS: When task is still running
        - SUCCESS: A batch is successful as long as the task is successful
        - FAILURE: When the task fails for any reason
        """

        task = AsyncResult(job_id)
        if task.state == 'PENDING':
            return BatchConversionJobResult(job_id=job_id, status="IN_PROGRESS")

        # Task completed successfully, but need to check individual conversion results
        if task.state == 'SUCCESS':
            conversion_results = task.get()
            job_results = []

            for result in conversion_results:
                if result.get('error'):
                    job_result = ConversationJobResult(status="FAILURE", error=result['error'])
                else:
                    job_result = ConversationJobResult(
                        status="SUCCESS", result=ConversionResult(**result).model_dump(exclude_unset=True)
                    )
                job_results.append(job_result)

            return BatchConversionJobResult(job_id=job_id, status="SUCCESS", conversion_results=job_results)

        return BatchConversionJobResult(job_id=job_id, status="FAILURE", error=str(task.result))
