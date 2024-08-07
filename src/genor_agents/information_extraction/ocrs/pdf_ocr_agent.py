import logging
import os
import time
from typing import Dict, List, Optional, Tuple

import fitz  # PyMuPDF
from azure.ai.formrecognizer import DocumentAnalysisClient
from azure.ai.formrecognizer._models import AnalyzeResult
from azure.core.credentials import AzureKeyCredential

from genor_agents.agent import Agent


class PDFOCRAgent(Agent):
    """
    PDFOCRAgent is a class that represents an OCR (Optical Character Recognition) agent for processing PDF documents.
    Attributes:
        model_name (str): The name of the OCR model to be used.
        max_attempts (int): The maximum number of attempts for the analysis.
    """

    def __init__(self, model_name="prebuilt-document", max_attempts: int = 3) -> None:
        self.client = self._initialize_document_analysis_client()
        self.model_name = model_name
        self.max_attempts = max_attempts
        self.logger = logging.getLogger(__name__)  # Initialize the logger

    def __call__(self, pdf_path: os.PathLike, pages: Optional[List[int]] = []) -> dict:
        """
        Processes a PDF document and returns the OCR results for specified pages.
        Args:
            pdf_path (os.PathLike): The path to the PDF file to be processed.
            pages (List[int], optional): The list of page numbers to be processed. Defaults to [] (process all pages).
        Returns:
            dict: The OCR results with metadata.
        """
        pdf_bytes, total_pages = self._read_pdf_and_get_metadata(pdf_path)
        pages_str = self._format_pages(pages, total_pages)
        self.logger.info(f"Analyzing pages: {pages_str if pages_str else 'all pages'}")
        result = self._analyze_document(pdf_bytes, pages_str)
        output = self._construct_output(result, total_pages, pages)
        self.logger.info(
            f"PDF processing completed. Metadata results: {output['metadata']}"
        )
        return output

    def _analyze_document(self, pdf_bytes: bytes, pages_str: str) -> AnalyzeResult:
        """
        Analyzes the PDF document using Azure Form Recognizer.
        Args:
            pdf_bytes (bytes): The content of the PDF file as bytes.
            pages_str (str): The formatted pages string.
            max_attempts (int, optional): The maximum number of attempts for the analysis. Defaults to 3.
            progress_callback (Callable, optional): A callback function to track the progress of the analysis. Defaults to None.
        Returns:
            AnalyzeResult: The result of the document analysis.
        """
        attempt = 0
        while attempt < self.max_attempts:
            try:
                self.logger.info(
                    f"Starting document analysis for pages: {pages_str if pages_str else 'all'}, attempt {attempt + 1}"
                )
                poller = self.client.begin_analyze_document(
                    model_id=self.model_name,
                    document=pdf_bytes,
                    pages=pages_str,
                )
                result = poller.result()
                self.logger.info("Document analysis completed.")
                return result
            except Exception as e:
                self.logger.error(
                    f"Failed to analyze document on attempt {attempt + 1}: {e}"
                )
                attempt += 1
                if attempt < self.max_attempts:
                    self.logger.info(
                        f"Retrying document analysis in 5 seconds... (Attempt {attempt + 1}/{self.max_attempts})"
                    )
                    time.sleep(5)
                else:
                    self.logger.error(
                        "Failed to analyze document on all attempts. Aborting."
                    )
                    raise

    def _construct_output(
        self, result: AnalyzeResult, total_pages: int, pages: Optional[List[int]]
    ) -> dict:
        """
        Constructs the output dictionary with OCR results and metadata.
        Args:
            result (AnalyzeResult): The result of the document analysis.
            total_pages (int): The total number of pages in the PDF.
            pages (List[int], optional): The list of page numbers to be processed.
        Returns:
            dict: The output dictionary.
        """
        output = {
            "metadata": {
                "total_pages": total_pages,
                "pages_analyzed": pages if pages else list(range(1, total_pages + 1)),
            },
            "raw_ocr_result": result,
            "content": result.content.replace('\n', ' '),
            "pages_content": {
                page.page_number: " ".join([line.content for line in page.lines]) for page in result.pages
            },
        }
        
        return output

    def _initialize_document_analysis_client(self):
        """
        Initializes the Document Analysis Client for Azure Form Recognizer.
        Returns:
            DocumentAnalysisClient: The initialized Document Analysis Client.
        """
        return DocumentAnalysisClient(
            endpoint=os.getenv("AZURE_UAE_ENDPOINT"),
            credential=AzureKeyCredential(os.getenv("AZURE_UAE_ENDPOINT_KEY")),
        )

    def _read_pdf_and_get_metadata(self, pdf_path: str) -> Tuple[bytes, int]:
        """
        Reads a PDF file and returns its content as bytes along with the total number of pages.
        Args:
            pdf_path (str): The path to the PDF file to be read.
        Returns:
            tuple: The content of the PDF file as bytes and the total number of pages.
        """
        try:
            with open(pdf_path, "rb") as file:
                pdf_bytes = file.read()
                pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
                total_pages = pdf_document.page_count
            return pdf_bytes, total_pages
        except FileNotFoundError:
            self.logger.error(f"The file was not found. {pdf_path}")
            raise FileNotFoundError("The file was not found.")
        except Exception as e:
            self.logger.error(f"Failed to read the file: {e}")
            raise ValueError(f"Failed to read the file: {e}")

    def _format_pages(self, pages: List[int], total_pages: int) -> str:
        """
        Formats the list of pages into a string suitable for the Azure Form Recognizer API.
        Args:
            pages (List[int]): The list of page numbers to be processed.
            total_pages (int): The total number of pages in the PDF.
        Returns:
            str: The formatted pages string.
        """
        valid_pages = sorted(set(p for p in pages if 1 <= p <= total_pages))
        if not valid_pages:
            return ""

        formatted_pages = []
        range_start = valid_pages[0]
        range_end = valid_pages[0]

        for i in range(1, len(valid_pages)):
            if valid_pages[i] == range_end + 1:
                range_end = valid_pages[i]
            else:
                if range_start == range_end:
                    formatted_pages.append(f"{range_start}")
                else:
                    formatted_pages.append(f"{range_start}-{range_end}")
                range_start = valid_pages[i]
                range_end = valid_pages[i]

        if range_start == range_end:
            formatted_pages.append(f"{range_start}")
        else:
            formatted_pages.append(f"{range_start}-{range_end}")

        return ",".join(formatted_pages)


def run_example():
    pdf_path = os.path.join(
        os.getenv("CASES_DIRS_PATH"),
        "case-300000203070",
        "case-300000203070-attachments",
        "6AA93CF904021EDDA2E810C435D3CEEC.pdf",
    )

    # Specify the pages you want to analyze, e.g., [1, 2, 4, 5]
    pages_to_analyze = []

    pdf_ocr_agent = PDFOCRAgent()
    res = pdf_ocr_agent(pdf_path, pages_to_analyze)
    print(res)


if __name__ == "__main__":
    run_example()
