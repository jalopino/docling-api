# Docling 2.25 → 2.74 compatibility

We use the following Docling APIs. As of 2.74 they remain supported:

- **Imports**: `docling.datamodel.base_models` (InputFormat, DocumentStream), `docling.datamodel.pipeline_options` (PdfPipelineOptions, EasyOcrOptions), `docling.document_converter` (PdfFormatOption, DocumentConverter), `docling_core.types.doc` (ImageRefMode, TableItem, PictureItem).
- **Pipeline options**: `do_ocr`, `ocr_options` (EasyOcrOptions with `lang`, `force_full_page_ocr`), `images_scale`, `generate_table_images`, `generate_picture_images`, `generate_page_images`.
- **Conversion**: `DocumentConverter.convert()` / `convert_all()` with `DocumentStream`, `raises_on_error`, `max_num_pages`.
- **Output**: `conv_res.document.export_to_markdown(image_mode=ImageRefMode.PLACEHOLDER)`, `conv_res.document.iterate_items()`.

## After upgrading

1. **Lockfile**: Run `poetry lock && poetry install` (or `poetry update docling`).
2. **Docker build**: The Dockerfile pre-downloads models with `StandardPdfPipeline.download_models_hf(force=True)`. If that line fails in 2.74, use the CLI instead, e.g. `RUN docling-tools models download` (see [Docling docs](https://docling-project.github.io/docling/reference/pipeline_options/)).
3. **Smoke test**: Convert a small PDF via the API and check markdown and (if enabled) images.

## 2.74 release notes (relevant)

- docling-parse v5; old parse backends deprecated (we use the high-level DocumentConverter, so no change required).
- CSV: default delimiter set by default; we use custom CSV handling in `document_converter.utils.handle_csv_file`, so behavior should be unchanged.
- Security: XXE-related fixes (no API change).
