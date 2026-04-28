import argparse
import asyncio
import json
import re
from pathlib import Path
from typing import Any, List

from mcp_server_qdrant.embeddings.factory import create_embedding_provider
from mcp_server_qdrant.qdrant import Entry, QdrantConnector
from mcp_server_qdrant.settings import EmbeddingProviderSettings, QdrantSettings

# --- CONFIGURACIÓN DE CHUNKING ---
CHUNK_SIZE = 1000  # Caracteres por fragmento
CHUNK_OVERLAP = 200 # Solapamiento para mantener el contexto entre fragmentos

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Vectoriza un .txt usando CHUNKING y lo guarda en Qdrant."
        )
    )
    parser.add_argument("txt_path", type=Path, help="Ruta del archivo .txt")
    parser.add_argument("collection", type=str, help="Nombre de la colección")
    parser.add_argument("--metadata-json", type=str, default=None, help="JSON con metadata extra")
    parser.add_argument("--qdrant-url", type=str, default=None, help="URL de Qdrant")
    parser.add_argument("--qdrant-api-key", type=str, default=None, help="API Key")
    parser.add_argument("--qdrant-local-path", type=str, default=None, help="Modo local path")
    parser.add_argument("--embedding-model", type=str, default=None, help="Modelo de embedding")
    return parser.parse_args(argv)

def _read_text_file(path: Path) -> str:
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"No existe el archivo: {path}")
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="latin-1")

def _get_chunks(text: str, size: int, overlap: int) -> List[str]:
    """Divide el texto en fragmentos intentando respetar los saltos de línea."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        # Intentamos ajustar el corte al último salto de línea para no romper párrafos
        if end < len(text):
            last_break = text.rfind('\n', start + overlap, end)
            if last_break != -1:
                end = last_break
        
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        
        start = end - overlap
        if start >= len(text) or end >= len(text):
            break
    return chunks

def _build_metadata(txt_path: Path, extra_metadata: dict[str, Any] | None, chunk_idx: int) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "source_name": txt_path.name,
        "chunk_index": chunk_idx,
        "source_type": "txt",
    }
    if extra_metadata:
        metadata.update(extra_metadata)
    return metadata

async def _run(args: argparse.Namespace) -> None:
    qdrant_settings = QdrantSettings()
    embedding_settings = EmbeddingProviderSettings()

    # Overrides
    if args.qdrant_url: qdrant_settings.location = args.qdrant_url
    if args.qdrant_api_key: qdrant_settings.api_key = args.qdrant_api_key
    if args.qdrant_local_path: qdrant_settings.local_path = args.qdrant_local_path
    if args.embedding_model: embedding_settings.model_name = args.embedding_model

    extra_metadata = json.loads(args.metadata_json) if args.metadata_json else {}

    content = _read_text_file(args.txt_path).strip()
    if not content:
        raise ValueError("El archivo está vacío.")

    # Inicializar componentes
    embedding_provider = create_embedding_provider(embedding_settings)
    connector = QdrantConnector(
        qdrant_url=qdrant_settings.location,
        qdrant_api_key=qdrant_settings.api_key,
        collection_name=None,
        embedding_provider=embedding_provider,
        qdrant_local_path=qdrant_settings.local_path,
    )

    # --- PROCESO DE CHUNKING ---
    print(f"Dividiendo archivo '{args.txt_path.name}' en fragmentos...")
    chunks = _get_chunks(content, CHUNK_SIZE, CHUNK_OVERLAP)
    print(f"Total de fragmentos a vectorizar: {len(chunks)}")

    for i, chunk_text in enumerate(chunks):
        metadata = _build_metadata(args.txt_path, extra_metadata, i)
        entry = Entry(content=chunk_text, metadata=metadata)
        
        # Guardar en Qdrant
        await connector.store(entry, collection_name=args.collection)
        print(f" [OK] Fragmento {i+1}/{len(chunks)} guardado.")

    print("\n✅ Proceso completado con éxito.")

def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    asyncio.run(_run(args))

if __name__ == "__main__":
    main()