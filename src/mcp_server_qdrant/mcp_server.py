import csv
import hashlib
import hmac
import json
import logging
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Any, Optional

from fastmcp import Context, FastMCP
from fastmcp.server.http import create_base_app
from pydantic import Field
from qdrant_client import models
from starlette.responses import FileResponse, PlainTextResponse
from starlette.routing import Route

from mcp_server_qdrant.common.filters import make_indexes
from mcp_server_qdrant.common.func_tools import make_partial_function
from mcp_server_qdrant.common.wrap_filters import wrap_filters
from mcp_server_qdrant.embeddings.base import EmbeddingProvider
from mcp_server_qdrant.embeddings.factory import create_embedding_provider
from mcp_server_qdrant.qdrant import ArbitraryFilter, Entry, Metadata, QdrantConnector
from mcp_server_qdrant.settings import (
    DownloadsSettings,
    EmbeddingProviderSettings,
    QdrantSettings,
    ToolSettings,
)

logger = logging.getLogger(__name__)


# FastMCP is an alternative interface for declaring the capabilities
# of the server. Its API is based on FastAPI.
class QdrantMCPServer(FastMCP):
    """
    A MCP server for Qdrant.
    """

    def __init__(
        self,
        tool_settings: ToolSettings,
        qdrant_settings: QdrantSettings,
        embedding_provider_settings: Optional[EmbeddingProviderSettings] = None,
        embedding_provider: Optional[EmbeddingProvider] = None,
        downloads_settings: Optional[DownloadsSettings] = None,
        name: str = "mcp-server-qdrant",
        instructions: str | None = None,
        **settings: Any,
    ):
        self.tool_settings = tool_settings
        self.qdrant_settings = qdrant_settings
        self.downloads_settings = downloads_settings or DownloadsSettings()

        if embedding_provider_settings and embedding_provider:
            raise ValueError(
                "Cannot provide both embedding_provider_settings and embedding_provider"
            )

        if not embedding_provider_settings and not embedding_provider:
            raise ValueError(
                "Must provide either embedding_provider_settings or embedding_provider"
            )

        self.embedding_provider_settings: Optional[EmbeddingProviderSettings] = None
        self.embedding_provider: Optional[EmbeddingProvider] = None

        if embedding_provider_settings:
            self.embedding_provider_settings = embedding_provider_settings
            self.embedding_provider = create_embedding_provider(
                embedding_provider_settings
            )
        else:
            self.embedding_provider_settings = None
            self.embedding_provider = embedding_provider

        assert self.embedding_provider is not None, "Embedding provider is required"

        self.qdrant_connector = QdrantConnector(
            qdrant_settings.location,
            qdrant_settings.api_key,
            qdrant_settings.collection_name,
            self.embedding_provider,
            qdrant_settings.local_path,
            make_indexes(qdrant_settings.filterable_fields_dict()),
        )

        super().__init__(name=name, instructions=instructions, **settings)

        self.setup_tools()

    def format_entry(self, entry: Entry) -> str:
        """
        Feel free to override this method in your subclass to customize the format of the entry.
        """
        entry_metadata = json.dumps(entry.metadata) if entry.metadata else ""
        return f"<entry><content>{entry.content}</content><metadata>{entry_metadata}</metadata></entry>"

    def _public_base_url(self) -> str:
        """
        Base URL used for building clickable download links.

        Override with FASTMCP_PUBLIC_BASE_URL (e.g. "https://example.com").
        """
        configured = self.downloads_settings.public_base_url
        if configured:
            return configured.rstrip("/")

        host = getattr(self.settings, "host", "127.0.0.1")
        port = getattr(self.settings, "port", 8000)

        # If bound to all interfaces, "localhost" is usually the right clickable host for local usage.
        if host in {"0.0.0.0", "::"}:
            host = "127.0.0.1"

        return f"http://{host}:{port}"

    def _downloads_secret(self) -> bytes | None:
        secret = self.downloads_settings.downloads_secret
        if not secret:
            return None
        return secret.encode("utf-8")

    def _downloads_ttl_seconds(self) -> int:
        return self.downloads_settings.downloads_ttl_seconds

    def _download_signature(self, filename: str, exp: int) -> str:
        secret = self._downloads_secret()
        if secret is None:
            raise ValueError("FASTMCP_DOWNLOADS_SECRET is not configured")
        msg = f"{filename}:{exp}".encode("utf-8")
        return hmac.new(secret, msg, hashlib.sha256).hexdigest()

    def _make_signed_download_url(self, filename: str) -> str:
        exp = int(datetime.now(timezone.utc).timestamp()) + self._downloads_ttl_seconds()
        sig = self._download_signature(filename, exp)
        return f"{self._public_base_url()}/downloads/{filename}?exp={exp}&sig={sig}"

    def setup_tools(self):
        """
        Register the tools in the server.
        """

        document_root = Path(self.downloads_settings.document_root).resolve()
        self._downloads_dir = (document_root / "storage/tmp/downloads").resolve()

        async def list_collections(ctx: Context) -> list[str]:
            """
            List all available collections in the Qdrant server.
            :param ctx: The context for the request.
            :return: A list of collection names.
            """
            await ctx.debug("Listing Qdrant collections")
            return await self.qdrant_connector.get_collection_names()

        async def store(
            ctx: Context,
            information: Annotated[str, Field(description="Text to store")],
            collection_name: Annotated[
                str, Field(description="The collection to store the information in")
            ],
            # The `metadata` parameter is defined as non-optional, but it can be None.
            # If we set it to be optional, some of the MCP clients, like Cursor, cannot
            # handle the optional parameter correctly.
            metadata: Annotated[
                Metadata | None,
                Field(
                    description="Extra metadata stored along with memorised information. Any json is accepted."
                ),
            ] = None,
        ) -> str:
            """
            Store some information in Qdrant.
            :param ctx: The context for the request.
            :param information: The information to store.
            :param metadata: JSON metadata to store with the information, optional.
            :param collection_name: The name of the collection to store the information in, optional. If not provided,
                                    the default collection is used.
            :return: A message indicating that the information was stored.
            """
            await ctx.debug(f"Storing information {information} in Qdrant")

            entry = Entry(content=information, metadata=metadata)

            await self.qdrant_connector.store(entry, collection_name=collection_name)
            if collection_name:
                return f"Remembered: {information} in collection {collection_name}"
            return f"Remembered: {information}"

        async def find(
            ctx: Context,
            query: Annotated[str, Field(description="What to search for")],
            collection_name: Annotated[
                str, Field(description="The collection to search in")
            ],
            query_filter: ArbitraryFilter | None = None,
        ) -> list[str] | None:
            """
            Find memories in Qdrant.
            :param ctx: The context for the request.
            :param query: The query to use for the search.
            :param collection_name: The name of the collection to search in, optional. If not provided,
                                    the default collection is used.
            :param query_filter: The filter to apply to the query.
            :return: A list of entries found or None.
            """

            # Log query_filter
            await ctx.debug(f"Query filter: {query_filter}")

            query_filter = models.Filter(**query_filter) if query_filter else None

            await ctx.debug(f"Finding results for query {query}")

            entries = await self.qdrant_connector.search(
                query,
                collection_name=collection_name,
                limit=self.qdrant_settings.search_limit,
                query_filter=query_filter,
            )
            if not entries:
                return None
            content = [
                f"Results for the query '{query}'",
            ]
            for entry in entries:
                content.append(self.format_entry(entry))
            return content

        async def export_csv(
            ctx: Context,
            collection_name: Annotated[
                str, Field(description="The collection to export from")
            ],
            csv_text: Annotated[
                str | None,
                Field(
                    default=None,
                    description=(
                        "CSV final ya generado (texto). Si se proporciona, se guardará TAL CUAL en el fichero y "
                        "se devolverá el link de descarga. Esta es la opción recomendada cuando el modelo genera "
                        "un CSV a partir de la información recuperada de Qdrant."
                    ),
                ),
            ] = None,
            query: Annotated[
                str | None,
                Field(
                    default=None,
                    description=(
                        "Optional semantic query. If provided, the CSV will include the top matches. "
                        "If omitted, the tool will scroll all points."
                    ),
                ),
            ] = None,
            limit: Annotated[
                int,
                Field(
                    default=1000,
                    ge=1,
                    le=100_000,
                    description="Maximum number of rows to export",
                ),
            ] = 1000,
            columns: Annotated[
                list[str] | None,
                Field(
                    default=None,
                    description=(
                        "CSV columns to include. Use 'content' and/or metadata keys "
                        "(e.g. ['content','source','year']). If omitted, exports ['content','metadata']."
                    ),
                ),
            ] = None,
            query_filter: ArbitraryFilter | None = None,
        ) -> str:
            """
            Export data from Qdrant into a CSV saved under `storage/tmp/downloads`.
            """
            await ctx.debug(
                f"Exporting CSV from collection={collection_name}, has_csv_text={bool(csv_text)}, query={query}, limit={limit}"
            )

            self._downloads_dir.mkdir(parents=True, exist_ok=True)

            safe_collection = re.sub(r"[^a-zA-Z0-9_.-]+", "_", collection_name).strip(
                "_"
            )
            ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            file_name = f"export_{safe_collection or 'collection'}_{ts}_{uuid.uuid4().hex[:8]}.csv"
            file_path = self._downloads_dir / file_name

            if csv_text is not None:
                with file_path.open("w", encoding="utf-8", newline="") as f:
                    f.write(csv_text.lstrip("\ufeff"))
                    if not csv_text.endswith("\n"):
                        f.write("\n")
            else:
                q_filter = models.Filter(**query_filter) if query_filter else None

                if query:
                    entries = await self.qdrant_connector.search(
                        query,
                        collection_name=collection_name,
                        limit=limit,
                        query_filter=q_filter,
                    )
                else:
                    entries = await self.qdrant_connector.list_entries(
                        collection_name=collection_name,
                        limit=limit,
                        query_filter=q_filter,
                    )

                if not columns:
                    columns = ["content", "metadata"]

                def value_for_column(entry: Entry, col: str) -> Any:
                    if col == "content":
                        return entry.content
                    if col == "metadata":
                        return (
                            json.dumps(entry.metadata, ensure_ascii=False)
                            if entry.metadata
                            else ""
                        )
                    if not entry.metadata:
                        return ""
                    return entry.metadata.get(col, "")

                with file_path.open("w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(
                        f, fieldnames=columns, extrasaction="ignore"
                    )
                    writer.writeheader()
                    for entry in entries:
                        row = {col: value_for_column(entry, col) for col in columns}
                        writer.writerow(row)

            # Return an absolute signed URL so clients can copy/paste even if they don't render links.
            return self._make_signed_download_url(file_name)

        find_foo = find
        store_foo = store
        export_csv_foo = export_csv

        filterable_conditions = (
            self.qdrant_settings.filterable_fields_dict_with_conditions()
        )

        if len(filterable_conditions) > 0:
            find_foo = wrap_filters(find_foo, filterable_conditions)
            export_csv_foo = wrap_filters(export_csv_foo, filterable_conditions)
        elif not self.qdrant_settings.allow_arbitrary_filter:
            find_foo = make_partial_function(find_foo, {"query_filter": None})
            export_csv_foo = make_partial_function(
                export_csv_foo, {"query_filter": None}
            )

        if self.qdrant_settings.collection_name:
            find_foo = make_partial_function(
                find_foo, {"collection_name": self.qdrant_settings.collection_name}
            )
            store_foo = make_partial_function(
                store_foo, {"collection_name": self.qdrant_settings.collection_name}
            )
            export_csv_foo = make_partial_function(
                export_csv_foo, {"collection_name": self.qdrant_settings.collection_name}
            )

        self.tool(
            find_foo,
            name="qdrant-find",
            description=self.tool_settings.tool_find_description,
        )

        self.tool(
            list_collections,
            name="qdrant-list-collections",
            description=self.tool_settings.tool_list_collections_description,
        )

        self.tool(
            export_csv_foo,
            name="export_csv",
            description=(
                "Guarda un CSV en storage/tmp/downloads y devuelve SOLO el link HTTP absoluto "
                "para descargarlo (por ejemplo: http://127.0.0.1:8000/downloads/archivo.csv). "
                "IMPORTANTE: si ya has generado el CSV final, pásalo en `csv_text` y se guardará TAL CUAL "
                "(esto es lo habitual). Solo si NO tienes el CSV generado, usa `query`/`columns` para exportar "
                "directamente desde Qdrant."
            ),
        )

        if not self.qdrant_settings.read_only:
            # Those methods can modify the database
            self.tool(
                store_foo,
                name="qdrant-store",
                description=self.tool_settings.tool_store_description,
            )

    def http_app(  # type: ignore[override]
        self,
        path: str | None = None,
        middleware=None,
        transport: str = "streamable-http",
    ):
        """
        Override FastMCP's default HTTP app to avoid 307 redirects on `/mcp`.

        Some clients (including OpenAI Agents' remote MCP connector) probe the endpoint with
        `GET /mcp` and do not follow the framework-generated redirect to `/mcp/`, which causes
        tool discovery to fail.
        """
        if transport != "streamable-http":
            return super().http_app(path=path, middleware=middleware, transport=transport)

        from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
        from starlette.applications import Starlette
        from contextlib import asynccontextmanager
        from typing import AsyncGenerator

        streamable_http_path = path or self.settings.streamable_http_path

        session_manager = StreamableHTTPSessionManager(
            app=self._mcp_server,
            event_store=None,
            json_response=self.settings.json_response,
            stateless=self.settings.stateless_http,
        )

        async def download_file(request):
            filename = request.path_params.get("filename", "")
            if not filename or "/" in filename or "\\" in filename or ".." in filename:
                return PlainTextResponse("Invalid filename", status_code=400)

            secret = self._downloads_secret()
            if secret is not None:
                exp_raw = request.query_params.get("exp", "")
                sig = request.query_params.get("sig", "")
                try:
                    exp = int(exp_raw)
                except ValueError:
                    return PlainTextResponse("Missing/invalid signature", status_code=403)

                now = int(datetime.now(timezone.utc).timestamp())
                if exp < now:
                    return PlainTextResponse("Link expired", status_code=403)

                try:
                    expected = self._download_signature(filename, exp)
                except ValueError:
                    return PlainTextResponse("Server not configured", status_code=500)

                if not sig or not hmac.compare_digest(sig, expected):
                    return PlainTextResponse("Missing/invalid signature", status_code=403)

            downloads_dir = getattr(self, "_downloads_dir", None)
            if downloads_dir is None:
                document_root = Path(self.downloads_settings.document_root).resolve()
                downloads_dir = (document_root / "storage/tmp/downloads").resolve()
            file_path = (Path(downloads_dir) / filename).resolve()
            if not str(file_path).startswith(str(Path(downloads_dir).resolve()) + "/"):
                return PlainTextResponse("Invalid path", status_code=400)
            if not file_path.exists() or not file_path.is_file():
                return PlainTextResponse("Not found", status_code=404)

            return FileResponse(
                file_path,
                media_type="text/csv; charset=utf-8",
                filename=filename,
            )

        class _StreamableHttpAsgiApp:
            def __init__(self, manager: StreamableHTTPSessionManager):
                self._manager = manager

            async def __call__(self, scope, receive, send) -> None:
                await self._manager.handle_request(scope, receive, send)

        asgi_app = _StreamableHttpAsgiApp(session_manager)

        routes = [
            # Serve both with and without trailing slash, without redirects.
            Route(
                streamable_http_path.rstrip("/"),
                endpoint=asgi_app,
                methods=["GET", "POST", "OPTIONS"],
            ),
            Route(
                streamable_http_path.rstrip("/") + "/",
                endpoint=asgi_app,
                methods=["GET", "POST", "OPTIONS"],
            ),
            Route(
                "/downloads/{filename:str}",
                endpoint=download_file,
                methods=["GET", "HEAD"],
                name="downloads",
            ),
        ]
        # Preserve any additional routes registered on the server.
        routes.extend(self._additional_http_routes)

        @asynccontextmanager
        async def lifespan(app: Starlette) -> AsyncGenerator[None, None]:
            async with session_manager.run():
                yield

        app = create_base_app(
            routes=routes,
            middleware=list(middleware) if middleware else [],
            debug=self.settings.debug,
            lifespan=lifespan,
        )
        app.state.fastmcp_server = self
        # Keep the same state attribute used by FastMCP for logging.
        app.state.path = streamable_http_path
        return app
