from __future__ import annotations

import importlib.util
import sys
from importlib.machinery import ModuleSpec
from types import ModuleType
from typing import Any, Optional, Sequence


def _module_exists(name: str) -> bool:
    try:
        spec = importlib.util.find_spec(name)
    except ModuleNotFoundError:
        return False
    return spec is not None


def ensure_ragatouille_dependencies() -> None:
    """
    Ensure ragatouille's optional langchain imports resolve even on langchain>=0.2.

    ragatouille<=0.0.9 still imports ``langchain.retrievers.document_compressors``,
    which was removed from modern langchain releases. We inject a lightweight
    compatibility stub that satisfies the import without forcing users to pin
    langchain to an older version.
    """

    target_module = "langchain.retrievers.document_compressors.base"
    if _module_exists(target_module):
        return

    import langchain  # noqa: WPS433  # Imported lazily for side effects
    from langchain_core.callbacks.manager import Callbacks
    from langchain_core.documents import Document
    try:
        from langchain_core.pydantic_v1 import BaseModel
    except ModuleNotFoundError:  # pragma: no cover - fallback for newer langchain
        from pydantic import BaseModel  # type: ignore[assignment]

    retrievers_module_name = "langchain.retrievers"
    doc_compressors_name = f"{retrievers_module_name}.document_compressors"

    retrievers_module = sys.modules.get(retrievers_module_name)
    if retrievers_module is None:
        retrievers_module = ModuleType(retrievers_module_name)
        retrievers_module.__spec__ = ModuleSpec(
            name=retrievers_module_name,
            loader=None,
        )
        retrievers_module.__path__ = []  # type: ignore[attr-defined]
        sys.modules[retrievers_module_name] = retrievers_module
    setattr(langchain, "retrievers", retrievers_module)

    doc_compressors_module = sys.modules.get(doc_compressors_name)
    if doc_compressors_module is None:
        doc_compressors_module = ModuleType(doc_compressors_name)
        doc_compressors_module.__spec__ = ModuleSpec(
            name=doc_compressors_name,
            loader=None,
        )
        doc_compressors_module.__path__ = []  # type: ignore[attr-defined]
        sys.modules[doc_compressors_name] = doc_compressors_module

    base_module = ModuleType(target_module)
    base_module.__spec__ = ModuleSpec(
        name=target_module,
        loader=None,
    )

    class BaseDocumentCompressor(BaseModel):
        """Minimal shim to satisfy ragatouille's inheritance checks."""

        class Config:
            arbitrary_types_allowed = True

        def compress_documents(  # noqa: D401
            self,
            documents: Sequence[Document],
            query: str,
            callbacks: Optional[Callbacks] = None,
            **kwargs: Any,
        ) -> Any:
            raise NotImplementedError

        async def acompress_documents(  # noqa: D401
            self,
            documents: Sequence[Document],
            query: str,
            callbacks: Optional[Callbacks] = None,
            **kwargs: Any,
        ) -> Any:
            raise NotImplementedError

    base_module.BaseDocumentCompressor = BaseDocumentCompressor
    sys.modules[target_module] = base_module
