import numpy as np
from typing import List, Callable, Any, Generator, Iterator, Self ,TypeVar, Optional

from mistletoe.block_token import BlockToken
from mistletoe.span_token import SpanToken, RawText


class EmbedTreeNode:

    __slots__ = ['node', 'type', 'parent', 'children', 'block_len', 'embedding','mean_emb','level', 'is_custom_node', 'is_pruned', "has_embedding"]

    def __init__(self, mistletoe_token, level=None):
        self.node = mistletoe_token
        self.type = mistletoe_token.__class__.__name__

        self.level = level
        self.children = []
        self.parent = None
        
        # Pipeline State
        self.embedding = None       # 1536-dim vector
        self.mean_emb = self.embedding
        self.is_custom_node = False # True if injected by LLM
        self.is_pruned = False
        self.has_embedding  = False
        self.block_len = 0          # Word count for subtree

    @property
    def raw_text(self) -> str:
        """
        Extracts text ONLY from the current element's inline/span children.
        If it's a BlockToken (like a List), it returns empty to prevent leakage.
        """
        if not self.node:
            return ""

        # This helper only collects text from SpanTokens (inlines)
        def _extract_inline(token) -> str:
            if isinstance(token, RawText):
                return token.content

            # Only recurse into inline formatting (Strong, Emphasis, etc.)
            if isinstance(token, SpanToken) and hasattr(token, 'children') and token.children is not None:
                return "".join(_extract_inline(c) for c in token.children)

            return ""

        # LOGIC PARTITION:
        # We only want to extract raw_text from nodes that hold text (Heading/Paragraph).
        # If this is a structural container (like a List or Document), 
        # we return "" because its "raw_text" is actually in its block-children.

        # 1. If it's a leaf span token itself
        if isinstance(self.node, SpanToken):
            return _extract_inline(self.node)

        # 2. If it's a block token, we ONLY gather its span-level children.
        # Note: Headings/Paragraphs contain Spans. Lists contain BlockTokens.
        if isinstance(self.node, BlockToken) and hasattr(self.node, 'children'):
            return "".join(
                _extract_inline(c) for c in self.node.children 
                if isinstance(c, SpanToken)
            ).strip()

        return ""

    @property
    def content(self) -> str:
        if not self.node: return ""
        
        # Internal helper to grab text from ANY depth within this specific node
        def _recursive_text(token) -> str:
            if isinstance(token, RawText):
                return token.content
            if hasattr(token, 'children') and token.children:
                # Join with space to prevent words from merging between cells/items
                return " ".join(_recursive_text(c) for c in token.children)
            return getattr(token, 'content', "")

        # Logic Partition:
        # If it's a Heading or Paragraph, we usually just want the span text.
        # If it's a Table or List, we need to dig into Rows/Cells/Items.
        return _recursive_text(self.node).strip()


    def add_child(self, child_node):
        child_node.parent = self
        self.children.append(child_node)

    def apply(self,func: Callable[["EmbedTreeNode"],Any])->Generator[Any,None,None]:
        """A visitor-pattern implementation to apply a function across every node in the tree."""
        if self.type!="root":
            yield func(self)#yeild the result of the func on current node
        for child in self.children: 
            yield from child.apply(func)#recursively yield from da children




 
def _utf16_len(text: str) -> int:
    """Google Docs indices are UTF-16 code units, not bytes or chars."""
    return len(text.encode("utf-16-le")) // 2
 
 
class GdocTreeNode:
    __slots__ = ['node', 'type', 'level', 'parent', 'children', 'content', 'matched_heading', 'requests','_table_create','_table_fill', '_table_data','_is_table','_table_start','_num_rows','_num_cols']
 
    def __init__(self, embed_node: EmbedTreeNode):
        self.node = embed_node
        self.type = embed_node.type
        self.level = embed_node.level or 0
        self.parent = None
        self.children: list['GdocTreeNode'] = []
        self.content: str = embed_node.content or ""
        self.matched_heading: str | None = None  # None = new/unmatched, str = DB heading label
        self.requests: list[dict] = []           # Google Docs BatchUpdate requests for this node
 
    def add_child(self, child_node: 'GdocTreeNode'):
        child_node.parent = self
        self.children.append(child_node)
 
    def apply(self, func: Callable[['GdocTreeNode'], Any]) -> Generator[Any, None, None]:
        if self.type != 'ROOT':
            yield func(self)
        for child in self.children:
            yield from child.apply(func)
 
    def __str__(self) -> str:
        return self._format_tree(level=0)
 
    def _format_tree(self, level: int) -> str:
        indent = "  " * level
        matched = f"[MATCHED: {self.matched_heading}]" if self.matched_heading else "[UNMATCHED]"
        req_count = f"[{len(self.requests)} reqs]"
        header = f"{indent}{matched} {req_count} {self.type.upper()} [L:{self.level}]"
        snippet = ""
        if self.content:
            preview = self.content.strip().replace("\n", " ")
            preview = (preview[:60] + "..") if len(preview) > 60 else preview
            snippet = f"\n{indent}  | \"{preview}\""
        children_str = "".join(child._format_tree(level + 1) for child in self.children)
        return f"\n{header}{snippet}{children_str}"
 
