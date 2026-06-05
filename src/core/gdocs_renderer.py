from mistletoe.block_token import BlockToken, List as MistletoeList, Table, Heading, Paragraph, ListItem
from mistletoe.span_token import SpanToken, RawText, Strong, Emphasis

from src.models.tree_nodes import EmbedTreeNode, GdocTreeNode, _utf16_len


 
 
# ---------------------------------------------------------------------------
# Heading level → Google Docs named style
# ---------------------------------------------------------------------------
_HEADING_STYLE = {
    1: "HEADING_1",
    2: "HEADING_2",
    3: "HEADING_3",
    4: "HEADING_4",
    5: "HEADING_5",
    6: "HEADING_6",
}
 
# Indentation per depth level in points
_INDENT_PT = 36
 
 
class GdocTreeBuilder:
    """
    Walks an EmbedTree (output of SemanticTreeBuilder + reconcile_structure)
    and produces a GdocTree where every node carries the Google Docs
    BatchUpdate requests needed to render its content.
 
    Usage:
        builder = GdocTreeBuilder(start_index=1)
        gdoc_root = builder.build(embed_root, matched_nodes)
        # gdoc_root.apply(lambda n: n.requests) → all requests in tree order
    """
 
    def __init__(self, start_index: int = 1):
        """
        start_index: the Google Doc insertion cursor (default 1 = doc start).
        The builder tracks a mutable cursor so every node knows exactly
        where to insert relative to what came before.
        """
        self._cursor = start_index
 
    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
 
    def build(
        self,
        embed_root: EmbedTreeNode,
        matched_nodes: dict[EmbedTreeNode, str],
    ) -> GdocTreeNode:
        """
        Recursively mirrors the EmbedTree into a GdocTree and populates
        each node's `requests` list in document order.
 
        embed_root    : the ROOT EmbedTreeNode from SemanticTreeBuilder
        matched_nodes : node_heading_pairs from reconcile_structure
                        { EmbedTreeNode → db_heading_label }
        """
        class _RootToken:
            pass
 
        root_embed = EmbedTreeNode(_RootToken(), level=0)
        root_embed.type = "ROOT"
 
        gdoc_root = GdocTreeNode(root_embed)
        gdoc_root.type = "ROOT"
 
        for embed_child in embed_root.children:
            self._visit(embed_child, gdoc_root, matched_nodes, depth=0)
 
        return gdoc_root
 
    # ------------------------------------------------------------------
    # Recursive visitor
    # ------------------------------------------------------------------
 
    def _visit(
        self,
        embed_node: EmbedTreeNode,
        gdoc_parent: GdocTreeNode,
        matched_nodes: dict,
        depth: int,
        list_mode: str | None = None,   # "BULLET" | "ORDERED" | None
    ) -> None:
        """
        Creates a GdocTreeNode for embed_node, generates its requests,
        attaches it to gdoc_parent, then recurses into children.
        """
        gdoc_node = GdocTreeNode(embed_node)
        gdoc_node.matched_heading = matched_nodes.get(embed_node)
        gdoc_parent.add_child(gdoc_node)
 
        node_type = embed_node.type  # e.g. "Heading", "PARA", "LIST", "TABLE", "QUOTE"
 
        if node_type == "Heading":
            self._gen_heading(gdoc_node, depth)
            # Recurse into children (sub-headings / paragraphs under this heading)
            for child in embed_node.children:
                self._visit(child, gdoc_node, matched_nodes, depth + 1, list_mode)
 
        elif node_type == "PARA":
            self._gen_paragraph(gdoc_node, depth, list_mode)
            # Paragraphs are leaves — no further recursion needed
 
        elif node_type == "LIST":
            # Determine bullet vs ordered from the underlying mistletoe token
            mode = self._list_mode(embed_node)
            # LIST itself emits no text; children (PARA leaves) do
            for child in embed_node.children:
                self._visit(child, gdoc_node, matched_nodes, depth, mode)
 
        elif node_type == "TABLE":
            self._gen_table(gdoc_node, depth)
            # Tables are treated as a single atomic block
 
        elif node_type == "QUOTE":
            self._gen_quote(gdoc_node, depth)
 
        else:
            # Unknown structural container — just recurse
            for child in embed_node.children:
                self._visit(child, gdoc_node, matched_nodes, depth, list_mode)
 
    # ------------------------------------------------------------------
    # Request generators  (each mutates self._cursor)
    # ------------------------------------------------------------------
 
    def _gen_heading(self, gdoc_node: GdocTreeNode, depth: int) -> None:
        """
        Inserts heading text and applies the matching HEADING_N paragraph style.
        depth 0 → HEADING_1, depth 1 → HEADING_2, etc. (capped at 6).
        """
        text = gdoc_node.content.strip() + "\n"
        text_len = _utf16_len(text)
        start = self._cursor
 
        style_name = _HEADING_STYLE.get(min(depth + 1, 6), "HEADING_6")
 
        gdoc_node.requests = [
            # 1. Insert the text
            {
                "insertText": {
                    "location": {"index": start},
                    "text": text,
                }
            },
            # 2. Apply heading paragraph style
            {
                "updateParagraphStyle": {
                    "range": {"startIndex": start, "endIndex": start + text_len},
                    "paragraphStyle": {"namedStyleType": style_name},
                    "fields": "namedStyleType",
                }
            },
        ]
 
        self._cursor += text_len
 
    def _gen_paragraph(
        self,
        gdoc_node: GdocTreeNode,
        depth: int,
        list_mode: str | None,
    ) -> None:
        """
        Inserts paragraph / list-item text with correct indentation.
        If list_mode is set, also attaches a bullet/numbering preset.
        """
        raw = _extract_text(gdoc_node.node.node).strip()
        if not raw:
            return
 
        # List items get a single newline; body text gets double for spacing
        text = raw + "\n" if list_mode else raw + "\n\n"
        text_len = _utf16_len(text)
        start = self._cursor
 
        bullet_indent = depth * _INDENT_PT
        text_indent   = (depth + 1) * _INDENT_PT
 
        # For plain paragraphs both indents align
        if not list_mode:
            bullet_indent = text_indent
 
        requests = [
            {
                "insertText": {
                    "location": {"index": start},
                    "text": text,
                }
            },
        ]
 
        if list_mode:
            preset = (
                "BULLET_DISC_CIRCLE_SQUARE"
                if list_mode == "BULLET"
                else "NUMBERED_DECIMAL_ALPHA_ROMAN"
            )
            requests.append({
                "createParagraphBullets": {
                    "range": {"startIndex": start, "endIndex": start + text_len},
                    "bulletPreset": preset,
                }
            })
 
        requests.append({
            "updateParagraphStyle": {
                "range": {"startIndex": start, "endIndex": start + text_len},
                "paragraphStyle": {
                    "namedStyleType": "NORMAL_TEXT",
                    "indentFirstLine": {"magnitude": bullet_indent, "unit": "PT"},
                    "indentStart":     {"magnitude": text_indent,   "unit": "PT"},
                },
                "fields": "namedStyleType,indentFirstLine,indentStart",
            }
        })
 
        gdoc_node.requests = requests
        self._cursor += text_len
    
    def _gen_table(self, gdoc_node: GdocTreeNode, depth: int) -> None:
        table_data = _extract_table_data(gdoc_node.node.node)
        if not table_data:
            return

        num_rows = len(table_data)
        num_cols = max(len(row) for row in table_data)
        start = self._cursor

        # Phase 1: Structure creation request goes inline with main text layout
        create_request = [
            {
                "insertTable": {
                    "rows": num_rows,
                    "columns": num_cols,
                    "location": {"index": start},
                }
            }
        ]

        fill_requests = []
        # Target cells relative to a clean, empty table structure layout
        cell_cursor = start + 2

        for r_idx, row in enumerate(table_data):
            padded_row = row + [""] * (num_cols - len(row))
            for cell_text in padded_row:
                if cell_text:
                    cell_len = _utf16_len(cell_text)
                    fill_requests.append({
                        "insertText": {
                            "location": {"index": cell_cursor},
                            "text": cell_text,
                        }
                    })
                    if r_idx == 0:
                        fill_requests.append({
                            "updateTextStyle": {
                                "range": {
                                    "startIndex": cell_cursor,
                                    "endIndex": cell_cursor + cell_len,
                                },
                                "textStyle": {"bold": True},
                                "fields": "bold",
                            }
                        })
                cell_cursor += 1  # Step past empty cell token slot
            cell_cursor += 1  # Step past row end token slot

        # The precise index footprint of an empty table structure
        empty_table_offset = 1 + (num_rows * num_cols) + num_rows + 1

        # Keep creation inline with standard text flows; defer fill payloads
        gdoc_node.requests = create_request
        gdoc_node._table_fill = fill_requests

        self._cursor += empty_table_offset
 
    def _gen_quote(self, gdoc_node: GdocTreeNode, depth: int) -> None:
        """
        Renders a block quote as indented NORMAL_TEXT with a left margin.
        """
        raw = gdoc_node.content.strip()
        if not raw:
            return
 
        text = raw + "\n\n"
        text_len = _utf16_len(text)
        start = self._cursor
        indent = (depth + 1) * _INDENT_PT
 
        gdoc_node.requests = [
            {
                "insertText": {
                    "location": {"index": start},
                    "text": text,
                }
            },
            {
                "updateParagraphStyle": {
                    "range": {"startIndex": start, "endIndex": start + text_len},
                    "paragraphStyle": {
                        "namedStyleType": "NORMAL_TEXT",
                        "indentFirstLine": {"magnitude": indent, "unit": "PT"},
                        "indentStart":     {"magnitude": indent, "unit": "PT"},
                    },
                    "fields": "namedStyleType,indentFirstLine,indentStart",
                }
            },
        ]
 
        self._cursor += text_len
 
    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
 
    @staticmethod
    def _list_mode(embed_node: EmbedTreeNode) -> str:
        """Checks the underlying mistletoe token to decide BULLET vs ORDERED."""
        token = embed_node.node
        # mistletoe sets token.start on ordered lists (start=1)
        if hasattr(token, 'start') and token.start is not None:
            return "ORDERED"
        return "BULLET"
 
    def collect_all_requests(self, gdoc_root: GdocTreeNode) -> tuple[list[dict], list[dict]]:
        """Returns (main_structural_requests, table_fill_requests)"""
        main_requests = []
        table_fills = []
    
        for node in gdoc_root.apply(lambda n: n):
            if hasattr(node, 'requests') and node.requests:
                main_requests.extend(node.requests)
            if hasattr(node, '_table_fill') and node._table_fill:
                table_fills.extend(node._table_fill)
    
        return main_requests, table_fills
 
 
# ---------------------------------------------------------------------------
# Pure text extraction helpers (no side effects)
# ---------------------------------------------------------------------------
 
def _extract_text(token) -> str:
    """
    Recursively extracts plain text from any mistletoe token.
    Works for Heading, Paragraph, ListItem, and inline spans.
    """
    if isinstance(token, RawText):
        return token.content
    if hasattr(token, 'children') and token.children:
        return " ".join(_extract_text(c) for c in token.children if c)
    return getattr(token, 'content', "") or ""
 
 
def _extract_table_data(token) -> list[list[str]]:
    if not hasattr(token, 'children'):
        return []

    rows = []

    # mistletoe stores the header row separately on token.header
    header = getattr(token, 'header', None)
    if header and hasattr(header, 'children'):
        cells = []
        for cell in header.children:
            text = _extract_text(cell).strip()
            if not all(c in "- " for c in text):
                cells.append(text)
        if cells:
            rows.append(cells)

    # Body rows are in token.children
    for row_token in token.children:
        if not hasattr(row_token, 'children'):
            continue
        cells = []
        for cell in row_token.children:
            text = _extract_text(cell).strip()
            if all(c in "- " for c in text):
                break
            cells.append(text)
        if cells:
            rows.append(cells)

    return rows
 
