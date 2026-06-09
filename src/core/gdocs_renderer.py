import io
import matplotlib.pyplot as plt
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
    """
 
    def __init__(self, start_index: int = 1):
        """
        start_index: the Google Doc insertion cursor (default 1 = doc start).
        The builder tracks a mutable cursor so every node knows exactly
        where to insert relative to what came before.
        """
        self._cursor = start_index
        self.api = None
        self._temp_drive_files = []

    def set_google_api(self, api_wrapper):
        """Pass your initialized GoogleDocsAPI instance into the renderer."""
        self.api = api_wrapper
 
    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
 
    def build(
        self,
        embed_root: EmbedTreeNode,
        matched_nodes: dict[EmbedTreeNode, str],
    ) -> GdocTreeNode:
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
        gdoc_node = GdocTreeNode(embed_node)
        gdoc_node.matched_heading = matched_nodes.get(embed_node)
        gdoc_parent.add_child(gdoc_node)
 
        node_type = embed_node.type  # e.g. "Heading", "PARA", "LIST", "TABLE", "QUOTE"
 
        if node_type == "Heading":
            self._gen_heading(gdoc_node, depth)
            for child in embed_node.children:
                self._visit(child, gdoc_node, matched_nodes, depth + 1, list_mode)
 
        elif node_type == "PARA":
            self._gen_paragraph(gdoc_node, depth, list_mode)
 
        elif node_type == "LIST":
            mode = self._list_mode(embed_node)
            for child in embed_node.children:
                self._visit(child, gdoc_node, matched_nodes, depth, mode)

        elif node_type == "LIST_ITEM":  # add this branch
            for child in embed_node.children:
                self._visit(child, gdoc_node, matched_nodes, depth, list_mode)  # pass through
        
        elif node_type == "TABLE":
            self._gen_table(gdoc_node, depth)
 
        elif node_type == "QUOTE":
            self._gen_quote(gdoc_node, depth)
 
        else:
            for child in embed_node.children:
                self._visit(child, gdoc_node, matched_nodes, depth, list_mode)
 
    # ------------------------------------------------------------------
    # Request generators (each mutates self._cursor)
    # ------------------------------------------------------------------
 
    def _gen_heading(self, gdoc_node: GdocTreeNode, depth: int) -> None:
        text = gdoc_node.content.strip() + "\n"
        text_len = _utf16_len(text)
        start = self._cursor
 
        style_name = _HEADING_STYLE.get(min(depth + 1, 6), "HEADING_6")
 
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
        Inserts block text and images sequentially, updating styles and 
        paragraph metrics without breaking the character flow layout.
        """
        token_source = gdoc_node.node.node
        runs = self._extract_styled_runs(token_source)
        if not runs:
            return
 
        suffix = "\n" if list_mode else "\n\n"
        start_index = self._cursor
        requests = []
 
        # Loop over every style span run inside the paragraph
        for text_segment, style in runs:
            if not text_segment:
                continue
 
            if style.get("is_math"):
                formula = text_segment.strip("$")
                math_req = self._insert_inline_math_image(formula, self._cursor)
                requests.append(math_req)
                self._cursor += 1  # Inline assets count as exactly 1 unit space
            else:
                seg_len = _utf16_len(text_segment)
                requests.append({
                    "insertText": {
                        "location": {"index": self._cursor},
                        "text": text_segment,
                    }
                })
 
                # Separate out text styling parameters from custom engine tracking flags
                text_styles = {k: v for k, v in style.items() if k != "is_math"}
                if text_styles:
                    requests.append({
                        "updateTextStyle": {
                            "textStyle": text_styles,
                            "fields": ",".join(text_styles.keys()),
                            "range": {
                                "startIndex": self._cursor,
                                "endIndex": self._cursor + seg_len
                            }
                        }
                    })
                self._cursor += seg_len
 
        # Insert structural spacing suffix
        suffix_len = _utf16_len(suffix)
        requests.append({
            "insertText": {
                "location": {"index": self._cursor},
                "text": suffix,
            }
        })
        self._cursor += suffix_len
        total_block_len = self._cursor - start_index
 
        # Structural layout indents matching depth tree locations
        bullet_indent = depth * _INDENT_PT
        text_indent   = (depth + 1) * _INDENT_PT
        if not list_mode:
            bullet_indent = text_indent
 
        if list_mode:
            preset = (
                "BULLET_DISC_CIRCLE_SQUARE"
                if list_mode == "BULLET"
                else "NUMBERED_DECIMAL_ALPHA_ROMAN"
            )
            requests.append({
                "createParagraphBullets": {
                    "range": {"startIndex": start_index, "endIndex": start_index + total_block_len},
                    "bulletPreset": preset,
                }
            })
 
        requests.append({
            "updateParagraphStyle": {
                "range": {"startIndex": start_index, "endIndex": start_index + total_block_len},
                "paragraphStyle": {
                    "namedStyleType": "NORMAL_TEXT",
                    "indentFirstLine": {"magnitude": bullet_indent, "unit": "PT"},
                    "indentStart":     {"magnitude": text_indent,   "unit": "PT"},
                },
                "fields": "namedStyleType,indentFirstLine,indentStart",
            }
        })
 
        gdoc_node.requests = requests
 
    def _gen_table(self, gdoc_node: GdocTreeNode, depth: int) -> None:
        table_data = _extract_table_data(gdoc_node.node.node)
        if not table_data:
            return

        num_rows = len(table_data)
        num_cols = max(len(row) for row in table_data)
        start = self._cursor

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
                cell_cursor += 1  
            cell_cursor += 1  

        empty_table_offset = 1 + (num_rows * num_cols) + num_rows + 1
        gdoc_node.requests = create_request
        gdoc_node._table_fill = fill_requests
        self._cursor += empty_table_offset
 
    def _gen_quote(self, gdoc_node: GdocTreeNode, depth: int) -> None:
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
    # Math Rendering Engine Additions
    # ------------------------------------------------------------------
 
    def _render_latex_to_bytes(self, formula: str) -> bytes:
        """Compiles standard LaTeX syntax fields to high-fidelity transparent images."""
        fig = plt.figure(figsize=(0.1, 0.1), dpi=300)
        fig.text(0, 0, f"${formula}$", fontsize=11, usetex=False)
        
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0.02, transparent=True)
        plt.close(fig)
        return buf.getvalue()

    def _insert_inline_math_image(self, formula_text: str, index: int) -> dict:
        """Side-loads image data blocks directly into the execution drive path."""
        if not self.api:
            raise RuntimeError("GoogleDocsAPI instance missing. Use builder.set_google_api(api) before execution.")
            
        img_bytes = self._render_latex_to_bytes(formula_text)
        uploaded_file = self.api.upload_image_bytes(img_bytes, filename=f"latex_{index}.png")
        
        image_url = uploaded_file.get("webContentLink")
        file_id = uploaded_file.get("id")
        self._temp_drive_files.append(file_id)

        return {
            "insertInlineImage": {
                "uri": image_url,
                "location": {"index": index},
                "objectSize": {
                    "height": {"magnitude": 13, "unit": "PT"}
                }
            }
        }

    def commit_and_cleanup_math(self, doc_id: str):
        """Executes the batch request, then purges the temporary LaTeX files from Drive."""
        if not self._temp_drive_files:
            return
            
        print(f"Purging {len(self._temp_drive_files)} temporary equation assets from Drive...")
        for file_id in self._temp_drive_files:
            try:
                self.api.drive_service.files().delete(fileId=file_id).execute()
            except Exception as e:
                print(f"Could not purge file {file_id}: {e}")
        self._temp_drive_files.clear()

    # ------------------------------------------------------------------
    # Inline Runs & Token Traversal Engine
    # ------------------------------------------------------------------
 
    def _extract_styled_runs(self, token) -> list[tuple[str, dict]]:
        """
        Flattens complex nested formatting span trees into explicit sequence chunks.
        Returns: list of (text_segment, style_dict)
        """
        runs = []
        
        def _walk(node, current_style: dict):
            # Evaluate style maps matching class declarations
            node_class = node.__class__.__name__
            new_style = current_style.copy()
            
            match node_class:
                case "Strong":        new_style["bold"] = True
                case "Emphasis":      new_style["italic"] = True
                case "Strikethrough": new_style["strikethrough"] = True
                case "InlineMath":    new_style["is_math"] = True

            if node_class == "RawText" or node_class == "InlineMath":
                # Leaf fragments emit actual rendering strings
                content = getattr(node, "content", "")
                if content:
                    runs.append((content, new_style))
            elif hasattr(node, "children") and node.children:
                for child in node.children:
                    if child:
                        _walk(child, new_style)
            else:
                content = getattr(node, "content", "")
                if content:
                    runs.append((content, new_style))

        if hasattr(token, "children") and token.children:
            for c in token.children:
                if c:
                    _walk(c, {})
        else:
            _walk(token, {})
            
        return runs

    @staticmethod
    def _list_mode(embed_node: EmbedTreeNode) -> str:
        token = embed_node.node        # EmbedTreeNode
        mistletoe_token = getattr(token, 'node', token)  # unwrap to mistletoe token
        if hasattr(mistletoe_token, 'start') and mistletoe_token.start is not None:
            return "ORDERED"
        return "BULLET"
    
    def collect_all_requests(self, gdoc_root: GdocTreeNode) -> tuple[list[dict], list[dict]]:
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
    if isinstance(token, RawText):
        return token.content
    if hasattr(token, 'children') and token.children:
        return " ".join(_extract_text(c) for c in token.children if c)
    return getattr(token, 'content', "") or ""
 
 
def _extract_table_data(token) -> list[list[str]]:
    if not hasattr(token, 'children'):
        return []

    rows = []
    header = getattr(token, 'header', None)
    if header and hasattr(header, 'children'):
        cells = []
        for cell in header.children:
            text = _extract_text(cell).strip()
            if not all(c in "- " for c in text):
                cells.append(text)
        if cells:
            rows.append(cells)

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