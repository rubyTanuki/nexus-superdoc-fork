import pymupdf
import pymupdf4llm
import mistletoe
from io import BytesIO
from mistletoe.block_token import BlockToken, Heading, Paragraph
from mistletoe.span_token import SpanToken, RawText
# --- 1. THE PROPERTY LOGIC (Fixed & Partitioned) ---
class SyntaxTreeNode:
    """Mock of your node wrapper to demonstrate the property."""
    def __init__(self, token):
        self.node = token
        self.type = token.__class__.__name__

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


# --- 2. THE MONOTONIC STACK RENDERER ---
class SemanticTreeBuilder(mistletoe.base_renderer.BaseRenderer):
    def __init__(self):
        super().__init__()
        self.root = {"type": "ROOT", "children": [], "level": 0}
        self.stack = [(0, self.root)]

    def _handle_block(self, token, type_name=None):
        """Helper to wrap block tokens, consumes children to prevent duplicates."""
        t_type = type_name or token.__class__.__name__.upper()
        
        # Use our property to get text
        label = SyntaxTreeNode(token).content
        
        # If the block is empty and has no children, skip it
        if not label and not hasattr(token, 'children'):
            return ""

        new_node = {
            "type": t_type, 
            "label": (label[:75] + "...") if len(label) > 75 else label,
            "children": []
        }
        
        self.stack[-1][1]["children"].append(new_node)
        
        # IMPORTANT: We do NOT call self.render_inner(token) here 
        # because we've already "consumed" the text via SyntaxTreeNode(token).content.
        # This prevents the children from being rendered again as siblings.
        return ""

    def render_heading(self, token):
        label = SyntaxTreeNode(token).content
        new_node = {"type": f"H{token.level}", "label": label, "children": []}
        
        # Monotonic stack logic: maintains the H1 > H2 > H3 hierarchy
        while len(self.stack) > 1 and self.stack[-1][0] >= token.level:
            self.stack.pop()
            
        self.stack[-1][1]["children"].append(new_node)
        self.stack.append((token.level, new_node))
        return ""

    def render_list(self, token):
        # We handle the list as one semantic unit
        return self._handle_block(token, "LIST")

    def render_table(self, token):
        # We handle the table as one semantic unit
        return self._handle_block(token, "TABLE")

    def render_paragraph(self, token):
        # Only render paragraph if it's not just a duplicate title
        label = SyntaxTreeNode(token).content
        if label.lower() in ["lists", "table", "quote"]:
            return "" # Skip "labels" that are actually just metadata
            
        return self._handle_block(token, "PARA")

    def render_quote_block(self, token):
        self._handle_block(token, "QUOTE")
        return ""

    def render_code_block(self, token):
        self._handle_block(token, "CODE")
        return ""

    def render_thematic_break(self, token):
        self._handle_block(token, "HR")
        return ""

    def render_document(self, token):
        self.render_inner(token)
        return self.root

# --- 3. DEBUG PRINTERS ---
def print_raw_ast(token, indent=0):
    """Shows mistletoe's native flat structure."""
    t_name = token.__class__.__name__
    t_info = f" (L{token.level})" if isinstance(token, Heading) else ""
    print("  " * indent + f"[{t_name}]{t_info}")
    if hasattr(token, 'children') and token.children:
        for c in token.children:
            if isinstance(c, (BlockToken, SpanToken)):
                print_raw_ast(c, indent + 1)

def print_semantic_tree(node, indent=0):
    """Shows the nested structure with formatted labels for Tables/Lists."""
    pref = "  " * indent
    
    # Get the label and clean it up
    raw_label = node.get('label', "")
    
    # If it's a table, let's at least show some structure
    if node['type'] == 'TABLE':
        # Simple cleanup: replace multiple spaces with one for a 'row' feel
        label = " | " + " ".join(raw_label.split())
    else:
        # Standard truncation for paragraphs/headings
        label = f": {raw_label[:80]}..." if len(raw_label) > 80 else f": {raw_label}"

    print(f"{pref}└── [{node['type']}]{label}")
    
    # Recurse
    for child in node.get('children', []):
        print_semantic_tree(child, indent + 1)



# --- 4. EXECUTION ---
def test_pdf_structure(file_path):
    print(f"--- Processing: {file_path} ---")
    
    # PDF to Markdown
    doc = pymupdf.open(file_path)
    md_text = pymupdf4llm.to_markdown(doc,force_markdown=True)
    print('----Markdown----')
    print(md_text)
    print('----End-Of-MD---')
    # Parse with Mistletoe
    mistletoe_doc = mistletoe.Document(md_text)
    
    print("\n[TEST 1] RAW MISTLETOE AST (The Siblings)")
    print_raw_ast(mistletoe_doc)
    
    print("\n" + "="*50)
    
    print("\n[TEST 2] NESTED SEMANTIC TREE (The Stack Output)")
    with SemanticTreeBuilder() as builder:
        nested_tree = builder.render(mistletoe_doc)
        print_semantic_tree(nested_tree)



if __name__ == "__main__":
    # Change this to your actual file path
    test_pdf_structure("files/basic-text.pdf")