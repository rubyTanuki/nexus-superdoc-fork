from mistletoe.base_renderer import BaseRenderer
from models.tree_nodes import EmbedTreeNode

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
    
