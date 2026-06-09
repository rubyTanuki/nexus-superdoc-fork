from mistletoe.base_renderer import BaseRenderer
from mistletoe.block_token import Heading

from src.models.tree_nodes import EmbedTreeNode


class SemanticTreeBuilder(BaseRenderer):
    def __init__(self):
        super().__init__()
        # 1. Create a dummy object to satisfy EmbedTreeNode's need for a class name
        class RootToken: pass
        
        # 2. Initialize with the dummy token
        self.root = EmbedTreeNode(RootToken(), level=0)
        self.root.type = "ROOT" # Manually override since RootToken.__name__ is 'RootToken'
        
        self.stack = [(0, self.root)]

    def _handle_block(self, token, type_override=None):
        """Helper to wrap block tokens using the expected Init signature."""
        # Check if it has content/children before adding
        content = EmbedTreeNode(token).content
        has_children = bool(getattr(token, 'children', None))
        if not content and not has_children:
            return ""

        new_node = EmbedTreeNode(token, level=self.stack[-1][0])
        if type_override:
            new_node.type = type_override
        self.stack[-1][1].add_child(new_node)
        return ""


    def render_heading(self, token):
        # EmbedTreeNode extracts type from token.__class__.__name__ automatically
        new_node = EmbedTreeNode(token, level=token.level)
        
        # Monotonic stack logic
        while len(self.stack) > 1 and self.stack[-1][0] >= token.level:
            self.stack.pop()
            
        self.stack[-1][1].add_child(new_node)
        self.stack.append((token.level, new_node))
        return ""

    def render_list(self, token):
        new_node = EmbedTreeNode(token, level=self.stack[-1][0])
        new_node.type = "LIST"
        # Carry the mistletoe start attribute through
        new_node.node.start = token.start  
        self.stack[-1][1].add_child(new_node)
        
        # Push list onto stack so list items attach under it
        self.stack.append((self.stack[-1][0], new_node))
        self.render_inner(token)
        self.stack.pop()
        return ""

    def render_list_item(self, token):
        new_node = EmbedTreeNode(token, level=self.stack[-1][0])
        new_node.type = "LIST_ITEM"
        self.stack[-1][1].add_child(new_node)
        
        # Push so nested paragraphs attach under this item
        self.stack.append((self.stack[-1][0], new_node))
        self.render_inner(token)
        self.stack.pop()
        return ""

    def render_table(self, token):
        return self._handle_block(token, "TABLE")

    def render_paragraph(self, token):
        # Use EmbedTreeNode to check content for metadata filtering
        label = EmbedTreeNode(token).content
        if label.lower() in ["lists", "table", "quote"]:
            return ""
        return self._handle_block(token, "PARA")

    def render_quote_block(self, token):
        return self._handle_block(token, "QUOTE")

    def render_document(self, token):
        self.render_inner(token)
        return self.root

    def render_inline_math(self,token): 
        # Pass it through cleanly so it's preserved in the token children array
        return ""