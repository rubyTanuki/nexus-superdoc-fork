import numpy as np

class EmbedTreeNode:
    def __init__(self, mistletoe_token, level=None):
        self.token = mistletoe_token
        self.type = mistletoe_token.__class__.__name__

        self.level = level
        self.children = []
        self.parent = None
        
        # Pipeline State
        self.embedding = None       # 1536-dim vector
        self.is_custom_node = False # True if injected by LLM
        self.is_pruned = False
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
                return token.raw_text

            # Only recurse into inline formatting (Strong, Emphasis, etc.)
            if isinstance(token, SpanToken) and hasattr(token, 'children'):
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


    def add_child(self, child):
        child.parent = self
        self.children.append(child)

    def apply(self,func: Callable[["EmbedTreeNode"],Any])->Generator[Any,None,None]:
        """A visitor-pattern implementation to apply a function across every node in the tree."""
        if self.node.type!="root":
            yield func(self)#yeild the result of the func on current node
        for child in self.children: 
            yield from child.apply(func)#recursively yield from da children
