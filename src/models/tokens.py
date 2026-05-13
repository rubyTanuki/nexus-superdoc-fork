from mistletoe.block_token import Heading
from mistletoe.span_token import RawText

class CustomHeading(Heading):
    """
    A heading token injected by the pipeline logic.
    """
    def __init__(self, text, level=2):
        # We simulate the structure mistletoe expects
        # Heading tokens expect a list of span tokens as children
        self.level = level
        self.children = [RawText(text)]
        self.is_custom = True  # Metadata for the renderer
        self.parent = None