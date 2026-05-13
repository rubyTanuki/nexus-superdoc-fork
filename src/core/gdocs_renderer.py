from mistletoe.base_renderer import BaseRenderer

class GoogleDocsRenderer(BaseRenderer):
    def __init__(self):
        super().__init__()
        self.cursor = 1
        self.batch_requests = []

    def render_from_etree(self, etree_root):
        """Custom entry point to render our Semantic Tree."""
        for node in etree_root.walk():
            # We call mistletoe's internal render method for the token
            self.render(node.token)
        return self.batch_requests

    def render_heading(self, token):
        # Calculate UTF-16 and generate JSON
        # self.batch_requests.append({...})
        # self.cursor += calculated_len
        pass