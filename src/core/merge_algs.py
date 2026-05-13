from models.tokens import CustomAIHeading
from models.tree_nodes import EmbedTreeNode
from mistletoe.span_token import RawText
from sercies.openai_client import OpenAIProcessor

from mistletoe.block_token import BlockToken
from mistletoe.span_token import SpanToken, RawText


MIN_BLOCK_LEN = 20
SIMILARITY_THRESHOLD=0.97


class TreeEmbedder:
    def __init__(self, OpenAIProcessor):
        """
        openai_processor: instance of OpenAIProcessor from services/openai_client.py
        """
        self.ai = openai_processor

    def embed_tree(self, root):
        """
        Orchestrates the post-render semantic processing using the OpenAI wrapper.
        """
        # Calculate block lengths of all subtrees 
        self._calculate_block_len(root)
        # 1. Collect all headings for batching (Stage 3)
        # Using root.apply or root.walk to get a flat list
        headings = [node for node in root.apply(lambda x: x) if (node.type == 'heading' and node.block_len>=MIN_BLOCK_LEN)]

        
        # 2. Batch Embedding Request via our wrapper
        # We extract content from the underlying mistletoe SyntaxTreeNode
        texts = [h.node.raw_text for h in headings]
        embeddings = self.ai.embed_documents(texts) 
        
        # 3. Assign Embeddings and metadata
        for node, emb in zip(headings, embeddings):
            node.emb = emb
            node.has_embedding = True
            
        # 4. Bottom-up traversal for word counts (Post-order)
        self._calculate_block_len(root)



        heading_nodes = [n for n in node.apply(lambda x: x) if n.type == 'heading']
        
        if not heading_nodes:
            return
        heading_nodes_content = [h_node.content for h_node in heading_nodes]
        heading_node_vectors = node.emb_model.embed_documents(heading_nodes_content)

        for h_node,h_vector in zip(heading_nodes,heading_node_vectors):
            h_node.has_embedding = True 
            h_node.emb = h_vector

    def _calculate_block_len(self, node):
        """
        Post-order traversal: Calculates word count for current node 
        plus all its descendants.
        """
        current_text_len = 0
        
        # Extract text from the mistletoe node content
        if hasattr(node, 'raw_text') and node.raw_text:
            current_text_len = len(node.raw_text.split())
            
        # Sum children recursively
        subtree_len = sum(self._calculate_block_metrics(child) for child in node.children)
        
        node.block_len = current_text_len + subtree_len
        return node.block_len





class SemanticReconciler:
    def __init__(self, embedding_service, llm_service):
        self.embeddings = embedding_service
        self.llm = llm_service

    def reconcile(self, root, db_headings):
        """Stage 5: High-level orchestration of algorithms."""
        self.detect_and_fix_stragglers(root)
        self.prune_semantic_mismatches(root, db_headings)

    def detect_and_fix_stragglers(self, node):
        """Stage 5.1 & 5.2: Injects custom headings for orphan batches."""
        orphans = [c for c in node.children if c.level is None]
        
        if len(orphans) > 5: # Threshold for a 'batch'
            # 1. Get AI Heading
            txt_batch = " ".join([c.token.content for c in orphans[:3]])
            ai_title = self.llm.generate_title(txt_batch)
            
            # 2. Build Custom Node
            mock_token = CustomAIHeading(ai_title)
            new_section = EmbedTreeNode(mock_token, level=node.level + 1)
            new_section.is_custom_node = True
            
            # 3. Surgery: Move orphans under the new section
            for o in orphans:
                node.children.remove(o)
                new_section.add_child(o)
            node.add_child(new_section)

        # Recurse
        for child in node.children:
            self.detect_and_fix_stragglers(child)