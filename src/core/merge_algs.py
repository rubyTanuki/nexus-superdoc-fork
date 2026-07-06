from models.tokens import CustomHeading
from models.tree_nodes import EmbedTreeNode
from mistletoe.span_token import RawText
from services.openai_client import OpenAIProcessor
from services.onnx_client import EMBED_DIM
from services.pinecone_client import DB_Heading

from mistletoe.block_token import BlockToken
from mistletoe.span_token import SpanToken, RawText



import numpy as np
from markdown_it import MarkdownIt
from typing import List, Generator, Optional, Dict, Tuple
from pydantic import BaseModel
MIN_BLOCK_LEN = 5
SIMILARITY_THRESHOLD=0.97


def _is_heading(node) -> bool:
    t = node.type.lower()
    return t == "heading" or t.startswith("h")


class TreeEmbedder:
    def __init__(self, openai_processor):
        """
        openai_processor: instance of OpenAIProcessor from services/openai_client.py
        """
        self.ai = openai_processor

    def embed_tree(self, root, batch_size: int = 64):
        # 1. Update subtree word counts (drives the MIN_BLOCK_LEN gate below).
        self._calculate_block_len(root)

        # 2. Collect every node that should carry its own content embedding:
        #      - headings (section anchors), and
        #      - content leaves (PARA / TABLE / QUOTE / ...): nodes with no children.
        targets = []  # list[tuple[EmbedTreeNode, str]]
        for node in root.apply(lambda x: x):
            if node.type.lower() == "root" or node.block_len < MIN_BLOCK_LEN:
                continue
            is_leaf = not node.children
            if not (_is_heading(node) or is_leaf):
                continue
            text = node.content
            if text:
                targets.append((node, text))

        if not targets:
            print("DEBUG: No nodes met the embedding criteria.")
            return

        # 3. Sort by character length so each fixed-size batch holds similar-length
        #    texts. The model pads every text in a batch up to the longest one, so
        #    length-homogeneous batches minimize total padding -> less wasted compute.
        targets.sort(key=lambda pair: len(pair[1]))

        # 4. Embed in length-bucketed batches; assign each vector back to its node.
        embedded = 0
        for i in range(0, len(targets), batch_size):
            chunk = targets[i:i + batch_size]
            vectors = self.ai.embed_documents([t for _, t in chunk])
            for (node, _), vec in zip(chunk, vectors):
                node.embedding = vec
                node.has_embedding = True
                embedded += 1

        n_headings = sum(1 for n, _ in targets if _is_heading(n))
        print(f"DEBUG: Embedded {embedded} nodes "
              f"({n_headings} headings, {embedded - n_headings} leaves).")

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
        subtree_len = sum(self._calculate_block_len(child) for child in node.children)
        
        node.block_len = current_text_len + subtree_len
        return node.block_len

    



class SemanticReconciler:
    def __init__(self, embedding_service, llm_service, similarity_threshold=0.8, min_block_len=20):
        self.embeddings = embedding_service
        self.llm = llm_service
        self.SIMILARITY_THRESHOLD = similarity_threshold
        self.MIN_BLOCK_LEN = min_block_len
        self.mdit = MarkdownIt()

    #
    # Matching logic
    #

    def match_headings(self, root, db_headings: List[DB_Heading]) -> Dict['EmbedTreeNode', str]:
        """
        Uses cosine similarity to find the best match between current tree branches 
        and existing headings in the database.
        """
        # Filter for valid embeddings
        heading_vecs = np.array([h.embedding for h in db_headings if len(h.embedding) == EMBED_DIM])
        
        def check_node_against_headings(node):
            if heading_vecs.size == 0: 
                return None
            # Standardize type check
            t_type = node.type.lower()
            if t_type == "root":
                return None
            if node.block_len < self.MIN_BLOCK_LEN:
                return None
            # Only section anchors are matched against DB headings; leaf paragraph
            # embeddings exist for the merkle roll-up, not for direct heading matching.
            if not (_is_heading(node) or getattr(node, "is_custom_node", False)):
                return None
            # Match on the section centroid (merkle roll-up); fall back to the node's
            # own vector if the roll-up hasn't populated mean_emb for some reason.
            node_vec = node.mean_emb if node.mean_emb is not None else node.embedding
            if node_vec is None:
                return None

            most_similar_idx, similarity = find_closest_cosine_sim(node_vec, heading_vecs)
            
            if similarity < self.SIMILARITY_THRESHOLD:
                return None   
            return (db_headings[most_similar_idx].heading, node)

        print(f"Fetched Headings len: {len(db_headings)}")
        
        # Use the apply method on the root to traverse
        node_heading_pairs = {
            node: heading 
            for result in root.apply(check_node_against_headings)
            if result is not None 
            for heading, node in [result]
        }
        return node_heading_pairs
    
    #
    # Straggler Detection 
    #
    
    @staticmethod
    def find_straggler_branches(node, min_block_len: int) -> Generator[List['EmbedTreeNode'], None, None]:
        """
        Identifies 'orphan' content blocks that aren't under a heading.
        """
        if node.block_len < min_block_len:
            return 
    
        # Boundary Check
        t_type = node.type.lower()
        # NB: heading-ness is tested via type / is_custom_node, NOT via the presence
        # of an embedding — content leaves now carry embeddings too and must not be
        # mistaken for section boundaries.
        is_boundary = (
            getattr(node, 'is_custom_node', False) or
            t_type.startswith('h')
        )
    
        if is_boundary and node.node.__class__.__name__.lower() != "roottoken":
            for child in node.children:
                yield from SemanticReconciler.find_straggler_branches(child, min_block_len)
            return
    
        # Traversal Logic
        if node.node.__class__.__name__.lower() == "roottoken" or getattr(node, 'has_custom_node', False):
            current_group = []
            for child in node.children:
                child_is_boundary = (
                    getattr(child, 'is_custom_node', False) or
                    child.type.lower().startswith('h') or
                    getattr(child, 'has_custom_node', False)
                )
    
                if not child_is_boundary:
                    if child.block_len >= min_block_len:
                        current_group.append(child)
                else:
                    if current_group:
                        yield current_group
                        current_group = []
                    yield from SemanticReconciler.find_straggler_branches(child, min_block_len)
    
            if current_group:
                yield current_group
        else:
            yield [node]

    #
    # Pruning
    #

    def mark_structural_mismatch(self, node, target_heading: str, node_heading_pairs: dict):
        for child in node.children:
            matched_heading = node_heading_pairs.get(child)
            if matched_heading and target_heading and (matched_heading != target_heading):
                child.is_pruned = True
            else:
                self.mark_structural_mismatch(child, target_heading, node_heading_pairs)

    def mark_semantic_mismatch(self, node, anchor_vector: np.ndarray):
        if anchor_vector is None: return
        for child in node.children:
            # Compare the child's section centroid (merkle roll-up) to the anchor, so
            # pruning reflects what the whole sub-branch is about, not just its heading.
            child_vec = child.mean_emb if child.mean_emb is not None else child.embedding
            if child_vec is not None:
                norm_a, norm_c = np.linalg.norm(anchor_vector), np.linalg.norm(child_vec)
                if norm_a > 0 and norm_c > 0:
                    if (np.dot(anchor_vector, child_vec) / (norm_a * norm_c)) < self.SIMILARITY_THRESHOLD:
                        child.is_pruned = True
                        continue
            self.mark_semantic_mismatch(child, anchor_vector)

    def execute_pruning(self, node):
        for child in list(node.children):
            if child.is_pruned:
                node.children.remove(child)
                child.parent = None
                yield child
            else:
                yield from self.execute_pruning(child)

    def _calc_mean_embedding(self, node):
        """
        Weighted merkle roll-up of a 'Semantic Centroid' (``mean_emb``) for every node.

        Post-order: each node's ``mean_emb`` is the block_len-weighted average of
            - its OWN content vector (``node.embedding``), weighted by its own word
              count (subtree length minus children), and
            - each child's ``mean_emb`` subtree centroid, weighted by the child's block_len.
        """
        for child in node.children:
            self._calc_mean_embedding(child)

        vecs, weights = [], []

        # Own content contribution (headings + leaves have an embedding; containers don't).
        children_total = sum(c.block_len for c in node.children)
        own_len = max(node.block_len - children_total, 0)
        if node.embedding is not None and np.any(node.embedding) and own_len > 0:
            vecs.append(np.asarray(node.embedding, dtype=float))
            if _is_heading(node):
                    weights.append(max(own_len, 1) * 10)  # weight headings more heavily
            else:
                weights.append(max(own_len, 1))

        # Children's subtree centroids.
        for c in node.children:
            if c.mean_emb is not None and np.any(c.mean_emb):
                vecs.append(np.asarray(c.mean_emb, dtype=float))
                weights.append(max(c.block_len, 1))

        if vecs:
            node.mean_emb = np.average(vecs, axis=0, weights=weights)
        elif node.embedding is not None and np.any(node.embedding):
            # Embedded node with no usable weights — fall back to its own vector.
            node.mean_emb = np.asarray(node.embedding, dtype=float)
    

    def deduplicate_by_ancestry(self,nodes: list) -> list:
        node_set = set(nodes)
        result = []
        for node in nodes:
            # Walk up the parent chain
            ancestor = node.parent
            is_redundant = False   
            while ancestor:
                if ancestor in node_set:
                    is_redundant = True
                    break
                ancestor = ancestor.parent
            if not is_redundant:
                result.append(node)
        return result

    def reconcile_structure(self, root: 'EmbedTreeNode', db_headings: List[DB_Heading]) -> Tuple[list, list]:
        """
        does the semantic merge:
        1. Injects LLM headings for orphans.
        2. Matches tree branches to database headings.
        3. Prunes mismatched content.
        """
        
        # 1. Detect Stragglers (Orphan batches)
        straggler_batches = [b for b in self.find_straggler_branches(root, self.MIN_BLOCK_LEN) if b]
        
        # 2. Sample text and Generate Headings via LLM
        # get_sampled_text logic should look at SyntaxTreeNode(node.node).content
        sampled_contents = [get_sampled_text(batch) for batch in straggler_batches]
        generated_headings = self.llm.generate_headings_batch(sampled_contents)

        batch_nodes = []
        for batch, heading_text in zip(straggler_batches, generated_headings):
            print(f"Injecting Custom Node: {heading_text}")
            
            # Use MarkdownIt to create a valid Mistletoe-compatible token structure
            tokens = self.mdit.parse(f"# {heading_text}")
            # tokens[0] is usually the heading_open, tokens[1] is inline
            # We wrap the first relevant token in our SyntaxTreeNode
            heading_token = tokens[0] 
            
            # Create the custom section node
            cus_node = EmbedTreeNode(heading_token, level=batch[0].level or 1)
            cus_node.is_custom_node = True
            cus_node.children = batch
            
            # Surgery: Update parent references for the batch
            for child in batch:
                child.parent = cus_node
            
            # Insert the new branch into the root (or appropriate parent)
            root.add_child(cus_node)
            batch_nodes.append(cus_node)

        # 3. Embed the new custom headings
        if generated_headings:
            vectors = self.embeddings.embed_documents(generated_headings)
            for h_node, h_vector in zip(batch_nodes, vectors):
                h_node.embedding = np.array(h_vector)

        # 4. Calculate Mean Embeddings (Recursive bottom-up)
        self._calc_mean_embedding(root)

        # 5. Match current tree branches to DB headings
        node_heading_pairs = self.match_headings(root, db_headings)

        # 6. Flag Mismatches
        for child in root.children:
            target_heading = node_heading_pairs.get(child)

            if target_heading:
                self.mark_structural_mismatch(child, target_heading, node_heading_pairs)
            
            # If it's a custom node we just made, check if children still belong semantically
            if child.is_custom_node and child.embedding is not None:
                self.mark_semantic_mismatch(child, child.embedding)

        # 7. Execute Pruning (Detach the flagged branches)
        pruned_nodes = list(self.execute_pruning(root))

        # 8. Identify all custom and new nodes for DB updates
        # apply(lambda n: n) gives a flat list of the survived tree
        main_tree_branches = [n for n in root.apply(lambda n: n) if getattr(n, 'is_custom_node', False)]
        all_cust_nodes = main_tree_branches + pruned_nodes
        
        all_matched_nodes = set(node_heading_pairs.keys())
        live_render_nodes = [
            n for n in root.apply(lambda n: n)
            if _is_heading(n) and n.has_embedding
        ]
        all_render_nodes = self.deduplicate_by_ancestry(live_render_nodes + pruned_nodes)

        # New ones = not in DB yet (no match found)
        new_cust_nodes = [n for n in all_render_nodes if n not in all_matched_nodes]
        for node in live_render_nodes:
            print(f"{node.content!r} parent → {node.parent.type if node.parent else None} | {node.parent.content if node.parent else None}")
        return new_cust_nodes, all_render_nodes, node_heading_pairs

def find_closest_cosine_sim(query_vec,list_vecs)->tuple[int,float]:
    """
    Standard vector math helper. Normalizes the vectors and calculates the dot product 
    to find the most semantically similar heading in a list.
    """
    
    # Force list_vecs to be 2D (rows, features)
    if list_vecs.ndim == 1:
        list_vecs = list_vecs[np.newaxis, :]
    query_norm = query_vec / np.linalg.norm(query_vec)
    list_norms = list_vecs / np.linalg.norm(list_vecs,axis=1)[:,np.newaxis]
    similarities = np.dot(list_norms,query_norm)
    closest_idx = np.argmax(similarities)
    return closest_idx,similarities[closest_idx]

def get_sampled_text(batch: list, chunk_size: int = 30) -> str:
    """
    Assembles all text content from a batch of EmbedTreeNodes (and their
    children recursively), then samples first/middle/last N characters.
    """
    def _collect_text(node) -> str:
        parts = []
        content = getattr(node, 'content', '') or ''
        if content.strip():
            parts.append(content.strip())
        for child in getattr(node, 'children', []) or []:
            parts.append(_collect_text(child))
        return ' '.join(filter(None, parts))

    # Assemble full text across all nodes in the batch
    full_text = ' '.join(
        _collect_text(node)
        for node in batch
        if node
    ).strip()

    if not full_text:
        return ""

    n = len(full_text)

    # If short enough return whole thing
    if n <= chunk_size * 3:
        return full_text

    start  = full_text[:chunk_size]
    middle = full_text[n // 2 - chunk_size // 2 : n // 2 + chunk_size // 2]
    end    = full_text[n - chunk_size:]

    return f"{start} [...] {middle} [...] {end}"
if __name__ == "__main__": 
    pass