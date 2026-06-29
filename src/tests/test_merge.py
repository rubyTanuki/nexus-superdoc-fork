"""
test_merge.py — Tests for the full merge_pdf_hierarchical pipeline.

Each test targets a specific PDF in /files and asserts on the
output of a particular pipeline stage.

Run a single test:
    pytest tests/test_merge.py::test_full_merge_basic_text -s
Run all merge tests:
    pytest tests/test_merge.py -s
"""

import pytest
import numpy as np
from services.onnx_client import EMBED_DIM


# --------------------------------------------------------------------------
# Full pipeline tests
# --------------------------------------------------------------------------

class TestFullMerge:

    def test_full_merge_basic_text(self, make_superdoc):
        """Smoke test: full pipeline runs without raising on a simple PDF."""
        sd, stream, doc_id = make_superdoc("basic-text.pdf")
        # Should complete without exception
        sd.merge_pdf_hierarchical(stream=stream)
        print(f"\n[PASS] Merged basic-text.pdf → doc {doc_id}")

    def test_full_merge_complex_pdf(self, make_superdoc):
        """Full pipeline on a more complex academic paper."""
        sd, stream, doc_id = make_superdoc("2301.07041v2.pdf")
        sd.merge_pdf_hierarchical(stream=stream)
        print(f"\n[PASS] Merged 2301.07041v2.pdf → doc {doc_id}")


# --------------------------------------------------------------------------
# Stage-level tests (run individual pipeline stages and assert on outputs)
# --------------------------------------------------------------------------

class TestPipelineStages:

    def test_pdf_to_nested_tree(self, make_superdoc):
        """PDF → mistletoe semantic tree produces a non-empty root."""
        sd, stream, _ = make_superdoc("basic-text.pdf")
        nested_tree = sd.pdf_to_nested_tree(stream=stream)

        assert nested_tree is not None, "pdf_to_nested_tree returned None"
        assert len(nested_tree.children) > 0, "Semantic tree root has no children"
        print(f"\n[PASS] Tree has {len(nested_tree.children)} top-level children")

    def test_embed_tree_populates_embeddings(self, make_superdoc):
        """After embed_tree, heading nodes should have non-None embeddings."""
        sd, stream, _ = make_superdoc("basic-text.pdf")
        nested_tree = sd.pdf_to_nested_tree(stream=stream)
        sd.embed_tree(nested_tree)

        embedded = [
            n for n in nested_tree.apply(lambda x: x)
            if getattr(n, 'has_embedding', False)
        ]
        assert len(embedded) > 0, "No nodes received embeddings"

        for node in embedded:
            emb = getattr(node, 'embedding', None)
            assert emb is not None, f"Node '{node.content}' has has_embedding=True but embedding is None"
            assert len(emb) == EMBED_DIM, f"Expected {EMBED_DIM}-dim embedding, got {len(emb)}"

        print(f"\n[PASS] {len(embedded)} nodes embedded at {EMBED_DIM} dims")

    def test_reconcile_returns_valid_structure(self, make_superdoc):
        """reconcile_structure returns three non-None values with consistent types."""
        from src.core.merge_algs import SemanticReconciler

        sd, stream, _ = make_superdoc("basic-text.pdf")
        nested_tree = sd.pdf_to_nested_tree(stream=stream)
        sd.embed_tree(nested_tree)

        existing_headings = sd.db.get_all_headings_for_doc(
            course_id=sd.COURSE_ID,
            superdoc_id=sd.DOCUMENT_ID
        )

        smr = SemanticReconciler(
            embedding_service=sd.embedder,
            llm_service=sd.ai,
            similarity_threshold=0.97,
            min_block_len=20
        )
        new_cust_nodes, all_render_nodes, node_heading_pairs = smr.reconcile_structure(
            nested_tree, existing_headings
        )

        assert isinstance(new_cust_nodes, list),    "new_cust_nodes should be a list"
        assert isinstance(all_render_nodes, list),  "all_render_nodes should be a list"
        assert isinstance(node_heading_pairs, dict), "node_heading_pairs should be a dict"

        print(f"\n[PASS] Reconciliation: {len(new_cust_nodes)} new, "
              f"{len(all_render_nodes)} render nodes, "
              f"{len(node_heading_pairs)} matched pairs")

    def test_new_cust_nodes_have_embeddings(self, make_superdoc):
        """
        Every node going into append_documents must have a non-None embedding
        that is a list (not a numpy array) so Pinecone won't reject it.
        """
        from src.core.merge_algs import SemanticReconciler

        sd, stream, _ = make_superdoc("basic-text.pdf")
        nested_tree = sd.pdf_to_nested_tree(stream=stream)
        sd.embed_tree(nested_tree)

        existing_headings = sd.db.get_all_headings_for_doc(
            course_id=sd.COURSE_ID,
            superdoc_id=sd.DOCUMENT_ID
        )

        smr = SemanticReconciler(
            embedding_service=sd.embedder,
            llm_service=sd.ai,
            similarity_threshold=0.97,
            min_block_len=20
        )
        new_cust_nodes, _, _ = smr.reconcile_structure(nested_tree, existing_headings)

        for node in new_cust_nodes:
            # append_documents stores the merkle centroid (mean_emb), falling back to
            # the node's own .embedding. Container/straggler render nodes carry only
            # mean_emb, so a usable vector is "mean_emb OR embedding".
            emb = getattr(node, 'mean_emb', None)
            if emb is None:
                emb = getattr(node, 'embedding', None)
            assert emb is not None, (
                f"Node '{getattr(node, 'content', '?')}' has neither mean_emb nor "
                "embedding — append_documents would skip it"
            )
            # Pinecone requires a plain list, not ndarray
            converted = emb.tolist() if hasattr(emb, 'tolist') else emb
            assert isinstance(converted, list), "Embedding must be convertible to list"

        print(f"\n[PASS] All {len(new_cust_nodes)} new_cust_nodes have valid embeddings")


# --------------------------------------------------------------------------
# Heading management tests
# --------------------------------------------------------------------------

class TestHeadingManagement:

    def test_create_and_delete_heading(self, make_superdoc):
        sd, _, _ = make_superdoc("basic-text.pdf")
        heading = "Test Heading Pytest"

        sd.create_heading(new_heading=heading)
        sd.docs_editor.get_document_structure(document_id=sd.DOCUMENT_ID)  # refresh
        sd.delete_heading(old_heading=heading)

    def test_update_heading(self, make_superdoc):
        sd, _, _ = make_superdoc("basic-text.pdf")
        old_h = "Old Pytest Heading"
        new_h = "New Pytest Heading"
    
        sd.create_heading(new_heading=old_h)
        sd.docs_editor.get_document_structure(document_id=sd.DOCUMENT_ID)  # refresh
        sd.update_heading(old_heading=old_h, new_heading=new_h)
        sd.docs_editor.get_document_structure(document_id=sd.DOCUMENT_ID)  # refresh
        sd.delete_heading(old_heading=new_h)
    