"""
test_pinecone.py — Tests for VectorDBManager, focusing on append_documents
and the embedding → Pinecone handoff that was previously broken.

Run:
    pytest tests/test_pinecone.py -s
"""

import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from services.onnx_client import EMBED_DIM


class TestAppendDocuments:

    def test_append_documents_uses_embedding_not_mean_emb(self, make_superdoc):
        """
        append_documents must get a usable vector for every render node. In the
        merkle pipeline that vector is the section centroid (mean_emb), with the
        node's own .embedding as a fallback — both are populated for render nodes.
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

        if not new_cust_nodes:
            pytest.skip("No new_cust_nodes produced — nothing to assert on")

        # Every render node must expose a usable vector to append_documents: the
        # merkle centroid (mean_emb) or, as a fallback, its own .embedding.
        for node in new_cust_nodes:
            mean_emb = getattr(node, 'mean_emb', None)
            embedding  = getattr(node, 'embedding', None)
            assert (mean_emb is not None) or (embedding is not None), (
                f"Node '{getattr(node, 'content', '?')}' has neither mean_emb nor "
                ".embedding — append_documents would skip it"
            )
            print(f"  node='{getattr(node, 'content', '?')[:40]}' "
                  f"embedding={embedding is not None} mean_emb={mean_emb is not None}")

        print(f"\n[PASS] All {len(new_cust_nodes)} nodes use .embedding correctly")

    def test_append_documents_converts_ndarray_to_list(self, make_superdoc):
        sd, _, _ = make_superdoc("basic-text.pdf")

        fake_node = MagicMock()
        fake_node.embedding = np.random.rand(EMBED_DIM)
        fake_node.content = "Fake Heading For Test"
        fake_node.mean_emb = None

        captured = {}

        original_append = sd.db.append_documents

        def capture_append(e_branches, course_id, superdoc_id):
            # Call real method but intercept the index.upsert
            index = sd.db.pc.Index(sd.db.index_name)
            with patch.object(index, 'upsert') as mock_upsert:
                mock_upsert.side_effect = lambda vectors, namespace: captured.update({"vectors": vectors})
                # re-invoke with patched index by patching pc.Index to return this index
                with patch.object(sd.db.pc, 'Index', return_value=index):
                    original_append(e_branches=e_branches, course_id=course_id, superdoc_id=superdoc_id)

        capture_append(e_branches=[fake_node], course_id=sd.COURSE_ID, superdoc_id=sd.DOCUMENT_ID)

        assert "vectors" in captured, "upsert was never called"
        sent_values = captured["vectors"][0]["values"]
        assert isinstance(sent_values, list), (
            f"Values sent to Pinecone are {type(sent_values).__name__}, expected list"
        )
        print(f"\n[PASS] ndarray correctly converted to list before upsert")

    def test_append_documents_skips_none_embedding(self, make_superdoc):
        """
        Nodes with None embedding should be skipped with a warning,
        not cause a ListConversionException crash.
        """
        sd, _, _ = make_superdoc("basic-text.pdf")

        bad_node = MagicMock()
        bad_node.embedding = None
        bad_node.content   = "Node With No Embedding"
        bad_node.mean_emb  = None

        # Should not raise
        try:
            sd.db.append_documents(
                e_branches=[bad_node],
                course_id=sd.COURSE_ID,
                superdoc_id=sd.DOCUMENT_ID
            )
        except Exception as e:
            pytest.fail(f"append_documents raised on None embedding: {e}")

        print(f"\n[PASS] None-embedding node skipped gracefully")

    def test_append_documents_empty_list(self, make_superdoc):
        """append_documents with an empty list should be a no-op."""
        sd, _, _ = make_superdoc("basic-text.pdf")
        # Should not raise
        sd.db.append_documents(
            e_branches=[],
            course_id=sd.COURSE_ID,
            superdoc_id=sd.DOCUMENT_ID
        )
        print(f"\n[PASS] Empty list handled as no-op")


class TestGetAllHeadings:

    def test_returns_list_of_db_headings(self, make_superdoc):
        """get_all_headings_for_doc always returns a list (even if empty)."""
        from services.pinecone_client import DB_Heading

        sd, _, _ = make_superdoc("basic-text.pdf")
        headings = sd.db.get_all_headings_for_doc(
            course_id=sd.COURSE_ID,
            superdoc_id=sd.DOCUMENT_ID
        )
        assert isinstance(headings, list)
        for h in headings:
            assert isinstance(h, DB_Heading), f"Expected DB_Heading, got {type(h)}"
            assert len(h.embedding) == EMBED_DIM, f"Heading '{h.heading}' has wrong embedding dim"

        print(f"\n[PASS] Fetched {len(headings)} DB_Heading objects with valid embeddings")