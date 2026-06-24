"""
test_table_render.py — Targeted table rendering test.

Finds the TABLE node that basic-text.pdf produces, wraps it in a minimal
stub tree, runs it through GdocTreeBuilder, and sends the requests to the
basic-text Google Doc.

Run:
    python sd.py test src/tests/test_table_render.py -s
"""

import pytest
from src.models.tree_nodes import EmbedTreeNode, GdocTreeNode
from src.core.gdocs_renderer import GdocTreeBuilder
from src.services.gdocs_client import GoogleDocsEditor


# ── helpers ───────────────────────────────────────────────────────────────────

def flatten_requests(requests) -> list:
    """
    collect_all_requests can return nested lists (e.g. table fill requests
    are returned as a sub-list). Flatten to a list of plain dicts.
    """
    flat = []
    for item in requests:
        if isinstance(item, list):
            flat.extend(flatten_requests(item))
        elif isinstance(item, dict):
            flat.append(item)
    return flat


def find_table_nodes(node, results=None):
    """Recursively collect all TABLE nodes from an EmbedTreeNode tree."""
    if results is None:
        results = []
    if node.type == "TABLE":
        results.append(node)
    for child in node.children:
        find_table_nodes(child, results)
    return results


def make_stub_root(table_node: EmbedTreeNode) -> EmbedTreeNode:
    """
    Wrap a single node in a minimal ROOT stub so GdocTreeBuilder
    has a proper parent to traverse from.
    """
    class _RootToken:
        pass

    stub = object.__new__(EmbedTreeNode)
    stub.type = "ROOT"
    stub.node = _RootToken()
    stub.level = 0
    stub.parent = None
    stub.children = [table_node]
    stub.embedding = None
    stub.mean_emb = None
    stub.is_custom_node = False
    stub.is_pruned = False
    stub.has_embedding = False
    stub.block_len = table_node.block_len

    original_parent = table_node.parent
    table_node.parent = stub

    return stub, original_parent


# ── tests ─────────────────────────────────────────────────────────────────────

class TestTableRendering:

    def test_table_node_exists_in_tree(self, make_superdoc):
        """basic-text.pdf should produce at least one TABLE node in the embed tree."""
        sd, stream, _ = make_superdoc("basic-text.pdf")
        nested_tree = sd.pdf_to_nested_tree(stream=stream)

        tables = find_table_nodes(nested_tree)
        assert len(tables) > 0, (
            "No TABLE nodes found in the embed tree — "
            "either pymupdf4llm didn't detect the table or SemanticTreeBuilder dropped it"
        )
        print(f"\n[PASS] Found {len(tables)} TABLE node(s)")
        for t in tables:
            print(f"  content preview: {t.content[:80]!r}")

    def test_table_node_has_content(self, make_superdoc):
        """TABLE node should have non-empty content (the raw cell text)."""
        sd, stream, _ = make_superdoc("basic-text.pdf")
        nested_tree = sd.pdf_to_nested_tree(stream=stream)

        tables = find_table_nodes(nested_tree)
        assert len(tables) > 0, "No TABLE nodes found — run test_table_node_exists_in_tree first"

        for table in tables:
            content = getattr(table, 'content', '') or ''
            assert content.strip(), (
                f"TABLE node has empty content — "
                f"GdocTreeBuilder won't know what to insert"
            )
        print(f"\n[PASS] All {len(tables)} TABLE node(s) have content")

    def test_table_generates_gdoc_requests(self, make_superdoc):
        """
        GdocTreeBuilder should produce at least an insertTable request
        for a TABLE node.
        """
        sd, stream, _ = make_superdoc("basic-text.pdf")
        nested_tree = sd.pdf_to_nested_tree(stream=stream)

        tables = find_table_nodes(nested_tree)
        assert len(tables) > 0, "No TABLE nodes found"

        table_node = tables[0]
        stub_root, original_parent = make_stub_root(table_node)

        try:
            builder = GdocTreeBuilder(start_index=1)
            gdoc_root = builder.build(stub_root, {})
            requests = builder.collect_all_requests(gdoc_root)
        finally:
            table_node.parent = original_parent

        flat = flatten_requests(requests)
        assert len(flat) > 0, "GdocTreeBuilder produced zero requests for TABLE node"

        req_types = [list(r.keys())[0] for r in flat]
        print(f"\n[PASS] Generated {len(flat)} requests: {req_types}")

        assert "insertTable" in req_types, (
            f"Expected an insertTable request, got: {req_types}"
        )

    def test_table_renders_to_google_doc(self, make_superdoc):
        """
        Full end-to-end: find the TABLE node, build GdocTree requests,
        and send them to the basic-text Google Doc via batchUpdate.
        This test writes to the live doc — check it visually after running.
        """
        sd, stream, doc_id = make_superdoc("basic-text.pdf")
        nested_tree = sd.pdf_to_nested_tree(stream=stream)

        tables = find_table_nodes(nested_tree)
        assert len(tables) > 0, "No TABLE nodes found in basic-text.pdf"

        table_node = tables[0]
        print(f"\nTable content preview: {table_node.content[:120]!r}")

        # Build the gdoc request tree
        stub_root, original_parent = make_stub_root(table_node)

        try:
            builder = GdocTreeBuilder(start_index=1)
            gdoc_root = builder.build(stub_root, {})
            requests = builder.collect_all_requests(gdoc_root)
        finally:
            table_node.parent = original_parent

        flat = flatten_requests(requests)
        assert len(flat) > 0, "No requests generated — nothing to send"

        req_types = [list(r.keys())[0] for r in flat]
        print(f"Sending {len(flat)} requests to doc {doc_id}: {req_types}")

        # Refresh doc structure so insertion index is current
        sd.docs_editor.get_document_structure(document_id=doc_id)

        # Send to Google Docs
        response = sd.docs_editor.doc_service.documents().batchUpdate(
            documentId=doc_id,
            body={"requests": flat}
        ).execute()

        assert response is not None, "batchUpdate returned None"
        print(f"\n[PASS] Table rendered to doc {doc_id} — check it visually")
        print(f"  https://docs.google.com/document/d/{doc_id}/edit")