"""
test_tree.py — Unit tests for SemanticTreeBuilder and TreeEmbedder.

These tests are intentionally lightweight: they don't hit Google Docs
and only optionally hit OpenAI (skipped if no API key is set).

Run:
    pytest tests/test_tree.py -s
"""

import pytest
import os
import mistletoe
from io import BytesIO
from pathlib import Path
from services.onnx_client import EMBED_DIM

FILES_DIR = Path(__file__).parent.parent / "files"


class TestSemanticTreeBuilder:

    def test_tree_has_root(self, make_superdoc):
        """The root node returned by pdf_to_nested_tree has type ROOT."""
        sd, stream, _ = make_superdoc("basic-text.pdf")
        tree = sd.pdf_to_nested_tree(stream=stream)
        assert tree is not None
        assert tree.type.upper() == "ROOT", f"Expected ROOT, got {tree.type}"

    def test_tree_children_have_types(self, make_superdoc):
        """All direct children of root have a non-empty type string."""
        sd, stream, _ = make_superdoc("basic-text.pdf")
        tree = sd.pdf_to_nested_tree(stream=stream)
        for child in tree.children:
            assert child.type, f"Child node has empty type: {child}"

    def test_heading_nodes_present(self, make_superdoc):
        """At least one HEADING node exists anywhere in the tree."""
        sd, stream, _ = make_superdoc("basic-text.pdf")
        tree = sd.pdf_to_nested_tree(stream=stream)

        all_nodes = list(tree.apply(lambda n: n))
        heading_nodes = [n for n in all_nodes if n.type.lower().startswith('h')]
        assert len(heading_nodes) > 0, "No heading nodes found in tree"
        print(f"\n[PASS] Found {len(heading_nodes)} heading nodes")

    def test_paragraph_nodes_have_content(self, make_superdoc):
        """PARAGRAPH nodes should have non-empty content."""
        sd, stream, _ = make_superdoc("basic-text.pdf")
        tree = sd.pdf_to_nested_tree(stream=stream)

        all_nodes = list(tree.apply(lambda n: n))
        paragraphs = [n for n in all_nodes if n.type.upper() == "PARAGRAPH"]

        empty = [n for n in paragraphs if not getattr(n, 'content', '').strip()]
        pct_empty = len(empty) / len(paragraphs) * 100 if paragraphs else 0

        # Allow up to 20% empty paragraphs (whitespace-only lines etc.)
        assert pct_empty < 20, (
            f"{pct_empty:.0f}% of PARAGRAPH nodes have no content "
            f"({len(empty)}/{len(paragraphs)})"
        )
        print(f"\n[PASS] {len(paragraphs)} paragraphs, {len(empty)} empty ({pct_empty:.1f}%)")


class TestTreeEmbedder:

    def test_block_len_calculated(self, make_superdoc):
        """Every node should have block_len >= 0 after embed_tree."""
        sd, stream, _ = make_superdoc("basic-text.pdf")
        tree = sd.pdf_to_nested_tree(stream=stream)
        sd.embed_tree(tree)

        all_nodes = list(tree.apply(lambda n: n))
        for node in all_nodes:
            bl = getattr(node, 'block_len', None)
            assert bl is not None, f"Node '{node.type}' missing block_len"
            assert bl >= 0, f"Negative block_len on node '{node.type}'"

    def test_embedding_dimensions(self, make_superdoc):
        """Embedded nodes must be exactly EMBED_DIM-dimensional (local MiniLM = 384)."""
        sd, stream, _ = make_superdoc("basic-text.pdf")
        tree = sd.pdf_to_nested_tree(stream=stream)
        sd.embed_tree(tree)

        embedded = [
            n for n in tree.apply(lambda n: n)
            if getattr(n, 'has_embedding', False)
        ]
        assert len(embedded) > 0, "No nodes were embedded"

        for node in embedded:
            assert len(node.embedding) == EMBED_DIM, (
                f"Node '{node.content[:40]}' has {len(node.embedding)}-dim embedding"
            )

    def test_no_embedding_on_root(self, make_superdoc):
        """The root node itself should not receive an embedding."""
        sd, stream, _ = make_superdoc("basic-text.pdf")
        tree = sd.pdf_to_nested_tree(stream=stream)
        sd.embed_tree(tree)

        assert not getattr(tree, 'has_embedding', False), \
            "Root node should not be embedded"

    def test_min_block_len_filter(self, make_superdoc):
        """Nodes with block_len < MIN_BLOCK_LEN should not be embedded."""
        from src.core.merge_algs import MIN_BLOCK_LEN

        sd, stream, _ = make_superdoc("basic-text.pdf")
        tree = sd.pdf_to_nested_tree(stream=stream)
        sd.embed_tree(tree)

        violations = [
            n for n in tree.apply(lambda n: n)
            if getattr(n, 'has_embedding', False) and n.block_len < MIN_BLOCK_LEN
        ]
        assert len(violations) == 0, (
            f"{len(violations)} nodes were embedded despite block_len < {MIN_BLOCK_LEN}: "
            + ", ".join(f"'{n.content[:30]}' (bl={n.block_len})" for n in violations[:3])
        )