import numpy as np
import re
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from pinecone import Pinecone, IndexModel, ServerlessSpec

from io import BytesIO
import pymupdf.layout
import pymupdf4llm
import pymupdf

import mistletoe
from core.semantic_renderer import SemanticTreeBuilder
from core.merge_algs import TreeEmbedder, SemanticReconciler
from core.gdocs_renderer import GdocTreeBuilder
from models.tree_nodes import EmbedTreeNode, GdocTreeNode
from services.openai_client import OpenAIProcessor
from services.onnx_client import OnnxProcessor
from services.gdocs_client import GoogleDocsEditor
from services.pinecone_client import VectorDBManager

from lexical.lexical_algs import extract_text_similarity_jaccard
from dotenv import load_dotenv
import os
import time

load_dotenv(os.path.join(os.environ.get('LAMBDA_TASK_ROOT', ''), '.env'))


class superdoc():

    def __init__(
        self,
        DOCUMENT_ID: str | None,
        COURSE_ID: str,
        # Prototype uses its OWN 384-dim index, separate from the shared 1536-dim
        # "superdoc-headings" index, so the local-embedding refactor never collides
        # with teammates' data. Override via PINECONE_PROTOTYPE_INDEX if needed.
        index_name=os.getenv("PINECONE_PROTOTYPE_INDEX", "superdoc-headings-minilm-384")
    ):
        self.DOCUMENT_ID = DOCUMENT_ID
        self.COURSE_ID = COURSE_ID
        self.docs_editor = GoogleDocsEditor()

        if DOCUMENT_ID is None:
            self.DOCUMENT_ID = self.docs_editor.create_google_doc(name=COURSE_ID).get('documentId')

        # Local ONNX MiniLM embedder (384-dim) — replaces OpenAI for all embeddings.
        # Loads the model once; no per-call API cost.
        self.embedder = OnnxProcessor()

        self.db = VectorDBManager(pc=Pinecone(os.environ.get("PINECONE_API_KEY")))
        self.db.initVectorStore(
            index_name=index_name,
            embedding=self.embedder
        )

        self.docs_editor.get_document_structure(document_id=self.DOCUMENT_ID)

        # OpenAI is retained for LLM heading *generation* only; all embeddings are local.
        self.ai = OpenAIProcessor()
        self.emb_model = self.embedder

    # ------------------------------------------------------------------
    #   PDF → Nested Tree
    # ------------------------------------------------------------------

    def pdf_to_nested_tree(self, stream: BytesIO):
        """Convert a PDF stream to a nested semantic tree via mistletoe."""
        doc = pymupdf.open(stream=stream)
        md_text = pymupdf4llm.to_markdown(doc, force_markdown=True)

        mistletoe_doc = mistletoe.Document(md_text)
        with SemanticTreeBuilder() as builder:
            nested_tree = builder.render(mistletoe_doc)

        return nested_tree

    def embed_tree(self, nested_tree):
        """Embed all nodes in the nested tree using TreeEmbedder."""
        tembdr = TreeEmbedder(self.embedder)
        tembdr.embed_tree(nested_tree)
        return nested_tree

    # ------------------------------------------------------------------
    #   Main Merge Algorithm
    # ------------------------------------------------------------------

    def merge_pdf_hierarchical(self, stream: BytesIO):
        """
        Tree-based merge: converts PDF → semantic tree → reconciles with
        existing DB headings → renders to Google Docs.
        """
        #print(f"--- Synchronizing Document ---")
        #self.sync_headings()
        print(f"--- Starting Hierarchical Merge for Doc: {self.DOCUMENT_ID} ---")

        timings = []

        def log_time(name, start_time):
            duration = time.time() - start_time
            timings.append((name, duration))
            print(f"-> {name} took {duration:.2f} seconds.")

        # 1. PDF → Markdown → Nested Semantic Tree
        start = time.time()
        nested_tree = self.pdf_to_nested_tree(stream=stream)
        log_time("PDF to Semantic Tree", start)

        # 2. Embed tree nodes
        start = time.time()
        self.embed_tree(nested_tree)
        log_time("Tree Embedding", start)
        self._log_semantic_tree(nested_tree)

        # 3. Fetch existing headings from Pinecone
        start = time.time()
        existing_headings = self.db.get_all_headings_for_doc(
            course_id=self.COURSE_ID,
            superdoc_id=self.DOCUMENT_ID
        )
        log_time("DB Heading Fetch", start)
        print(f"Fetched {len(existing_headings)} existing headings from DB.")

        # 4. Semantic Reconciliation
        start = time.time()
        smr = SemanticReconciler(
            embedding_service=self.embedder,
            llm_service=self.ai,
            similarity_threshold=0.97,
            min_block_len=20
        )
        new_cust_nodes, all_render_nodes, node_heading_pairs = smr.reconcile_structure(
            nested_tree, existing_headings
        )
        log_time("Semantic Reconciliation", start)
        self._log_reconciliation_results(new_cust_nodes, all_render_nodes, node_heading_pairs)

        '''
        # 5. Lexical Redundancy Check
        if all_render_nodes:
            print("Running Lexical Redundancy Check on Tree Branches...")
            self.docs_editor.get_document_structure(document_id=self.DOCUMENT_ID)

            pruned_render_nodes = []
            pruned_node_heading_pairs = {}

            for branch in all_render_nodes:
                heading_text = node_heading_pairs.get(branch, "")
                children = getattr(branch, 'children', [])

                if children:
                    print(f"Checking {len(children)} nodes under: '{heading_text}'")
                    branch.children = self.lexical_redundancyCheck(
                        nodes=children,
                        heading=heading_text
                    )

                if not hasattr(branch, 'children') or len(branch.children) > 0:
                    pruned_render_nodes.append(branch)
                    if branch in node_heading_pairs:
                        pruned_node_heading_pairs[branch] = node_heading_pairs[branch]

            all_render_nodes = pruned_render_nodes
            node_heading_pairs = pruned_node_heading_pairs

        '''
        # 6. Sync new headings to Vector DB
        if new_cust_nodes:
            print(f"Embedding {len(new_cust_nodes)} new custom heading nodes...")
            tembdr = TreeEmbedder(self.ai)
            for node in new_cust_nodes:
                tembdr.embed_tree(node)   # embed each branch in place

            print(f"Syncing {len(new_cust_nodes)} new custom headings to Pinecone...")
            self.db.append_documents(
                e_branches=new_cust_nodes,
                course_id=self.COURSE_ID,
                superdoc_id=self.DOCUMENT_ID
            )

        # 7. Render to Google Docs
        start = time.time()
        print(f"Rendering {len(all_render_nodes)} branches to Google Doc...")
        self.docs_editor.get_document_structure(document_id=self.DOCUMENT_ID)
        self.docs_editor.mutate_named_ranges(document_id=self.DOCUMENT_ID)
        self.docs_editor.render_trees(
            superdoc_id=self.DOCUMENT_ID,
            all_render_nodes=all_render_nodes,
            node_heading_pairs=node_heading_pairs
        )
        log_time("Google Docs Rendering", start)

        self._log_performance_summary(timings)

    # ------------------------------------------------------------------
    #   Heading Management
    # ------------------------------------------------------------------

    def delete_heading(self, old_heading: str):
        self.docs_editor.delete_heading(old_heading=old_heading)
        self.db.remove_vectordb_heading(
            heading=old_heading,
            course_id=self.COURSE_ID,
            superdoc_id=self.DOCUMENT_ID
        )

    def create_heading(self, new_heading: str):
        self.docs_editor.create_heading(new_heading=new_heading)
        self.db.create_vectordb_heading(
            heading_text=new_heading,
            course_id=self.COURSE_ID,
            superdoc_id=self.DOCUMENT_ID
        )

    def update_heading(self, old_heading: str, new_heading: str):
        """Updates old_heading in both Google Docs and VectorDB."""
        self.docs_editor.update_heading(old_heading=old_heading, new_heading=new_heading)
        self.db.replace_vectordb_heading_with_text(
            old_heading=old_heading,
            new_heading_text=new_heading,
            course_id=self.COURSE_ID,
            superdoc_id=self.DOCUMENT_ID
        )

    def sync_headings(self):
        self.fix_new_content()
        self.fix_heading_update()

    def fix_heading_update(self):
        """
        Synchronizes the document structure from Google Docs with Pinecone.
        Finds changed or deleted headings and processes them in batch.
        """
        self.docs_editor.get_document_structure(document_id=self.DOCUMENT_ID)
        doc_headings = self.db.get_all_headings_for_doc(
            course_id=self.COURSE_ID,
            superdoc_id=self.DOCUMENT_ID
        )

        delete_ids = []
        headings_to_embed = []

        #print text content of headings not the entire headings object
        #print(f"Num of vector-db-headings: {doc_headings}")

        for db_entry in doc_headings:
            heading_text = db_entry.get("heading")
            vector_id = db_entry.get("id")

            content_under_heading = self.docs_editor.get_text_in_range_from_doc_obj(heading_text)
            print(f"Content of {heading_text}: {content_under_heading}")

            if not content_under_heading:
                delete_ids.append(vector_id)
                continue

            prefix_target = f"{heading_text}:"
            if prefix_target not in content_under_heading:
                delete_ids.append(vector_id)

                parts = content_under_heading.split(":", 1)
                potential_new_heading = parts[0].strip()

                if potential_new_heading:
                    headings_to_embed.append(potential_new_heading)

        vectors_to_upsert = []
        if headings_to_embed:
            print(f"Batch embedding {len(headings_to_embed)} changed headings...")
            for text_chunk in self._chunk_list(headings_to_embed, 500):
                embedded_vectors = self.emb_model.embed_documents(text_chunk)
                for text, vector in zip(text_chunk, embedded_vectors):
                    vectors_to_upsert.append({
                        "id": self.db.generate_timestamp_id(self.COURSE_ID),
                        "values": vector,
                        "metadata": {
                            "superdoc": self.DOCUMENT_ID,
                            "heading": text,
                        }
                    })

        if delete_ids:
            self.db.batch_delete_vectors(vector_ids=delete_ids, course_id=self.COURSE_ID)

        if vectors_to_upsert:
            self.db.batch_upsert_vectors(vectors=vectors_to_upsert, course_id=self.COURSE_ID)

        print("Superdoc synchronization complete.")

    def fix_new_content(self):
        """
        Scans un-indexed document gaps for user-defined <HEADING>: tags,
        formats them in the Doc, and syncs to Pinecone.
        """
        self.docs_editor.get_document_structure(document_id=self.DOCUMENT_ID)
        skips = self.docs_editor.catch_skips()

        vectors_to_upsert = []
        headings_queue = []

        for skip in skips:
            start_idx = skip.get('startIndex')
            end_idx = skip.get('endIndex')

            text = self.docs_editor.get_text_in_indices_from_doc_obj(start_idx, end_idx)

            if not text:
                continue

            matches = list(re.finditer(r"<(.*?)>:", text))

            if not matches:
                continue

            for match in matches:
                extracted_heading = match.group(1).strip()

                text_before_match = text[:match.start()]
                utf16_offset = self.docs_editor.text_utf16_len(text_before_match)
                tag_start_in_doc = start_idx + utf16_offset

                print(f"Detected manual heading tag: '{extracted_heading}'")

                headings_queue.append({
                    'heading': extracted_heading,
                    'startIndex': tag_start_in_doc
                })

                content_after_tag = text[match.end():].strip()[:500]
                vectors_to_upsert.append({
                    "heading": extracted_heading,
                    "content_sample": content_after_tag
                })

        if headings_queue:
            self.docs_editor.batch_create_headings_and_named_ranges(headings_queue)

        if vectors_to_upsert:
            print(f"Batch embedding {len(vectors_to_upsert)} manual headings...")
            heading_strings = [v["heading"] for v in vectors_to_upsert]
            embedded_vectors = self.emb_model.embed_documents(heading_strings)

            pinecone_payload = []
            for i, vector in enumerate(embedded_vectors):
                original_data = vectors_to_upsert[i]
                pinecone_payload.append({
                    "id": self.db.generate_timestamp_id(self.COURSE_ID),
                    "values": vector,
                    "metadata": {
                        "superdoc": self.DOCUMENT_ID,
                        "heading": original_data["heading"],
                        "content_preview": original_data["content_sample"][:200]
                    }
                })

            self.db.batch_upsert_vectors(vectors=pinecone_payload, course_id=self.COURSE_ID)

        print("Superdoc manual tag sync complete.")

    # ------------------------------------------------------------------
    #   Google Doc Utilities
    # ------------------------------------------------------------------

    def create_document(self, name: str, course_id: str):
        return self.docs_editor.create_google_doc(name=name, courseid=course_id)

    def get_docids(self, course_id: str) -> dict:
        document_ids = self.docs_editor.get_docids(courseid=course_id)
        docs_map = {}
        for docid in document_ids:
            doc_structure = self.docs_editor.get_document_structure(docid)
            doc_name = doc_structure.get('title', 'Untitled Document')
            docs_map[doc_name] = docid
        return docs_map

    # ------------------------------------------------------------------
    #   Lexical Redundancy
    # ------------------------------------------------------------------

    def lexical_redundancyCheck(self, nodes: list, heading: str):
        print(f"\n[DEBUG] Starting Check for {len(nodes)} nodes under '{heading}'")

        pruned_tree = []
        LEXICAL_CHECK_CONSTANT = 0.7

        raw_doc_text = self.docs_editor.get_text_in_range_from_doc_obj(heading)
        existing_lines = [line.strip() for line in (raw_doc_text or "").split("\n") if line.strip()]
        print(f"  [CACHE] {len(existing_lines)} lines in GDoc under '{heading}'")

        for i, node in enumerate(nodes):
            node_text = self._extract_leaf_text(node)

            if not node_text:
                print(f"  [NODE {i}] SKIP: No text found.")
                pruned_tree.append(node)
                continue

            is_redundant = False
            highest_score = 0

            for line in existing_lines:
                score = extract_text_similarity_jaccard(node_text, line)
                if score > highest_score:
                    highest_score = score
                if score >= LEXICAL_CHECK_CONSTANT:
                    is_redundant = True
                    break

            if not is_redundant:
                pruned_tree.append(node)
                print(f"  [KEEP]  '{node_text[:60]}' (max sim: {highest_score:.2f})")
            else:
                print(f"  [PRUNE] '{node_text[:60]}' (score: {highest_score:.2f})")

        print(f"[DEBUG] Done. Kept {len(pruned_tree)}/{len(nodes)}\n")
        return pruned_tree

    def _extract_leaf_text(self, node) -> str:
        """Walk down the tree until we find a node with actual content."""
        content = node.content.strip() if getattr(node, 'content', None) else ""
        if content:
            return content
        for child in getattr(node, 'children', []):
            result = self._extract_leaf_text(child)
            if result:
                return result
        return ""

    # ------------------------------------------------------------------
    #   Logging / Diagnostics
    # ------------------------------------------------------------------

    def _log_semantic_tree(self, node, indent=0):
        """
        Recursively prints the embedded semantic tree after TreeEmbedder runs.
        Shows node type, word count, embedding snippet, and content preview.
        Mirrors print_semantic_tree() from the test suite.
        """
        pref = "  " * indent

        # Content label
        raw_label = getattr(node, 'content', '') or ''

        # Embedding snippet — first 3 dimensions to 5dp
        if getattr(node, 'embedding', None) is not None:
            snippet = [f"{v:.5f}" for v in node.embedding[:3]]
            emb_str = f" [Emb: {', '.join(snippet)}...]"
        else:
            emb_str = " [No Emb]"

        t_type = getattr(node, 'type', '?')

        if t_type == 'TABLE':
            label = " | " + " ".join(raw_label.split())
        else:
            label = f": {raw_label[:80]}..." if len(raw_label) > 80 else f": {raw_label}"

        stats = f" [Words: {getattr(node, 'block_len', '?')}]{emb_str}"
        print(f"{pref}└── [{t_type}]{stats}{label}")

        for child in getattr(node, 'children', []):
            self._log_semantic_tree(child, indent + 1)

    def _log_reconciliation_results(self, new_cust_nodes, all_render_nodes, node_heading_pairs):
        """
        Prints a summary of what the SemanticReconciler produced:
        which headings are new, which render nodes matched existing headings,
        and a per-node tree dump.
        Mirrors the debug block in test_render_to_gdocs() from the test suite.
        """
        print(f"\n--- Reconciliation Results ---")
        print(f"New headings to insert into DB : {len(new_cust_nodes)}")
        for node in new_cust_nodes:
            print(f"  + {node.content!r}")

        print(f"All render nodes (deduplicated): {len(all_render_nodes)}")
        for node in all_render_nodes:
            matched = node_heading_pairs.get(node)
            label = (node.content.strip()[:60] if getattr(node, 'content', None) else repr(node))
            status = f"[MATCHED: {matched}]" if matched else "[UNMATCHED]"
            print(f"  {status} {label}")

        print(f"\n--- All Render Node Trees (post-reconciliation) ---")
        for i, node in enumerate(all_render_nodes):
            matched = node_heading_pairs.get(node)
            status = f"[MATCHED: {matched}]" if matched else "[UNMATCHED]"
            print(f"\n[Render Node {i + 1}] {status}")
            parent = getattr(node, 'parent', None)
            parent_label = (
                f"{parent.type} | {parent.content}" if parent else "None"
            )
            print(f"  parent: {parent_label}")
            self._log_semantic_tree(node)

    def _log_gdoc_tree(self, gdoc_root: 'GdocTreeNode', indent: int = 0):
        """
        Recursively prints a GdocTreeNode with its content preview and
        every Google Docs request it will generate (insertText, insertTable,
        updateTextStyle, etc.).
        Mirrors print_gdoc_tree() + _print_request() from the test suite.
        """
        prefix = "  " * indent
        node_type = getattr(gdoc_root, 'type', '?')
        content = getattr(gdoc_root, 'content', '') or ''
        content_preview = repr(content.strip()[:50]) if content.strip() else ''

        print(f"{prefix}[{node_type}] {content_preview}")

        for req in getattr(gdoc_root, 'requests', []) or []:
            req_type = list(req.keys())[0]
            self._log_gdoc_request(req_type, req[req_type], prefix + "  ")

        for req in getattr(gdoc_root, '_table_fill', []) or []:
            req_type = list(req.keys())[0]
            print(f"{prefix}  [TABLE_FILL]")
            self._log_gdoc_request(req_type, req[req_type], prefix + "    ")

        for child in getattr(gdoc_root, 'children', []) or []:
            self._log_gdoc_tree(child, indent + 1)

    def _log_gdoc_request(self, req_type: str, details: dict, prefix: str):
        """Formats a single Google Docs API request dict for terminal output."""
        if req_type == "insertText":
            text = details.get('text', '')
            idx = details.get('location', {}).get('index', '?')
            print(f"{prefix}→ insertText       @ {idx:<6} {repr(text[:40])}")

        elif req_type == "insertTable":
            idx = details.get('location', {}).get('index', '?')
            rows = details.get('rows', '?')
            cols = details.get('columns', '?')
            print(f"{prefix}→ insertTable      @ {idx:<6} [{rows}x{cols}]")

        elif req_type == "insertInlineImage":
            idx = details.get('location', {}).get('index', '?')
            uri = details.get('uri', '')[:40]
            print(f"{prefix}→ insertImage      @ {idx:<6} {uri}")

        elif req_type == "updateTextStyle":
            r = details.get('range', {})
            style = details.get('textStyle', {})
            print(f"{prefix}→ updateTextStyle  @ [{r.get('startIndex', '?')},{r.get('endIndex', '?')}] {style}")

        elif req_type == "updateParagraphStyle":
            r = details.get('range', {})
            style = details.get('paragraphStyle', {})
            named = style.get('namedStyleType', '')
            indent_start = style.get('indentStart', {}).get('magnitude', '')
            print(f"{prefix}→ updateParaStyle  @ [{r.get('startIndex', '?')},{r.get('endIndex', '?')}] {named} indent={indent_start}")

        elif req_type == "createParagraphBullets":
            r = details.get('range', {})
            preset = details.get('bulletPreset', '?')
            print(f"{prefix}→ createBullets    @ [{r.get('startIndex', '?')},{r.get('endIndex', '?')}] {preset}")

        else:
            print(f"{prefix}→ {req_type}")

    def _log_performance_summary(self, timings: list):
        """Prints a sorted performance summary table. Pass the timings list from merge_pdf_hierarchical."""
        print("\n" + "=" * 50)
        print("PERFORMANCE SUMMARY (Sorted by time):")
        for name, duration in sorted(timings, key=lambda x: x[1], reverse=True):
            print(f"{name:<35} : {duration:.2f}s")
        print("=" * 50 + "\n")

    # ------------------------------------------------------------------
    #   Helpers
    # ------------------------------------------------------------------

    def _chunk_list(self, lst, n):
        """Split a list into batch-safe chunks of size n."""
        for i in range(0, len(lst), n):
            yield lst[i:i + n]


# ----------------------------------------------------------------------
#   Entry Point
# ----------------------------------------------------------------------

if __name__ == '__main__':
    with open("files/basic-text.pdf", "rb") as f:
        pdf_bytes = f.read()
    strm = BytesIO(pdf_bytes)

    sd = superdoc(
        DOCUMENT_ID='1Q1whz1kFN9wj1_mamWgaDbKh7przNmc5owdOSNovC04',
        COURSE_ID="Goof1202"
    )
    sd.merge_pdf_hierarchical(stream=strm)