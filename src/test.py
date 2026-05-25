import pymupdf
import pymupdf4llm

from io import BytesIO

import mistletoe
from mistletoe.block_token import BlockToken, Heading, Paragraph
from mistletoe.span_token import SpanToken, RawText

from src.models.tree_nodes import EmbedTreeNode
from src.core.semantic_renderer import SemanticTreeBuilder
from src.core.gdocs_renderer import GdocTreeBuilder
from src.core.merge_algs import TreeEmbedder, SemanticReconciler, DB_Heading
from src.services.openai_client import OpenAIProcessor
from src.services.pinecone_client import VectorDBManager

from pinecone import Pinecone, IndexModel, ServerlessSpec

from langchain_openai import OpenAIEmbeddings,ChatOpenAI



import os
from dotenv import load_dotenv
load_dotenv()
# --- 3. DEBUG PRINTERS ---
def print_raw_ast(token, indent=0):
    """Shows mistletoe's native flat structure."""
    t_name = token.__class__.__name__
    t_info = f" (L{token.level})" if isinstance(token, Heading) else ""
    print("  " * indent + f"[{t_name}]{t_info}")
    if hasattr(token, 'children') and token.children:
        for c in token.children:
            if isinstance(c, (BlockToken, SpanToken)):
                print_raw_ast(c, indent + 1)

def print_semantic_tree(node, indent=0):
    pref = "  " * indent
    
    # 1. Extract text content using the Syntax helper
    if hasattr(node, 'node') and node.node:
        # Use the helper to reach into the mistletoe token
        raw_label = EmbedTreeNode(node.node).content
    else:
        raw_label = ""

    # 2. Format Embedding snippet (First 5 digits of the first 3 dimensions)
    emb_str = ""
    if hasattr(node, 'embedding') and node.embedding is not None:
        # Take the first 3 values and format each to 5 decimal places
        snippet = [f"{val:.5f}" for val in node.embedding[:3]]
        emb_str = f" [Emb: {', '.join(snippet)}...]"
    else:
        emb_str = " [No Emb]"

    # 3. Access attributes defined in your __slots__
    t_type = node.type
    
    if t_type == 'TABLE':
        # Table formatting
        label = " | " + " ".join(raw_label.split())
    else:
        # Paragraph/Heading formatting
        label = f": {raw_label[:80]}..." if len(raw_label) > 80 else f": {raw_label}"
    
    # 4. Assemble the final debug line
    stats = f" [Words: {node.block_len}]{emb_str}"
    print(f"{pref}└── [{t_type}]{stats}{label}")
    
    # 5. Recurse
    for child in node.children:
        print_semantic_tree(child, indent + 1)



# --- 4. EXECUTION ---
def test_pdf_structure(file_path):
    print(f"--- Processing: {file_path} ---")

    # PDF → Markdown
    doc = pymupdf.open(file_path)
    md_text = pymupdf4llm.to_markdown(doc, force_markdown=True)
    print('----Markdown----')
    print(md_text)
    print('----End-Of-MD---')

    # Shared AI client
    ai = OpenAIProcessor()

    # Parse with Mistletoe + build semantic tree
    mistletoe_doc = mistletoe.Document(md_text)
    print("\nNESTED SEMANTIC TREE (The Stack Output)")
    with SemanticTreeBuilder() as builder:
        nested_tree = builder.render(mistletoe_doc)

    # Embed tree nodes
    tembdr = TreeEmbedder(ai)
    tembdr.embed_tree(nested_tree)
    print_semantic_tree(nested_tree)
    print("\n" + "=" * 50)

    # Fetch existing DB headings
    index_name = os.getenv("PINECONE_INDEX", "superdoc-headings")
    vec_db = VectorDBManager(pc=Pinecone(os.environ.get("PINECONE_API_KEY")))
    vec_db.initVectorStore(index_name=index_name, embedding=OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY")))

    COURSE_ID = "Goof1202"
    DOCUMENT_ID = "1Q1whz1kFN9wj1_mamWgaDbKh7przNmc5owdOSNovC04"
    existing_headings =  vec_db.get_all_headings_for_doc(course_id=COURSE_ID,superdoc_id=DOCUMENT_ID)
    print(f"Fetched {len(existing_headings)} existing headings from DB.")

    # Run merge / reconciliation
    smr = SemanticReconciler(
        embedding_service=ai,
        llm_service=ai,              # OpenAIProcessor satisfies both interfaces
        similarity_threshold=0.97,   # matches your module-level SIMILARITY_THRESHOLD
        min_block_len=20             # matches MIN_BLOCK_LEN
    )
    new_cust_nodes, all_cust_nodes = smr.reconcile_structure(nested_tree, existing_headings)

    print(f"\n--- Reconciliation Results ---")
    print(f"New headings to insert into DB : {len(new_cust_nodes)}")
    for node in new_cust_nodes:
        print(f"  + {getattr(node, 'heading', repr(node))}")

    print(f"All custom/pruned nodes tracked: {len(all_cust_nodes)}")
    for node in all_cust_nodes:
        pruned = getattr(node, 'is_pruned', False)
        label = getattr(node, 'heading', repr(node))
        print(f"  {'[PRUNED]' if pruned else '[KEPT  ]'} {label}")



def test_render_to_gdocs(file_path):
    print(f"--- Processing: {file_path} ---")
 
    # PDF → Markdown
    doc = pymupdf.open(file_path)
    md_text = pymupdf4llm.to_markdown(doc, force_markdown=True)
    print('----Markdown----')
    print(md_text)
    print('----End-Of-MD---')
 
    # Shared AI client
    ai = OpenAIProcessor()
 
    # Parse with Mistletoe + build semantic tree
    mistletoe_doc = mistletoe.Document(md_text)
    print("\nNESTED SEMANTIC TREE (The Stack Output)")
    with SemanticTreeBuilder() as builder:
        nested_tree = builder.render(mistletoe_doc)
 
    # Embed tree nodes
    tembdr = TreeEmbedder(ai)
    tembdr.embed_tree(nested_tree)
    print_semantic_tree(nested_tree)
    print("\n" + "=" * 50)
 
    # Fetch existing DB headings
    index_name = os.getenv("PINECONE_INDEX", "superdoc-headings")
    vec_db = VectorDBManager(pc=Pinecone(os.environ.get("PINECONE_API_KEY")))
    vec_db.initVectorStore(
        index_name=index_name,
        embedding=OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))
    )
    COURSE_ID = "Goof1202"
    DOCUMENT_ID = "1Q1whz1kFN9wj1_mamWgaDbKh7przNmc5owdOSNovC04"
    existing_headings = vec_db.get_all_headings_for_doc(
        course_id=COURSE_ID,
        superdoc_id=DOCUMENT_ID
    )
    print(f"Fetched {len(existing_headings)} existing headings from DB.")
 
    # Run merge / reconciliation
    smr = SemanticReconciler(
        embedding_service=ai,
        llm_service=ai,
        similarity_threshold=0.97,
        min_block_len=20
    )
    new_cust_nodes, all_render_nodes, node_heading_pairs = smr.reconcile_structure(
        nested_tree, existing_headings
    )
 
 
    print(f"\n--- Reconciliation Results ---")
    print(f"New headings to insert into DB : {len(new_cust_nodes)}")
    for node in new_cust_nodes:
        print(f"  + {node.content!r}")
    print(f"All render nodes (deduplicated): {len(all_render_nodes)}")
    for node in all_render_nodes:
        matched = node_heading_pairs.get(node)
        label = node.content.strip()[:60] if node.content else repr(node)
        status = f"[MATCHED: {matched}]" if matched else "[UNMATCHED]"
        print(f"  {status} {label}")
 
    print(f"\n--- All Render Node Trees (post-reconciliation) ---")
    for i, node in enumerate(all_render_nodes):
        matched = node_heading_pairs.get(node)
        status = f"[MATCHED: {matched}]" if matched else "[UNMATCHED]"
        print(f"\n[Render Node {i+1}] {status}")
        parent_label = f"{node.parent.type} | {node.parent.content}" if node.parent else "None"
        print(f"  parent: {parent_label}")
        print_semantic_tree(node)

    print("\n" + "=" * 50)
    # --- Build a GdocTree for each render node and print it ---
    print(f"\n--- GdocTree Output per Render Node ---")
 
    all_gdoc_trees = []
    # Each render node is a self-contained branch — give each its own
    # builder with a fresh cursor so indices are correct per-branch.
    # In production you'd thread a single cursor across all branches;
    # here we start each at 1 for readability.
    for i, render_node in enumerate(all_render_nodes):
        gdoc_builder = GdocTreeBuilder(start_index=1)
 
        # Build a minimal stub root so the builder has a parent to attach to
        class _RootToken:
            pass
        stub_root = EmbedTreeNode(_RootToken(), level=0)
        stub_root.type = "ROOT"
        stub_root.children = [render_node]
        # Temporarily reparent so the builder traversal works cleanly
        original_parent = render_node.parent
        render_node.parent = stub_root
 
        gdoc_root = gdoc_builder.build(stub_root, node_heading_pairs)
 
        # Restore original parent
        render_node.parent = original_parent
 
        all_gdoc_trees.append(gdoc_root)
 
        heading_label = render_node.content.strip()[:50] if render_node.content else f"Node {i}"
        print(f"\n[Branch {i+1}] '{heading_label}'")
        print(gdoc_root)
 
        requests = gdoc_builder.collect_all_requests(gdoc_root)
        print(f"  → {len(requests)} total requests generated")
        for j, req in enumerate(requests):
            req_type = list(req.keys())[0]
            print(f"    [{j+1}] {req_type}")



    

if __name__ == "__main__":
    # Change this to your actual file path
    test_render_to_gdocs("files/basic-text.pdf")