import pymupdf
import pymupdf4llm

from io import BytesIO

import mistletoe
from mistletoe.block_token import BlockToken, Heading, Paragraph
from mistletoe.span_token import SpanToken, RawText

from src.models.tree_nodes import EmbedTreeNode, GdocTreeNode
from src.core.semantic_renderer import SemanticTreeBuilder
from src.core.gdocs_renderer import GdocTreeBuilder
from src.core.merge_algs import TreeEmbedder, SemanticReconciler, DB_Heading
from src.services.openai_client import OpenAIProcessor
from src.services.pinecone_client import VectorDBManager
from src.services.gdocs_client import GoogleDocsEditor

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
        stub_root = EmbedTreeNode(_RootToken(), 0)
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



def test_render_to_gdocs2(file_path):
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

    # --- Render all branches to Google Docs ---
    print(f"\n--- Rendering {len(all_render_nodes)} branch(es) to Google Doc ---")
    gdoc_editor = GoogleDocsEditor()
    gdoc_editor.render_trees(
        superdoc_id=DOCUMENT_ID,
        all_render_nodes=all_render_nodes,
        node_heading_pairs=node_heading_pairs
    )
    print(f"\n--- render_trees() complete ---")


def test_table_structure(file_path):
    """
    Isolates and prints everything about how a PDF's table flows through
    the pipeline: mistletoe token structure, EmbedTree, and GdocTree requests.
    """
    import pymupdf
    import pymupdf4llm
    import mistletoe

    from mistletoe import Document
    from mistletoe.span_token import SpanToken, add_token
    from src.models.tokens import InlineMath

    print("=" * 60)
    print("STEP 1: RAW MARKDOWN")
    print("=" * 60)
    doc = pymupdf.open(file_path)
    md_text = pymupdf4llm.to_markdown(doc, force_markdown=True)
    print(md_text)

    print("=" * 60)
    print("STEP 2: MISTLETOE TOKEN TREE (raw parse)")
    print("=" * 60)
    add_token(InlineMath)
    mistletoe_doc = mistletoe.Document(md_text)

    def print_mistletoe_tree(token, indent=0):
        prefix = "  " * indent
        cls = token.__class__.__name__
        content = ""
        if hasattr(token, 'content') and token.content:
            content = f" | content={repr(token.content[:60])}"
        children = getattr(token, 'children', None) or []
        print(f"{prefix}[{cls}]{content} ({len(children)} children)")
        for child in children:
            print_mistletoe_tree(child, indent + 1)

    print_mistletoe_tree(mistletoe_doc)

    print("=" * 60)
    print("STEP 3: TABLE TOKEN DEEP DIVE")
    print("=" * 60)

    def find_tables(token, path="root"):
        cls = token.__class__.__name__
        if cls == "Table":
            print(f"\nFound Table at: {path}")
            print(f"  children count (rows): {len(getattr(token, 'children', []))}")
            for r_idx, row in enumerate(getattr(token, 'children', [])):
                row_cls = row.__class__.__name__
                cells = getattr(row, 'children', [])
                print(f"  Row {r_idx} [{row_cls}] — {len(cells)} cells")
                for c_idx, cell in enumerate(cells):
                    cell_cls = cell.__class__.__name__
                    # Try to extract text from cell
                    def _text(t):
                        from mistletoe.span_token import RawText
                        if isinstance(t, RawText): return t.content
                        kids = getattr(t, 'children', None) or []
                        return " ".join(_text(k) for k in kids)
                    text = _text(cell)
                    print(f"    Cell [{c_idx}] [{cell_cls}] text={repr(text)}")
        for child in getattr(token, 'children', []) or []:
            find_tables(child, path + f">{child.__class__.__name__}")

    find_tables(mistletoe_doc)

    print("=" * 60)
    print("STEP 4: EMBED TREE (SemanticTreeBuilder output)")
    print("=" * 60)


    with SemanticTreeBuilder() as builder:
        nested_tree = builder.render(mistletoe_doc)

    def print_embed_tree(node, indent=0):
        prefix = "  " * indent
        content_preview = ""
        if node.content:
            content_preview = f" | {repr(node.content.strip()[:60])}"
        print(f"{prefix}[{node.type}] level={node.level}{content_preview} ({len(node.children)} children)")
        for child in node.children:
            print_embed_tree(child, indent + 1)

    print_embed_tree(nested_tree)

    print("=" * 60)
    print("STEP 5: GDOC REQUESTS FOR TABLE NODE")
    print("=" * 60)

    # Find the TABLE embed node
    def find_embed_tables(node, results=None):
        if results is None: results = []
        if node.type == "TABLE":
            results.append(node)
        for child in node.children:
            find_embed_tables(child, results)
        return results

    table_nodes = find_embed_tables(nested_tree)
    print(f"Found {len(table_nodes)} TABLE node(s) in EmbedTree")

    for i, table_node in enumerate(table_nodes):
        print(f"\n--- Table {i+1} ---")
        print(f"  content preview: {repr(table_node.content[:100])}")

        # Build a minimal stub tree with just this table node
        stub_root = object.__new__(EmbedTreeNode)
        stub_root.type = "ROOT"
        stub_root.node = None
        stub_root.level = 0
        stub_root.parent = None
        stub_root.children = [table_node]
        stub_root.embedding = None
        stub_root.mean_emb = None
        stub_root.is_custom_node = False
        stub_root.is_pruned = False
        stub_root.has_embedding = False
        stub_root.block_len = 0

        builder = GdocTreeBuilder(start_index=1)
        gdoc_root = builder.build(stub_root, {})

        requests = builder.collect_all_requests(gdoc_root)
        print(f"  Total requests generated: {len(requests)}")
        for j, req in enumerate(requests):
            req_type = list(req.keys())[0]
            details = req[req_type]
            print(f"  [{j+1}] {req_type}")
            if req_type == "insertTable":
                print(f"        rows={details['rows']} cols={details['columns']} at index={details['location']['index']}")
            elif req_type == "insertText":
                print(f"        text={repr(details['text'])} at index={details['location']['index']}")
            elif req_type == "updateTextStyle":
                r = details['range']
                print(f"        range=[{r['startIndex']},{r['endIndex']}] style={details['textStyle']}")

def print_detailed_tree(node, indent: str = "", is_last: bool = True) -> None:
    """
    Recursively prints a highly detailed, cascading terminal visualization 
    of either an EmbedTreeNode or a GdocTreeNode structure, fully expanding
    paragraphs into their individual nested inline style and raw-text children.
    """
    if node is None:
        return

    # Determine Unicode tree branches for clean layout alignment
    marker = "└── " if is_last else "├── "
    next_indent = indent + ("    " if is_last else "│   ")

    # 1. Handle Class Type Identification
    node_type = getattr(node, "type", "Unknown")
    
    # Extract the underlying data item depending on the wrapper type
    if hasattr(node, "node") and hasattr(node.node, "node"):
        # GdocTreeNode -> EmbedTreeNode -> Mistletoe Token
        mistletoe_token = node.node.node
    elif hasattr(node, "node"):
        # EmbedTreeNode -> Mistletoe Token
        mistletoe_token = node.node
    else:
        mistletoe_token = node

    token_class = mistletoe_token.__class__.__name__ if mistletoe_token else "Container"

    # 2. Gather Node Attributes / Flags
    extra_details = []
    if hasattr(node, "level") and node.level is not None:
        extra_details.append(f"Lvl: {node.level}")
    if getattr(node, "matched_heading", None):
        extra_details.append(f"MATCH: '{node.matched_heading}'")
    if hasattr(node, "requests") and node.requests:
        extra_details.append(f"Reqs: {len(node.requests)}")

    details_str = f" [{', '.join(extra_details)}]" if extra_details else ""

    # 3. Handle Content Snippets Safely
    # If it's an explicit text leaf or math leaf, capture its content directly
    raw_content = getattr(mistletoe_token, "content", "")
    if not raw_content and hasattr(node, "content"):
        raw_content = node.content
        
    content_snippet = repr(raw_content.strip()[:60]) if (raw_content and raw_content.strip()) else ""

    # Print the current node branch row
    print(f"{indent}{marker}[{node_type} ({token_class})]{details_str} {content_snippet}")

    # 4. Process Children Arrays (Cascading downward)
    # Collect both structural tree children AND inline Mistletoe span children
    children_to_process = []
    
    # Add structural block children (from EmbedTreeNode / GdocTreeNode layout)
    if hasattr(node, "children") and node.children:
        children_to_process.extend(node.children)
        


    child_count = len(children_to_process)
    for i, child in enumerate(children_to_process):
        print_detailed_tree(child, next_indent, is_last=(i == child_count - 1))



def print_mistletoe_tree(token, indent=0):
    prefix = "  " * indent
    cls = token.__class__.__name__
    content = ""
    
    if cls == "List":
        list_type = "ORDERED_LIST" if token.start is not None else "UNORDERED_LIST"
        print(f"{prefix}[{list_type}] start={token.start} ({len(token.children)} children)")
    elif cls == "ListItem":
        print(f"{prefix}[LIST_ITEM] leader={token.leader} loose={token.loose}")
    else:
        if hasattr(token, 'content') and token.content:
            content = f" | content={repr(token.content[:60])}"
        children = getattr(token, 'children', None) or []
        print(f"{prefix}[{cls}]{content} ({len(children)} children)")

    for child in getattr(token, 'children', None) or []:
        print_mistletoe_tree(child, indent + 1)



def test_show_structure(file_path):
    """
    Diagnostic runner that processes a PDF file through the parser 
    and displays the cascading structural tree in the terminal.
    """
    print(f"\n" + "=" * 60)
    print(f"--- Visual Structural Audit for: {file_path} ---")
    print(f"=" * 60)
 
    # 1. Convert PDF to Markdown
    import pymupdf
    import pymupdf4llm
    import mistletoe
    
    doc = pymupdf.open(file_path)
    md_text = pymupdf4llm.to_markdown(doc, force_markdown=True)
    
    print('\n[Step 1] Extracted Raw Markdown Snippet:')
    print("-" * 40)
    # Print just the first 500 characters to keep stdout clear
    print(md_text[:500] + ("\n... [Truncated for preview] ..." if len(md_text) > 500 else ""))
    print("-" * 40)
 
    # 2. Build the Initial Mistletoe AST and Semantic Tree
    # Ensure mistletoe.Document.insert_plugin(InlineMath) happened somewhere globally
    mistletoe_doc = mistletoe.Document(md_text)
    
    print("\n[Step 2] Building Semantic Stack Tree...")
    with SemanticTreeBuilder() as builder:
        nested_tree = builder.render(mistletoe_doc)
 
    gdoc_builder = GdocTreeBuilder(start_index=1)

    gdoc_tree = gdoc_builder.build(nested_tree,{})
    print("\n[Step 3] Detailed Structural Output:")
    print("-" * 40)
    print_gdoc_tree(gdoc_tree)
    print("-" * 40)
    print(f"\n--- Diagnostic Audit Complete ---\n")


def check_math_in_pdf(file_path: str, preview_chars: int = 3000) -> str:
    """
    Extracts markdown from a PDF and checks for inline math ($...$).
    
    Args:
        file_path: Path to the PDF file.
        preview_chars: How many characters of markdown to preview.
    
    Returns:
        The full extracted markdown string.
    """
    import pymupdf
    import pymupdf4llm

    doc = pymupdf.open(file_path)
    md = pymupdf4llm.to_markdown(doc, force_markdown=True)

    dollar_count = md.count('$')
    print(f"Total '$' signs found: {dollar_count}")
    print(f"InlineMath will {'FIRE' if dollar_count >= 2 else 'NOT fire'} on this PDF\n")
    print("=" * 60)
    print(f"MARKDOWN PREVIEW (first {preview_chars} chars)")
    print("=" * 60)
    print(md[:preview_chars])

    return md



def print_gdoc_tree(gdoc_root: GdocTreeNode, indent: int = 0) -> None:
    """
    Prints the GdocTree with each node's type, content preview,
    and the Google Docs requests it will generate.
    """
    prefix = "  " * indent
    node_type = getattr(gdoc_root, 'type', '?')
    content = getattr(gdoc_root, 'content', '') or ''
    content_preview = repr(content.strip()[:50]) if content.strip() else ''

    print(f"{prefix}[{node_type}] {content_preview}")

    # Print requests attached to this node
    requests = getattr(gdoc_root, 'requests', []) or []
    table_fill = getattr(gdoc_root, '_table_fill', []) or []

    for req in requests:
        req_type = list(req.keys())[0]
        details = req[req_type]
        _print_request(req_type, details, prefix + "  ")

    if table_fill:
        print(f"{prefix}  [TABLE_FILL]")
        for req in table_fill:
            req_type = list(req.keys())[0]
            details = req[req_type]
            _print_request(req_type, details, prefix + "    ")

    for child in getattr(gdoc_root, 'children', []) or []:
        print_gdoc_tree(child, indent + 1)


def _print_request(req_type: str, details: dict, prefix: str) -> None:
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
        print(f"{prefix}→ updateTextStyle  @ [{r.get('startIndex','?')},{r.get('endIndex','?')}] {style}")

    elif req_type == "updateParagraphStyle":
        r = details.get('range', {})
        style = details.get('paragraphStyle', {})
        named = style.get('namedStyleType', '')
        indent_start = style.get('indentStart', {}).get('magnitude', '')
        print(f"{prefix}→ updateParaStyle  @ [{r.get('startIndex','?')},{r.get('endIndex','?')}] {named} indent={indent_start}")

    elif req_type == "createParagraphBullets":
        r = details.get('range', {})
        preset = details.get('bulletPreset', '?')
        print(f"{prefix}→ createBullets    @ [{r.get('startIndex','?')},{r.get('endIndex','?')}] {preset}")

    else:
        print(f"{prefix}→ {req_type}")



if __name__ == "__main__":
    # Change this to your actual file path
    test_render_to_gdocs2("files/basic-text.pdf")
    #test_show_structure("files/basic-text.pdf")