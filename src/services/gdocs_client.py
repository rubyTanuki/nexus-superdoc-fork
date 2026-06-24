import os.path
import json
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google.oauth2 import service_account
from googleapiclient.discovery import build
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone, IndexModel, ServerlessSpec
from itertools import chain
from diskcache import Index

from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build, Resource
from googleapiclient.errors import HttpError
from langchain_core.documents import Document

from models.tree_nodes import EmbedTreeNode
from core.gdocs_renderer import GdocTreeBuilder
from io import BytesIO

from collections import defaultdict

from dynamodb.dynamodb import append_to_course_docs, fetch_all_course_docs

# If modifying these SCOPES, delete the file token.json.
SCOPES = ["https://www.googleapis.com/auth/documents","https://www.googleapis.com/auth/drive.file"] # Use full scope like 'https://www.googleapis.com/auth/documents' for write operations

MIN_BLOCK_LEN = 2
#Hella bad practice, but works for maintaining a buffer between named ranges
MIN_RANGE_BUFFER = 3

class GoogleDocsAPI:
    """
    Base class for Google API interaction. 
    Handles authentication and service building for Docs and Drive
    """

    def __init__(self, credentials_file='credentials.json', token_file='token.json'):
        """Initializes the API wrapper with paths to credential and token files."""
        self.credentials_file = credentials_file
        self.token_file = token_file
        (self.doc_service,self.drive_service) = self.authenticate()
        #self.doc = None
    
    def authenticate(self):
        """
        Handles the OAuth2 authentication flow. 
        It prioritizes existing tokens, handles memory-only refreshes (to stay Lambda-friendly),
        and initializes the Docs (v1) and Drive (v3) service objects.
        """
    
        creds = None
        
        if os.path.exists(self.token_file):
            # 1. Read the file into memory first
            with open(self.token_file, 'r') as f:
                token_data = json.load(f)
            
            # 2. Use 'from_authorized_user_info' to prevent the library 
            # from trying to manage (and write to) the file on disk.
            creds = Credentials.from_authorized_user_info(token_data, SCOPES)
        
        # 3. Handle token refresh
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                print("Refreshing expired Google token in memory...")
                try:
                    # This refreshes the 'creds' object but DOES NOT save to disk
                    creds.refresh(Request())
                except Exception as e:
                    print(f"Failed to refresh token: {e}")
                    raise Exception("Google Refresh Token is invalid or revoked.")
            else:
                # In Lambda, we cannot run flow.run_local_server(). 
                # If we reach this block, it means token.json is missing or totally invalid.
                raise FileNotFoundError("Valid token.json not found. Please generate it locally first.")
    
        print(f"Credentials valid: {creds.valid}")
        
        # Build services
        doc_service = build('docs', 'v1', credentials=creds)
        drive_service = build('drive', 'v3', credentials=creds)
        
        return (doc_service, drive_service)
        

        
class GoogleDocsEditor(GoogleDocsAPI):
    """
    Extended class providing high-level editing capabilities.
    Manages document structure, headings, named ranges, and batch updates.
    """
    def __init__(self):
        """Initializes the editor by calling the base API authentication."""
        super().__init__()

    def get_text_in_indices_from_doc_obj(self,start_index:int,end_index:int):
        """
        Extracts raw text given a start_index and end_index within the current document object.
        Iterates through document elements and slices text runs based on UTF-16 indices.
        """
        extracted = []
        content = self.doc.get('body').get('content', [])

        for element in content:
            el_start = element.get('startIndex')
            el_end = element.get('endIndex')

            if el_start is not None and el_end is not None:
                if el_start < end_index and el_end > start_index:
                    if 'paragraph' in element:
                        for part in element['paragraph']['elements']:
                            if 'textRun' in part:
                                text = part['textRun']['content']
                                p_start = part.get('startIndex')

                                rel_start = max(0, start_index - p_start)
                                rel_end = min(len(text), end_index - p_start)

                                if rel_start < rel_end:
                                    extracted.append(text[rel_start:rel_end])
        return "".join(extracted)
    
        
    def get_text_in_range_from_doc_obj(self,heading:str):
        """
        Extracts raw text from a specific named range (heading) within the current document object.
        """
        
        named_range = self.find_named_range(heading)
        if not named_range: 
            return ""
        start_index = named_range['namedRanges'][0]['ranges'][0]['startIndex']
        end_index = named_range['namedRanges'][0]['ranges'][0]['endIndex']
        return self.get_text_in_indices_from_doc_obj(start_index=start_index,end_index=end_index)
        
    def create_google_doc(self, name:str, courseid:str):
        """
        Creates a new Google Doc, logs the ID to DynamoDB, and sets global 
        'anyone with link can edit' permissions via the Drive API.
        """
        try:
            # Create the document using the Docs API
            #print(f"Service Account Email: {self.doc_service._http.credentials.service_account_email}")
            response = self.doc_service.documents().create(body={'title': name}).execute()
            document_id = response.get('documentId')
            append_to_course_docs(courseId=courseid,new_doc_ids=[document_id])#self.idstore.push_new_docId(courseid=courseid,docname=name,docid=document_id)
            print(f'Created Document ID: {document_id}')

            # Set the document to be editable by anyone with the link
            permission_result = self.drive_service.permissions().create(
                fileId=document_id,
                body={
                    'role': 'writer',
                    'type': 'anyone'
                }
            ).execute()
            print(permission_result)
            print("Document sharing permissions updated.")
            print(document_id)

            return response

        except Exception as err:
            print(f'Error creating document: {err}')
            return None
    
    def get_docids(self,courseid:str): 
        """Retrieves a list of all document IDs associated with a specific course from DynamoDB."""
        return fetch_all_course_docs(courseId=courseid)
      
    def get_document_structure(self, document_id):
        """
        Fetches the complete JSON representation (structure) of a Google Doc.
        This is required before performing range-based lookups or mutations.
        """
        try:
            ##print("CALLED DOCUMENT STRUCTURE")
            self.document_id = document_id
            self.doc = self.doc_service.documents().get(documentId=document_id).execute()
            return self.doc
           # print(f"CALLED DOCUMETN STRUCTURE:{self.doc}")
        except Exception as e:
            print(f"Error fetching document: {e}")
            return None
    
    def text_utf16_len(self,text:str):
        """
        Calculates the length of text in UTF-16 code units.
        Google Docs API indices are based on UTF-16, not standard Python UTF-8 strings.
        """ 
        return len((text).encode("utf-16-le"))//2
        
    
            
    def batch_update(self, requests):
        """
        Executes a list of formatting or structural requests in a single API call.
        """
        if not requests or len(requests) == 0:
            return

        try:
            result = self.doc_service.documents().batchUpdate(
                documentId=self.document_id,
                body={'requests': requests}
            ).execute()
            return True

        except Exception as e:
            print(f"Error in batchUpdate: {e}")
            # Binary search to find the exact failing request
            if len(requests) == 1:
                print(f"Failing request: {requests[0]}")
                return False

            mid = len(requests) // 2
            print(f"Bisecting {len(requests)} requests — trying first half ({mid} requests)...")
            first_ok = self.batch_update(requests[:mid])
            if first_ok:
                print(f"First half succeeded — problem is in second half:")
                self.batch_update(requests[mid:])
            else:
                print(f"Problem is in first half:")
                self.batch_update(requests[:mid])

            return False
    
    
  
    def find_named_range(self,heading:str)->dict|None:
        """Looks up and returns the metadata for a specific 'namedRange' in the document."""
        document = self.doc
        named_ranges = document.get("namedRanges",{})
        for range_name in named_ranges.keys():
            #print(f'-{range_name}')
            if range_name==heading: 
                return named_ranges[heading] 
        return None

    def create_heading(self,new_heading:str): 
        """
        Inserts new heading text, applies HEADING_2 style and bolding, 
        and wraps the text in a Named Range for future referencing.
        """
        (_,startIndex,_) = self.find_insertion_point()
        named_range = self.find_named_range(heading=new_heading)
        if named_range: 
            raise Exception(f"Heading: {new_heading} already exists")
        
        #newHeadingLen = len((new_heading+":\n\n").encode("utf-16-le"))//2
        #endIndex = startIndex+newHeadingLen
        
        protected_text = f"{new_heading}:"
        padding = "\n"
        full_text = protected_text + padding
        
        # Calculate indices based on specific parts
        protected_len = self.text_utf16_len(protected_text)
        total_len = self.text_utf16_len(full_text)
        
        endIndex_for_range = startIndex + protected_len
        endIndex_for_formatting = startIndex + total_len
        
        requests = [
        {
            'insertText': {
                'location': {
                    'index': startIndex
                },
                    'text': full_text#new_heading+":\n\n"
            }
        },
        # Create a new named range for the updated heading
        {
            'createNamedRange': {
                'name': new_heading,
                'range': {
                    'startIndex': startIndex,
                    'endIndex': endIndex_for_formatting
                }
            }
        },
        {
            'updateParagraphStyle': {
                'paragraphStyle': {'namedStyleType': 'HEADING_1'},
                'range': {
                    'startIndex': startIndex,
                    'endIndex': endIndex_for_formatting
                },
                'fields': 'namedStyleType'
            }
        },
        {
            'updateTextStyle': {
                'textStyle': {'bold': True},
                'range': {
                    'startIndex': startIndex,
                    'endIndex': endIndex_for_range
                },
                'fields': 'bold'
            }
        }
    ]
        self.batch_update(requests=requests)
        return (startIndex,endIndex_for_formatting)
    
    
    def create_headings(self,headings:list[str]):
        """
        Batch-processes a list of strings, creating new headings for any that don't exist
        and returning the start/end indices for all.

        --refractor to make more efficent in future(batch update)
        """ 
        ranges = []
        processed = {}
        for heading in headings:
            nm_range = self.find_named_range(heading)
            startIndex = -1
            endIndex = -1
            if heading in processed: 
                startIndex = processed[heading][0]
                endIndex = processed[heading][1]
            elif not nm_range:
                rn = self.create_heading(heading)
                processed[heading] = rn
                startIndex = rn[0]
                endIndex = rn[1]
                continue 
            else: 
                startIndex = nm_range['namedRanges'][0]['ranges'][0]['startIndex']
                endIndex = nm_range['namedRanges'][0]['ranges'][0]['endIndex']

            ranges.append((startIndex,endIndex))
        return ranges

    def batch_create_headings_and_named_ranges(self, headings_to_apply: list[dict]):
        """
        Takes a list of dictionaries containing heading text and indices,
        and applies them to the Google Doc in a single batched API call.
        
        Args:
            headings_to_apply: A list of dicts formatted like:
                               [{'heading': 'Biology', 'startIndex': 450}, ...]
        """
        if not headings_to_apply:
            return

        #CRITICAL: Sort by startIndex in REVERSE order.
        # This prevents text shifts from breaking our index calculations
        headings_sorted = sorted(headings_to_apply, key=lambda x: x['startIndex'], reverse=True)
        
        all_requests = []
        
        for item in headings_sorted:
            new_heading = item['heading']
            start_idx = item['startIndex']
            
            # Quick duplicate check
            named_range = self.find_named_range(heading=new_heading)
            if named_range: 
                print(f"[WARN] Heading: '{new_heading}' already exists. Skipping.")
                continue
            
            #text_to_insert = new_heading + ":\n\n"
            
            # Using UTF-16 length calculation logic
            #new_heading_len = len(text_to_insert.encode("utf-16-le")) // 2
            #end_idx = start_idx + new_heading_len
            
            protected_text = f"{new_heading}:"
            full_text = protected_text + "\n\n"
            protected_len = self.text_utf16_len(protected_text)

            # Append the isolated requests for this specific heading
            all_requests.extend([
                {
                    'insertText': {
                        'location': {
                            'index': start_idx
                        },
                        'text': full_text
                    }
                },
                {
                    'createNamedRange': {
                        'name': new_heading,
                        'range': {
                            'startIndex': start_idx,
                            'endIndex': start_idx+protected_len#end_idx
                        }
                    }
                },
                {
                    'updateParagraphStyle': {
                        'paragraphStyle': {'namedStyleType': 'HEADING_2'},
                        'range': {
                            'startIndex': start_idx,
                            'endIndex': start_idx + protected_len#end_idx
                        },
                        'fields': 'namedStyleType'
                    }
                },
                {
                    'updateTextStyle': {
                        'textStyle': {'bold': True},
                        'range': {
                            'startIndex': start_idx,
                            'endIndex': start_idx + protected_len#end_idx
                        },
                        'fields': 'bold'
                    }
                }
            ])
            
        #Fire them all in a single payload
        if all_requests:
            print(f"Batch updating Google Docs with {len(headings_to_apply)} new headings...")
            self.batch_update(requests=all_requests)           
        
    def delete_heading(self,old_heading:str):
        """Removes both the text and the metadata associated with a specific heading.""" 
        named_range = self.find_named_range(heading=old_heading)
        if not named_range: 
            raise Exception(f"Heading: {old_heading} does not exist")
        print(named_range)
        startIndex = named_range['namedRanges'][0]['ranges'][0]['startIndex']
        endIndex = named_range['namedRanges'][0]['ranges'][0]['endIndex']
        oldHeadingLen = len((old_heading).encode("utf-16-le"))//2
        requests = [
        # Delete the old heading text from the document
        {
            'deleteContentRange': {
                'range': {
                    'startIndex': startIndex,
                    'endIndex': startIndex + oldHeadingLen
                }
            }
        },
        # Remove the old named range
        {
            'deleteNamedRange': {
                'name': old_heading
            }
        },
    ]
        self.batch_update(requests=requests)


    def update_heading(self,old_heading:str,new_heading:str):
        """
        Renames a heading by deleting the old text and inserting the new text,
        then recreating the Named Range with adjusted indices.
        """ 
        named_range = self.find_named_range(heading=old_heading)
        if not named_range: 
            raise Exception(f"Heading: {old_heading} does not exist")
        print(named_range)
        startIndex = named_range['namedRanges'][0]['ranges'][0]['startIndex']
        endIndex = named_range['namedRanges'][0]['ranges'][0]['endIndex']
        oldHeadingLen = len((old_heading).encode("utf-16-le"))//2
        newHeadingLen = len((new_heading).encode("utf-16-le"))//2
        requests = [
        # Delete the old heading text from the document
        {
            'deleteContentRange': {
                'range': {
                    'startIndex': startIndex,
                    'endIndex': startIndex + oldHeadingLen
                }
            }
        },
        # Insert the new heading text at the same position
        {
            'insertText': {
                'location': {
                    'index': startIndex
                },
                'text': new_heading
            }
        },
        # Remove the old named range
        {
            'deleteNamedRange': {
                'name': old_heading
            }
        },
        # Create a new named range for the updated heading
        {
            'createNamedRange': {
                'name': new_heading,
                'range': {
                    'startIndex': startIndex,
                    'endIndex': endIndex-1+(newHeadingLen-oldHeadingLen)-MIN_RANGE_BUFFER
                }
            }
        },
    ]
        self.batch_update(requests=requests)
        #print(f"Named Range: {named_range}")
        
            
    def find_insertion_point(self, target_heading=None):
        """
        Determines where in the document to insert content, if provided a target_heading, returns the end of that heading content. 
        Returns (startIndex, endIndex, status_code). Defaults to the end of the doc.
        """
        document = self.doc
        body = document.get('body', {})
        content = body.get('content', [])
        named_ranges = document.get("namedRanges",{})
        for range_name in named_ranges.keys():
            print(f"- {range_name}")
        #print(f"Body:{body}")
        # If no target heading specified, append to the very end
        if not (target_heading):
            print(f"Printing to {content[-1].get('endIndex') - 1}")
            return (content[-1].get('endIndex'),content[-1].get('endIndex') - 1, -2)  # Adjust for 0-based indexing
        #If the heading is non-null, and is new(doesn't exist in document)
        try: 
            named_ranges[target_heading]
        except KeyError as e: 
            return (content[-1].get('endIndex'),content[-1].get('endIndex') - 1, -1)

        # Search for the specific heading
        named_range_data = named_ranges[target_heading]
        for range_group in named_range_data["namedRanges"]:
            for range_info in range_group["ranges"]:
                return (range_info["startIndex"],range_info["endIndex"],0)
            
        # If heading not found, append to end
        print(f"Heading '{target_heading}' not found. Appending to end.")
        print(f"Printing to {content[-1].get('endIndex') - 1}")
        return (content[-1].get('endIndex'),content[-1].get('endIndex') - 1,-1)


    def catch_skips(self) -> list[dict]:
        """
        Scans all named ranges (anchors) and reads the in-memory text inside the 
        gaps between them to identify unmanaged text or new heading markers.
        """
        document = self.doc
        named_ranges = document.get("namedRanges", {})
        
        if not named_ranges:
            print("No named ranges found in document.")
            return []

        # Sort named ranges purely by where they appear in the document
        sorted_items = sorted(
            named_ranges.items(),
            key=lambda item: item[1].get("namedRanges", [{}])[0]
                                   .get("ranges", [{}])[0]
                                   .get("startIndex", 0)
        ) 
        
        gaps = []
        
        # 1. Catch gaps between consecutive named ranges
        for i in range(1, len(sorted_items)):
            prev_ranges = sorted_items[i-1][1].get("namedRanges", [{}])[0].get("ranges", [{}])[0]
            curr_ranges = sorted_items[i][1].get("namedRanges", [{}])[0].get("ranges", [{}])[0]
            
            prev_end_idx = prev_ranges.get("endIndex", 0)
            curr_start_idx = curr_ranges.get("startIndex", 0)
            
            diff = curr_start_idx - prev_end_idx
            
            if diff > MIN_BLOCK_LEN: 
                # Slicing from in-memory self.doc is practically zero overhead
                gap_text = self.get_text_in_indices_from_doc_obj(prev_end_idx, curr_start_idx)
                gaps.append({
                    'startIndex': prev_end_idx,
                    'endIndex': curr_start_idx,
                    'text': gap_text,
                    'preceded_by_heading': sorted_items[i-1][0] # Name of the section owning this text
                })

        # 2. Check the trailing gap after the absolute last named range
        if sorted_items: 
            last_range = sorted_items[-1][1].get("namedRanges", [{}])[0].get("ranges", [{}])[0]
            last_range_end = last_range.get("endIndex", 0)

            doc_body = document.get("body", {})
            doc_content = doc_body.get("content", [])

            if doc_content: 
                doc_end_index = doc_content[-1].get("endIndex", 0)
                usable_doc_end = doc_end_index - 1
                final_diff = usable_doc_end - last_range_end
                
                if final_diff > MIN_BLOCK_LEN: 
                    gap_text = self.get_text_in_indices_from_doc_obj(last_range_end, usable_doc_end)
                    gaps.append({
                        'startIndex': last_range_end, 
                        'endIndex': usable_doc_end,
                        'text': gap_text,
                        'preceded_by_heading': sorted_items[-1][0]
                    })
                    
        return gaps

    def find_new_headings_in_gaps(self, gaps: list[dict]) -> list[dict]:
        """
        Scans extracted gap text for a heading pattern (e.g., lines ending with ':') 
        and calculates their true global document index.
        """
        import re
        new_headings = []
        
        # Matches lines that end with a colon (ignoring trailing spaces)
        # e.g., "Neural Networks:" or "<Heading>:"
        pattern = re.compile(r'^(.+?):\s*$', re.MULTILINE)
        
        for gap in gaps:
            for match in pattern.finditer(gap['text']):
                heading_text = match.group(1).strip()
                
                # Crucial step: Find match start character index relative to the gap string
                char_offset = match.start()
                
                # Convert character offset to a true global document UTF-16 index
                # We extract text up to the match location to accurately count UTF-16 spacing
                text_before_match = gap['text'][:char_offset]
                utf16_offset = self.text_utf16_len(text_before_match)
                
                global_start_index = gap['startIndex'] + utf16_offset
                
                new_headings.append({
                    'heading': heading_text,
                    'startIndex': global_start_index
                })
                
        return new_headings

    def mutate_named_ranges(self,document_id:str):
        """
        Scans all named ranges and bridges gaps between them. 
        If there is extra whitespace or content between ranges, it deletes and 
        recreates the range to ensure full document coverage for semantic syncing.
        """
        #self.get_document_structure(document_id=document_id)
        document = self.doc
        named_ranges = document.get("namedRanges",{})
        sorted_items = sorted(named_ranges.items(),key=lambda item: item[1].get("namedRanges", [{}])[0]
                               .get("ranges", [{}])[0]
                               .get("endIndex", 0)) 
        print(sorted_items)
        requests= []
        for i in range(1,len(sorted_items)):
            prev_ranges = sorted_items[i-1][1]\
                        .get("namedRanges",[{}])[0]\
                        .get("ranges",[{}])[0]
            curr_ranges = sorted_items[i][1]\
                        .get("namedRanges",[{}])[0]\
                        .get("ranges",[{}])[0]
            prevEndIdx = prev_ranges.get("endIndex",0)
            currStartIdx = curr_ranges.get("startIndex",0)
            #heading = sorted_items[i][0]
            diff = currStartIdx - prevEndIdx
            print(diff)
            if((diff)>self.text_utf16_len('\n')): 
                print("hit")
                requests.append(
                    {
                        'deleteNamedRange': {
                            'name': sorted_items[i-1][0]#prev_heading
                        }
                    }
                    )
                requests.append(# Create a new named range for the updated heading
                    {
                        'createNamedRange': {
                            'name': sorted_items[i-1][0],
                            'range': {
                                'startIndex': prev_ranges.get("startIndex",0),
                                'endIndex': prevEndIdx-1+(diff)-MIN_RANGE_BUFFER
                            }
                        }
                    }
                )
        #print(requests,self.text_utf16_len('\n'))
        self.batch_update(requests=requests) 
        self.get_document_structure(document_id=document_id) 
        #named_ranges = document.get("namedRanges",{})
    

    def render_trees(self, superdoc_id: str, all_render_nodes: list[EmbedTreeNode], node_heading_pairs: dict):
        """
        Renders a list of EmbedTreeNode branches into a Google Doc using GdocTreeBuilder.
        1. Ensures headings exist in the Doc.
        2. Builds GdocTree for each branch using the new renderer.
        3. Fires main requests for each branch, anchored to the correct heading via named ranges.
        """
        print(f"Connecting to Google Doc: {superdoc_id}")
        self.get_document_structure(document_id=superdoc_id)

        # Create headings in reverse so doc indices stay stable during insertion
        headings = [node.content for node in all_render_nodes]
        headings_reversed = list(reversed(headings))
        print(f"Received headings: {headings}")
        self.create_headings(headings_reversed)
        self.get_document_structure(document_id=superdoc_id)

        # Collect named range end indices for each heading (insertion points)
        heading_ranges = {}
        for heading in headings:
            named_range = self.find_named_range(heading)
            if named_range:
                heading_ranges[heading] = {
                    'startIndex': named_range['namedRanges'][0]['ranges'][0]['startIndex'],
                    'endIndex':   named_range['namedRanges'][0]['ranges'][0]['endIndex']
                }
            else:
                raise ValueError(f"Heading '{heading}' not found after creation. This should never happen.")
                # Fallback to end of doc if named range missing
                #doc_content = self.doc.get('body', {}).get('content', [])
                #end = doc_content[-1].get('endIndex', 1) - 1
                #heading_ranges[heading] = {'startIndex': end, 'endIndex': end}

        # Sort branches bottom-to-top so insertions don't corrupt higher indices
        all_render_nodes_sorted = sorted(
            all_render_nodes,
            key=lambda n: heading_ranges[n.content]['startIndex'],
            reverse=True
        )

        all_main_requests = []
        #all_table_fills = []

        for node in all_render_nodes_sorted:
            heading = node.content
            insert_at = heading_ranges[heading]['endIndex']
            start_index = heading_ranges[heading]['startIndex']

            builder = GdocTreeBuilder(start_index=insert_at)
            gdoc_root = builder.build(embed_root=node, matched_nodes=node_heading_pairs)
            main_requests, table_fills, _ = builder.collect_all_requests(gdoc_root)

            # Keep track of text inserted, but DO NOT expand named range to cover it
            all_main_requests.extend(main_requests)

            # FIXED: Recreate named range anchored ONLY to the label string length
            protected_len = self.text_utf16_len(f"{heading}:")
            
            all_main_requests.extend([
                {'deleteNamedRange': {'name': heading}},
                {
                    'createNamedRange': {
                        'name': heading,
                        'range': {
                            'startIndex': start_index,
                            'endIndex': start_index + protected_len
                        }
                    }
                }
            ])
            
        if all_main_requests:
            self.batch_update(requests=all_main_requests)

        #if all_table_fills:
            #self.batch_update(all_table_fills)
            #print("Table fill batchUpdate complete.")

        print("render_trees() complete.")

    def _collect_table_nodes(self, embed_nodes: list) -> list:
        """DFS collect all TABLE EmbedTreeNodes across a list of root nodes."""
        results = []
        def _dfs(node):
            if node.type == "TABLE":
                results.append(node)
            for child in node.children:
                _dfs(child)
        for root in embed_nodes:
            _dfs(root)
        return results

    def clear_document(self, superdoc_id: str):
        doc = self.doc_service.documents().get(documentId=superdoc_id).execute()
        body = doc.get('body', {}).get('content', [])
        end_index = body[-1].get('endIndex', 1)
        if end_index > 2:
            self.document_id = superdoc_id
            self.batch_update([{
                'deleteContentRange': {
                    'range': {'startIndex': 1, 'endIndex': end_index - 1}
                }
            }])

def main():
    """Main entry point for local debugging."""
    gde = GoogleDocsEditor()
    gde.get_document_structure(document_id="1Q1whz1kFN9wj1_mamWgaDbKh7przNmc5owdOSNovC04")
    gde.append_italic_text("This is a test of the italic text insertion function.")
    pass


if __name__ == "__main__":
    main()
