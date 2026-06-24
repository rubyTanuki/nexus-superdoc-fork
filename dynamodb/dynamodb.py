import boto3
from botocore.exceptions import ClientError
import os


def test():
    session = boto3.Session()
    credentials = session.get_credentials()
    current_credentials = credentials.get_frozen_credentials()
    
    # This will print which access key is being used to your logs
    #print(f"Using Access Key: {current_credentials.access_key}")


dynamodb = boto3.resource('dynamodb')
TABLE_NAME = os.getenv("DYNAMO_TABLE", "NEXUS-superdoc-id-store")#'NEXUS-superdoc-id-store'#docidstore'
table = dynamodb.Table(TABLE_NAME)

def save_course_docs(courseId, doc_ids):
    """Creates or completely overwrites a course entry with a list of doc IDs."""
    try:
        table.put_item(
            Item={
                'courseId': courseId,
                'docIds': doc_ids
            }
        )
        return True
    except ClientError as e:
        print(f"DynamoDB Put Error: {e}")
        return False

def append_to_course_docs(courseId, new_doc_ids):
    """Adds new document IDs to an existing list. Creates item if missing."""
    if not isinstance(new_doc_ids, list):
        new_doc_ids = [new_doc_ids] # Ensure it's a list for list_append

    try:
        table.update_item(
            Key={'courseId': courseId},
            UpdateExpression="SET docIds = list_append(if_not_exists(docIds, :empty_list), :new_docs)",
            ExpressionAttributeValues={
                ':new_docs': new_doc_ids,
                ':empty_list': []
            }
        )
        return True
    except ClientError as e:
        print(f"DynamoDB Update Error: {e}")
        return False

def fetch_all_course_docs(courseId):
    """Retrieves the list of document IDs for a given course ID."""
    try:
        response = table.get_item(
            Key={'courseId': courseId},
            ProjectionExpression='docIds'
        )
        item = response.get('Item')
        return item.get('docIds', []) if item else []
    except ClientError as e:
        print(f"DynamoDB Fetch Error: {e}")
        return []

if __name__ == '__main__': 
    #print(append_to_course_docs(courseId="prof-1302",new_doc_ids=['idfelfeff']))
    print(fetch_all_course_docs(courseId='prof-1302'))