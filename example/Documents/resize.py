import boto3
from PIL import Image
import io

s3_client = boto3.client('s3')

def resize_image(image_content):
    with Image.open(io.BytesIO(image_content)) as img:
        width, height = img.size
        img = img.resize((int(width*0.75), int(height*0.75)))
        buffer = io.BytesIO()
        img.save(buffer, 'JPEG') 
        return buffer.getvalue()

def lambda_handler(event, context):
    source_bucket = 'src'
    dest_bucket = 'dest'
    
    for record in event['Records']:
        key = record['s3']['object']['key']
        image_content = s3_client.get_object(Bucket=source_bucket, Key=key)['Body'].read()

        resized_image_content = resize_image(image_content)
        s3_client.put_object(Bucket=dest_bucket, Key=key, Body=resized_image_content)
    
    return {
        'statusCode': 200,
        'body': 'Images resized and stored in the destination bucket.'
    }

