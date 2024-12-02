def preprocess(image, resize_fn):
    """Preprocess function for resizing and color conversion."""
    image = resize_fn(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def upload_image_to_s3(s3_client, bucket_name, s3_key, image):
    """Upload image to S3 from memory."""
    # 이미지 인코딩 (JPEG 포맷으로 변환)
    _, buffer = cv2.imencode('.jpg', image)
    try:
        # S3에 업로드
        s3_client.put_object(Bucket=bucket_name, Key=s3_key, Body=buffer.tobytes(), ContentType='image/jpeg')
        print(f"Uploaded image to s3://{bucket_name}/{s3_key}")
        return f"s3://{bucket_name}/{s3_key}"
    except Exception as e:
        print(f"Error uploading image to S3: {e}")
        return None