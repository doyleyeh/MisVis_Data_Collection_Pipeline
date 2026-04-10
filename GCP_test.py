from google.cloud import storage

client = storage.Client()
 
print("Listing buckets: ")
buckets = client.list_buckets()
for bucket in buckets:
    print(bucket)
