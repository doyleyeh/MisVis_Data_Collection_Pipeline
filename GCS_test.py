from google.cloud import storage

client = storage.Client()
 
# Print the Project ID
print(f"Project ID: {client.project}")

print("Listing buckets: ")
buckets = client.list_buckets()
for bucket in buckets:
    print(bucket)
