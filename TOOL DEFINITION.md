{
"name": "retrieve_image_background",
"description": "Retrieve grounded context, exact source (if it exists), and a strict evidence-based summary for an image using OpenRouter vision, SerpApi reverse image search, and Brave web search.",
"parameters": {
"type": "object",
"properties": {
"image": {
"type": "string",
"description": "Input image (URL or base64)"
},
"user_query": {
"type": "string",
"description": "Optional user question about the image"
}
},
"required": ["image"]
}
}
