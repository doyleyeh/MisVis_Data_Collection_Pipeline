{
"name": "retrieve_image_background",
"description": "Retrieve grounded context, exact source (if it exists), and a strict evidence-based summary for an image using OpenRouter vision, SerpApi reverse image search, and Brave web search.",
"parameters": {
"type": "object",
"properties": {
"image": {
"type": "string",
"description": "Input image as a public URL, local path, base64 string, or data URL"
},
"user_query": {
"type": "string",
"description": "Optional user question about the image"
},
"llm_summary_config": {
"type": "object",
"description": "Optional explicit overrides for the env-based LLM summary rewrite settings. If omitted, environment variables remain the baseline.",
"properties": {
"enable_llm_context_summary": {
"type": "boolean",
"description": "Master override for optional LLM summary rewriting"
},
"page_summary_enabled": {
"type": "boolean",
"description": "Override for candidate page summary rewriting"
},
"likely_context_enabled": {
"type": "boolean",
"description": "Override for likely_context rewriting"
},
"concise_overview_enabled": {
"type": "boolean",
"description": "Override for concise_overview rewriting"
},
"summary_model": {
"type": "string",
"description": "Override OpenRouter summary model name"
},
"timeout_seconds": {
"type": "integer",
"description": "Override timeout in seconds for optional summary rewrites"
}
}
}
},
"required": ["image"]
}
}
