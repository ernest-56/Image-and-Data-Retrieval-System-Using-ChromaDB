import gradio as gr
from models.clip_model import load_clip_model, generate_image_embedding, generate_text_embedding
from utils.chromadb_client import setup_chromadb, add_embeddings_to_collection

# Load model and set up ChromaDB
model, processor = load_clip_model()
collection = setup_chromadb()

def search_data(query=None, image_file=None):
    if query:
        query_embedding = generate_text_embedding(query, model, processor)
    elif image_file:
        query_embedding = generate_image_embedding(image_file, model, processor)
    else:
        return None, "Please provide either a query or an image."

    # Perform the search
    results = collection.query(query_embeddings=query_embedding, n_results=1)

    # Handle and display result
    metadata = results['metadatas'][0][0]
    if 'image' in metadata:
        result_image = Image.open(metadata['image'])
        return result_image, f"Matched Image with ID: {results['ids'][0][0]}"
    elif 'text' in metadata:
        return None, f"Matched Text: {metadata['text']}"

with gr.Blocks() as gr_interface:
    gr.Markdown("# Multi-modal Vector Search using ChromaDB")

    with gr.Row():
        with gr.Column():
            custom_query = gr.Textbox(placeholder="Enter text query here", label="Text Query")
            image_upload = gr.Image(type="pil", label="Upload an Image")

        with gr.Column():
            result_output = gr.Image(type="pil", label="Search Result")
            result_text = gr.Textbox(label="Result")

        submit_button = gr.Button("Submit Query")

    submit_button.click(fn=search_data, inputs=[custom_query, image_upload], outputs=[result_output, result_text])

gr_interface.launch()
