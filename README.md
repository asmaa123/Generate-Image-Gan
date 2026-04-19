# AI Image Generator

A powerful AI image generation system using Stable Diffusion v1.5 that creates stunning images from text prompts, fully runnable on CPU without requiring GPU resources.

## What It Does

- **Text-to-Image Generation**: Create beautiful images from text descriptions
- **CPU-Optimized**: Runs efficiently without GPU requirements
- **Interactive Interface**: User-friendly Streamlit web application
- **Creative AI**: Leverages Stable Diffusion v1.5 for high-quality outputs

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the web application
streamlit run app_streamlit_generate_image.py
```

## Project Structure

```
Generate-Image-Gan/
    app_streamlit_generate_image.py    # Main Streamlit application
    generate_image.py                  # Core image generation logic
    generate_image.ipynb              # Development notebook
    streamlit_generate.ipynb          # Streamlit integration
    requirements.txt                  # Python dependencies
```

## Technical Stack

### Core Technologies
- **AI Model**: Stable Diffusion v1.5 (Hugging Face Diffusers)
- **Framework**: Python with Streamlit
- **Image Processing**: PIL, NumPy
- **Web Interface**: Streamlit components

### Dependencies
```
torch>=1.9.0
diffusers>=0.10.0
transformers>=4.20.0
streamlit>=1.0.0
Pillow>=8.0.0
numpy>=1.20.0
```

## How to Use

### Web Application
1. **Launch App**: Run `streamlit run app_streamlit_generate_image.py`
2. **Enter Prompt**: Type your image description
3. **Generate**: Click generate to create your image
4. **Download**: Save your generated artwork

### Python Script
```python
from generate_image import generate_from_prompt

# Generate image from text
image = generate_from_prompt("A beautiful sunset over mountains")
image.save("output.png")
```

## Features

### Image Generation
- **Text Prompts**: Natural language to image conversion
- **High Quality**: Stable Diffusion v1.5 outputs
- **Customizable**: Adjustable generation parameters
- **Fast Processing**: Optimized for CPU execution

### User Interface
- **Intuitive Design**: Clean, user-friendly interface
- **Real-time Preview**: See generation progress
- **Parameter Controls**: Adjust settings for different styles
- **Export Options**: Download generated images

## Model Information

### Stable Diffusion v1.5
- **Architecture**: Latent Diffusion Model
- **Training**: Large-scale image-text dataset
- **Capabilities**: Text-to-image generation
- **Quality**: High-resolution, detailed outputs

### CPU Optimization
- **Memory Efficient**: Optimized for CPU execution
- **No GPU Required**: Accessible on any machine
- **Fast Inference**: Optimized processing pipeline
- **Resource Management**: Efficient memory usage

## Use Cases

### Creative Applications
- **Art Generation**: Create unique artwork
- **Design Concepts**: Visualize ideas quickly
- **Content Creation**: Generate images for media
- **Prototyping**: Visual design mockups

### Professional Use
- **Marketing Materials**: Create promotional visuals
- **Concept Art**: Develop creative concepts
- **Illustrations**: Generate custom illustrations
- **Presentations**: Add visual elements

## Configuration

### Generation Parameters
```python
# Adjustable parameters
prompt = "Your text description here"
num_inference_steps = 50  # Generation steps
guidance_scale = 7.5       # Prompt adherence
height = 512              # Image height
width = 512               # Image width
```

### Model Settings
- **Model Path**: Stable Diffusion v1.5 checkpoint
- **Scheduler**: DDIM scheduler for fast inference
- **Device**: CPU optimization settings
- **Memory**: Efficient memory management

## Performance

### CPU Performance
- **Generation Time**: ~30-60 seconds per image
- **Memory Usage**: ~4-8 GB RAM
- **Quality**: High-resolution outputs
- **Compatibility**: Works on standard CPUs

### Optimization Features
- **Memory Efficiency**: Minimal RAM usage
- **Speed**: Optimized inference pipeline
- **Quality**: Maintains high output quality
- **Stability**: Reliable generation process

## Future Enhancements

- **GPU Support**: Optional GPU acceleration
- **More Models**: Additional AI models
- **Batch Processing**: Generate multiple images
- **Advanced Controls**: More parameter options
- **Style Transfer**: Artistic style applications

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

This project is licensed under the MIT License.

## Acknowledgments

- **Hugging Face**: Diffusers library and models
- **Stability AI**: Stable Diffusion model
- **Streamlit**: Web application framework

---

**Start creating amazing AI art today!**
