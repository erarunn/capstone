# 🚀 Hugging Face Spaces Deployment Guide

## Step-by-Step Instructions

### 1. Get GROQ API Key
- Go to [console.groq.com](https://console.groq.com)
- Sign up/login and create an API key
- Save the key securely

### 2. Prepare GitHub Repository
```bash
# If you have code locally
git init
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/YOUR_USERNAME/youtube-video-qa-system.git
git push -u origin main
```

### 3. Create Hugging Face Space
1. Visit [huggingface.co/spaces](https://huggingface.co/spaces)
2. Click "Create new Space"
3. Configure:
   - **Owner**: Your username
   - **Space name**: `youtube-video-qa-system`
   - **SDK**: Streamlit
   - **Hardware**: CPU
   - **Visibility**: Public
   - **Repository**: Select your GitHub repo

### 4. Add API Key Secret
1. Go to Space Settings
2. Scroll to "Repository secrets"
3. Add new secret:
   - **Name**: `GROQ_API_KEY`
   - **Value**: Your GROQ API key

### 5. Deploy
- Hugging Face will automatically build your app
- Monitor progress in "Build logs" tab
- Your app will be live at: `https://huggingface.co/spaces/YOUR_USERNAME/youtube-video-qa-system`

## Required Files
- `app.py` - Main application
- `youtube_processor.py` - Processing logic
- `requirements.txt` - Dependencies
- `README.md` - Documentation

## Troubleshooting
- **Build fails**: Check requirements.txt and imports
- **API errors**: Verify GROQ_API_KEY secret
- **Memory issues**: Consider GPU upgrade for paid tier

## Your App URL
`https://huggingface.co/spaces/YOUR_USERNAME/youtube-video-qa-system` 