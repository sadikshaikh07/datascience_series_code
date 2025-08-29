# Setup Guide 🛠️

Complete setup instructions for the AI Fundamentals Series.

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/username/ai-fundamentals-series
cd ai-fundamentals-series
```

### 2. Install Python Dependencies
```bash
# Install all required packages
pip install -r shared/requirements.txt

# Or install in a virtual environment (recommended)
python -m venv ai-fundamentals
source ai-fundamentals/bin/activate  # On Windows: ai-fundamentals\Scripts\activate
pip install -r shared/requirements.txt
```

### 3. Set Up Environment Variables
```bash
# Copy the example environment file
cp shared/.env.example .env

# Edit .env with your actual API keys
nano .env  # or use your preferred editor
```

### 4. Configure Your API Key
Add your OpenAI API key to the `.env` file:
```bash
OPENAI_API_KEY=sk-your-actual-api-key-here
OPENAI_MODEL=gpt-4
OPENAI_MAX_TOKENS=1000
OPENAI_TEMPERATURE=0.7
```

### 5. Test Your Setup
```bash
# Run a simple test
python 01-ai-agents/examples/demo_all_agents.py
```

## 🔧 Detailed Setup

### Python Version Requirements
- **Python 3.8+** required
- **Python 3.9+** recommended for best compatibility

Check your Python version:
```bash
python --version
```

### Virtual Environment Setup (Recommended)
```bash
# Create virtual environment
python -m venv ai-fundamentals

# Activate on Linux/Mac
source ai-fundamentals/bin/activate

# Activate on Windows
ai-fundamentals\Scripts\activate

# Install dependencies
pip install -r shared/requirements.txt
```

### OpenAI API Key Setup
1. Go to [OpenAI Platform](https://platform.openai.com/)
2. Create an account or sign in
3. Navigate to API Keys section
4. Create a new API key
5. Add to your `.env` file

### Alternative LLM Providers

#### Ollama (Local Models)
```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Download a model
ollama pull llama3.1

# Configure in .env
OPENAI_BASE_URL=http://localhost:11434/v1
OPENAI_MODEL=llama3.1
```

#### LocalAI
```bash
# Configure in .env
OPENAI_BASE_URL=http://localhost:8080/v1
OPENAI_MODEL=gpt-3.5-turbo
```

## 📦 Dependencies Explained

### Core Dependencies
- **openai**: Official OpenAI API client
- **pydantic**: Data validation and settings management
- **python-dotenv**: Environment variable management
- **requests**: HTTP library for API calls

### Optional Dependencies
- **tiktoken**: Token counting for OpenAI models
- **jupyter**: Interactive development environment
- **pandas**: Data manipulation and analysis
- **sqlite3**: Built-in database for examples

### Development Dependencies
- **pytest**: Testing framework
- **black**: Code formatting
- **flake8**: Code linting

## 🌐 Network Configuration

### Firewall and Proxy
If you're behind a corporate firewall:
```bash
# Set proxy in environment
export HTTP_PROXY=http://proxy.company.com:8080
export HTTPS_PROXY=http://proxy.company.com:8080

# Or configure in Python
import os
os.environ['HTTP_PROXY'] = 'http://proxy.company.com:8080'
os.environ['HTTPS_PROXY'] = 'http://proxy.company.com:8080'
```

### Rate Limiting
OpenAI API has rate limits:
- **Free tier**: 3 requests/minute
- **Paid tier**: Higher limits based on usage tier

Monitor your usage at [OpenAI Usage Dashboard](https://platform.openai.com/usage).

## 🐛 Troubleshooting

### Common Issues

#### "Module not found" errors
```bash
# Make sure you're in the right directory
pwd  # Should show .../ai-fundamentals-series

# Reinstall dependencies
pip install -r shared/requirements.txt

# Check Python path
python -c "import sys; print(sys.path)"
```

#### API Key errors
```bash
# Check if .env file exists and has correct format
cat .env

# Verify API key format (should start with sk-)
echo $OPENAI_API_KEY
```

#### Import path issues
```bash
# Run from the root directory
cd ai-fundamentals-series
python 01-ai-agents/examples/demo_all_agents.py
```

#### Rate limit exceeded
- Wait a few minutes before retrying
- Upgrade to paid OpenAI account
- Use local models with Ollama

### Getting Help
1. Check the [troubleshooting guide](troubleshooting.md)
2. Search existing [GitHub issues](https://github.com/username/ai-fundamentals-series/issues)
3. Create a new issue with error details
4. Join the community discussions

## 💻 IDE Setup

### VS Code (Recommended)
Install these extensions:
- Python (Microsoft)
- Pylance (Microsoft)  
- Python Docstring Generator
- autoDocstring

### PyCharm
1. Open project directory
2. Configure Python interpreter
3. Install Python plugin
4. Set up run configurations

### Jupyter Notebooks
```bash
# Install Jupyter
pip install jupyter

# Start Jupyter server
jupyter notebook

# Create new notebooks in any section directory
```

## 🚀 Advanced Configuration

### Environment Variables
```bash
# Model configuration
OPENAI_MODEL=gpt-4              # Model to use
OPENAI_MAX_TOKENS=1000          # Response length limit
OPENAI_TEMPERATURE=0.7          # Creativity level (0.0-2.0)
OPENAI_TOP_P=1.0               # Vocabulary diversity

# API configuration  
OPENAI_BASE_URL=                # Custom endpoint for compatible APIs
OPENAI_TIMEOUT=30              # Request timeout in seconds

# Logging and debugging
LOG_LEVEL=INFO                 # Logging verbosity
DEBUG_MODE=false               # Enable detailed debugging
```

### Custom Models
For using custom or fine-tuned models:
```bash
OPENAI_MODEL=your-custom-model-id
```

### Multiple API Keys
For different examples or rate limit management:
```bash
OPENAI_API_KEY_PRIMARY=sk-key1
OPENAI_API_KEY_SECONDARY=sk-key2
```

## ✅ Verification Checklist

- [ ] Python 3.8+ installed
- [ ] Virtual environment created and activated
- [ ] Dependencies installed successfully
- [ ] `.env` file created with API key
- [ ] Test script runs without errors
- [ ] Network connectivity to OpenAI API
- [ ] IDE configured for Python development

## 🔗 Next Steps

Once your setup is complete:
1. Start with [01-ai-agents/](../01-ai-agents/) for foundational concepts
2. Work through each section in order
3. Experiment with the examples
4. Build your own projects using the patterns

Need help? Check the [troubleshooting guide](troubleshooting.md) or open an issue!