#!/bin/bash

# Jekyll Setup Script for Safe RL Documentation Website
# This script sets up the Jekyll environment and builds the documentation site

set -e  # Exit on any error

echo "🚀 Setting up Jekyll Documentation Website for Safe RL Human-Robot Shared Control"
echo "============================================================================="

# Check if we're in the correct directory
if [ ! -f "_config.yml" ]; then
    echo "❌ Error: _config.yml not found. Please run this script from the docs/ directory."
    exit 1
fi

# Check if Ruby is installed
if ! command -v ruby &> /dev/null; then
    echo "❌ Ruby is not installed. Please install Ruby first:"
    echo "   - macOS: brew install ruby"
    echo "   - Ubuntu/Debian: sudo apt-get install ruby-full"
    echo "   - Windows: Download from https://rubyinstaller.org/"
    exit 1
fi

echo "✅ Ruby version: $(ruby --version)"

# Check if Bundler is installed, install if not
if ! command -v bundle &> /dev/null; then
    echo "📦 Installing Bundler..."
    gem install bundler
else
    echo "✅ Bundler is already installed: $(bundle --version)"
fi

# Install Jekyll and dependencies
echo "📦 Installing Jekyll and dependencies..."
bundle install

# Check if installation was successful
if [ $? -eq 0 ]; then
    echo "✅ Jekyll and dependencies installed successfully!"
else
    echo "❌ Failed to install dependencies. Please check the error messages above."
    exit 1
fi

# Create missing directories if they don't exist
echo "📁 Creating necessary directories..."
mkdir -p assets/images/results
mkdir -p assets/images/methodology  
mkdir -p assets/images/architecture
mkdir -p assets/fonts

# Check if MathJax is accessible (optional, for offline development)
echo "🔬 Checking MathJax accessibility..."
if curl -s --head https://cdnjs.cloudflare.com/ajax/libs/mathjax/3.2.0/es5/tex-mml-chtml.js | grep "200 OK" > /dev/null; then
    echo "✅ MathJax CDN is accessible"
else
    echo "⚠️  Warning: MathJax CDN may not be accessible. Math rendering might not work offline."
fi

# Build the Jekyll site
echo "🔨 Building Jekyll site..."
bundle exec jekyll build

if [ $? -eq 0 ]; then
    echo "✅ Site built successfully!"
    echo "📂 Built site is available in the _site/ directory"
else
    echo "❌ Site build failed. Please check the error messages above."
    exit 1
fi

# Function to check if port is in use
check_port() {
    local port=$1
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null ; then
        return 0  # Port is in use
    else
        return 1  # Port is free
    fi
}

# Find an available port (starting from 4000)
PORT=4000
while check_port $PORT; do
    ((PORT++))
    if [ $PORT -gt 4010 ]; then
        echo "❌ No available ports found between 4000-4010"
        exit 1
    fi
done

echo "🌐 Starting Jekyll development server on port $PORT..."
echo "📖 Documentation will be available at: http://localhost:$PORT"
echo ""
echo "🎯 Available pages:"
echo "   - Home: http://localhost:$PORT/"
echo "   - About: http://localhost:$PORT/pages/about.html"
echo "   - Methodology: http://localhost:$PORT/pages/methodology.html"
echo "   - Results: http://localhost:$PORT/pages/results.html"
echo "   - Conclusions: http://localhost:$PORT/pages/conclusion.html"
echo "   - Contact: http://localhost:$PORT/pages/contact.html"
echo ""
echo "💡 Tips:"
echo "   - Press Ctrl+C to stop the server"
echo "   - Changes to files will automatically rebuild the site"
echo "   - Check the terminal for build errors if pages don't update"
echo ""

# Start the Jekyll server with auto-reload
bundle exec jekyll serve --host 0.0.0.0 --port $PORT --watch --drafts --future --incremental

echo ""
echo "🔄 Jekyll server stopped."
echo "📝 To restart the server, run: bundle exec jekyll serve --port $PORT --watch"
echo ""
echo "🚀 Jekyll Documentation Website Setup Complete!"