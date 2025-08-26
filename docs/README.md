# Safe RL Documentation Website

This directory contains the Jekyll documentation website for the Safe Reinforcement Learning for Human-Robot Shared Control project.

## 🚀 Quick Start

### Prerequisites

- **Ruby** (version 2.7 or higher)
- **Bundler** gem
- **Git**

### Setup and Build

1. **Navigate to docs directory:**
   ```bash
   cd docs/
   ```

2. **Run the setup script:**
   ```bash
   ./setup.sh
   ```
   
   This script will:
   - Install Jekyll and all dependencies
   - Create necessary directories
   - Build the site
   - Start the development server

3. **Access the documentation:**
   - Open your browser to `http://localhost:4000`
   - The site will automatically rebuild when you make changes

### Manual Setup (Alternative)

If you prefer manual setup:

```bash
# Install dependencies
bundle install

# Build the site
bundle exec jekyll build

# Serve the site locally
bundle exec jekyll serve --watch
```

## 📁 Site Structure

```
docs/
├── _config.yml              # Jekyll configuration
├── Gemfile                  # Ruby dependencies
├── index.md                 # Home page
├── _layouts/                # Page layouts
│   ├── default.html         # Main layout template
│   └── post.html           # Blog post layout
├── _includes/               # Reusable components
│   ├── head.html           # HTML head section
│   ├── header.html         # Site header and navigation
│   └── footer.html         # Site footer
├── assets/                  # Static assets
│   ├── css/
│   │   └── style.scss      # Main stylesheet
│   ├── js/
│   │   └── main.js         # JavaScript functionality
│   └── images/             # Images and visualizations
│       ├── results/        # Results plots and charts
│       ├── methodology/    # Methodology diagrams
│       └── architecture/   # System architecture diagrams
├── pages/                   # Main documentation pages
│   ├── about.md            # About the project
│   ├── methodology.md      # Methodology and theory
│   ├── results.md          # Experimental results
│   ├── conclusion.md       # Conclusions and future work
│   └── contact.md          # Contact information
├── _posts/                 # Blog posts
│   └── 2024-01-01-project-overview.md
└── setup.sh               # Setup script
```

## 📖 Content Pages

### Core Documentation

- **[Home](index.md)**: Project overview, key results, and navigation
- **[About](pages/about.md)**: Motivation, problem statement, and contributions
- **[Methodology](pages/methodology.md)**: Mathematical formulation and implementation
- **[Results](pages/results.md)**: Comprehensive experimental analysis
- **[Conclusions](pages/conclusion.md)**: Key findings and future directions
- **[Contact](pages/contact.md)**: Team information and collaboration opportunities

### Features

- **Mathematical Notation**: LaTeX rendering with MathJax
- **Interactive Elements**: Search, table of contents, reading progress
- **Responsive Design**: Mobile-friendly layout
- **Publication Quality**: High-resolution plots and professional styling
- **Accessibility**: WCAG 2.1 compliant design

## 🎨 Customization

### Styling

The main stylesheet is in `assets/css/style.scss`. Key customization options:

- **Colors**: Modify color variables at the top of the file
- **Typography**: Font families and sizes
- **Layout**: Grid and spacing parameters
- **Components**: Styling for specific elements

### Content Updates

1. **Text Content**: Edit the markdown files in `pages/` and root directory
2. **Images**: Add new visualizations to `assets/images/`
3. **Navigation**: Update `_config.yml` header_pages section
4. **Metadata**: Modify site information in `_config.yml`

### Automatic Updates

The documentation can be automatically updated from analysis results using the integration script:

```python
from safe_rl_human_robot.src.utils.doc_generator import DocumentationGenerator

# Initialize generator
doc_gen = DocumentationGenerator()

# Generate documentation from results
doc_gen.generate_documentation(
    training_data=your_training_data,
    safety_data=your_safety_data,
    rebuild_site=True
)
```

## 🔧 Development

### Local Development

For active development:

```bash
# Start with auto-reload and drafts
bundle exec jekyll serve --watch --drafts --incremental

# Build only (no server)
bundle exec jekyll build

# Clean build files
bundle exec jekyll clean
```

### Adding New Pages

1. Create a new markdown file in `pages/` or root directory
2. Add front matter with layout and metadata
3. Update navigation in `_config.yml` if needed
4. Rebuild site to see changes

Example front matter:
```yaml
---
layout: default
title: "New Page Title"
permalink: /pages/new-page.html
---
```

### Adding Blog Posts

1. Create a new file in `_posts/` with format: `YYYY-MM-DD-title.md`
2. Add front matter with post metadata
3. Posts automatically appear in chronological order

## 📊 Integration with Analysis Pipeline

The documentation website integrates with the Phase 5 analysis pipeline:

### Automatic Plot Generation

The `doc_generator.py` utility can:
- Generate plots from training data
- Update results pages with latest metrics
- Rebuild the Jekyll site automatically

### Usage Example

```python
# Generate documentation from saved results
from safe_rl_human_robot.src.utils.doc_generator import generate_documentation_from_results

result = generate_documentation_from_results(
    results_dir="./results",
    docs_dir="./docs", 
    rebuild_site=True
)
```

## 🚀 Deployment

### GitHub Pages

If deploying to GitHub Pages:

1. Push the docs folder to your repository
2. Enable GitHub Pages in repository settings
3. Select "main" branch and "/docs" folder as source
4. Site will be available at `https://username.github.io/repository-name`

### Custom Server

For deployment on a custom server:

1. Build the site: `bundle exec jekyll build`
2. Copy the `_site` directory contents to your web server
3. Configure your web server to serve the static files

### Continuous Integration

Example GitHub Actions workflow for automatic deployment:

```yaml
# .github/workflows/deploy-docs.yml
name: Deploy Documentation

on:
  push:
    branches: [main]
    paths: ['docs/**']

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: ruby/setup-ruby@v1
        with:
          ruby-version: 3.0
      - name: Build Jekyll site
        run: |
          cd docs
          bundle install
          bundle exec jekyll build
      - name: Deploy to GitHub Pages
        uses: peaceiris/actions-gh-pages@v3
        with:
          github_token: ${{ secrets.GITHUB_TOKEN }}
          publish_dir: docs/_site
```

## 🐛 Troubleshooting

### Common Issues

**Jekyll not found:**
```bash
gem install jekyll bundler
```

**Bundle install fails:**
```bash
# Update RubyGems
gem update --system

# Install bundler
gem install bundler
```

**MathJax not rendering:**
- Check internet connection (MathJax loads from CDN)
- Verify MathJax configuration in `_includes/head.html`

**Images not displaying:**
- Check file paths are correct
- Ensure images exist in `assets/images/`
- Verify image file permissions

### Performance Issues

**Slow build times:**
```bash
# Use incremental builds
bundle exec jekyll serve --incremental

# Exclude unnecessary files
# Add to _config.yml exclude list
```

**Large image files:**
- Optimize images before adding to assets
- Use appropriate image formats (PNG for diagrams, JPEG for photos)
- Consider lazy loading for large images

## 📚 Resources

### Jekyll Documentation
- [Official Jekyll Docs](https://jekyllrb.com/docs/)
- [GitHub Pages Docs](https://docs.github.com/en/pages)
- [Liquid Template Language](https://shopify.github.io/liquid/)

### Markdown References
- [Markdown Guide](https://www.markdownguide.org/)
- [GitHub Flavored Markdown](https://github.github.com/gfm/)

### MathJax
- [MathJax Documentation](https://docs.mathjax.org/)
- [LaTeX Mathematical Notation](https://en.wikibooks.org/wiki/LaTeX/Mathematics)

## 🤝 Contributing

To contribute to the documentation:

1. Fork the repository
2. Create a feature branch for your changes
3. Make your edits to the markdown files
4. Test locally with `./setup.sh`
5. Submit a pull request with a clear description

### Style Guidelines

- Use clear, concise language
- Include code examples where appropriate
- Add alt text to all images
- Test mathematical notation renders correctly
- Ensure responsive design on mobile devices

---

For questions about the documentation website, please see the [Contact](pages/contact.md) page or open an issue in the repository.